# Step 1: Import required libraries
#
# No external packages are required for this demo — every import is part of
# Python's standard library. This makes the file self-contained and runnable
# in any Python 3.x environment without a virtual environment or pip install.
#
# json                 — standard library module for serialising Python objects
#                        to compact JSON byte strings and deserialising them back;
#                        used here as one of the two wire formats agents exchange
# xml.etree.ElementTree
#                      — standard library XML parser and builder; provides
#                        ET.Element, ET.SubElement, ET.tostring, and ET.fromstring
#                        which are the only primitives needed to build and parse
#                        the minimal XML payloads in this demo
# List, Dict, Any, Optional
#                      — typing helpers that annotate function signatures so IDEs
#                        and type checkers can verify argument types at development
#                        time; they carry no runtime cost

import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional


# Step 2: Define format negotiation logic
#
# negotiate() is the heart of the demo. It takes the capability advertisements
# from both agents (which formats each can speak) and their ordered preference
# lists, then returns the single format string they should use — or None if
# there is no overlap at all.
#
# The algorithm is deliberately simple so the mechanics are easy to follow:
#   1. Compute the intersection of the two supported-format sets.
#   2. Walk the sender's preference list first; return the first hit.
#   3. If the sender's preferences are all outside the overlap, walk the
#      receiver's preference list and return the first hit.
#   4. Fall back to an arbitrary member of the overlap if neither preference
#      list yields a match (should not happen in practice with well-formed data).
#   5. Return None when the intersection is empty — no negotiation is possible.
#
# Checking sender preference first is an intentional design choice: the sender
# already has the payload in memory encoded in one of its preferred formats, so
# honouring the sender's preference minimises the chance that a conversion step
# is needed and therefore reduces latency and the risk of data-loss bugs in
# the converter.
#
# Parameters:
#   sender_supported   — list of format strings the sender can produce
#   receiver_supported — list of format strings the receiver can consume
#   sender_pref        — sender's formats in preference order (most preferred first)
#   receiver_pref      — receiver's formats in preference order

def negotiate(sender_supported: List[str], receiver_supported: List[str],
              sender_pref: List[str], receiver_pref: List[str]) -> Optional[str]:

    # Convert to upper-case sets for case-insensitive comparison.
    # Using sets for the overlap calculation gives O(min(n,m)) performance
    # regardless of how long the capability lists grow.
    s = {x.upper() for x in sender_supported}
    r = {x.upper() for x in receiver_supported}
    overlap = s & r

    if not overlap:
        return None

    # Honour sender's preference first, then receiver's.
    for p in [x.upper() for x in sender_pref]:
        if p in overlap:
            return p
    for p in [x.upper() for x in receiver_pref]:
        if p in overlap:
            return p

    # Guaranteed fallback: overlap is non-empty, so this always returns a value.
    return next(iter(overlap))


# Step 3: Create JSON and XML converter helpers
#
# These six functions form a minimal, demo-grade codec layer. They are kept
# intentionally simple (flat key-value structures only) so the focus stays on
# the negotiation and routing logic rather than on serialisation edge cases.
#
# dict_to_json_bytes    — encodes a Python dict to a UTF-8 JSON byte string;
#                         separators=(",", ":") removes all whitespace for a
#                         compact wire representation
# json_bytes_to_dict    — decodes a UTF-8 JSON byte string back to a dict
# dict_to_xml_bytes     — encodes a Python dict to an XML byte string where the
#                         root element is <message> and each key becomes a child
#                         element whose text content is the string form of the value
# xml_bytes_to_dict     — parses an XML byte string back to a dict; numeric strings
#                         are automatically coerced to int or float so round-trip
#                         fidelity is preserved for simple numeric fields
# convert_bytes         — dispatches to the correct pair of functions above based
#                         on the src_fmt and dst_fmt strings; raises ValueError for
#                         unsupported formats rather than silently dropping data

def dict_to_json_bytes(d: Dict) -> bytes:
    return json.dumps(d, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

def json_bytes_to_dict(b: bytes) -> Dict:
    return json.loads(b.decode("utf-8"))

def dict_to_xml_bytes(d: Dict) -> bytes:
    root = ET.Element("message")
    for k, v in d.items():
        child = ET.SubElement(root, str(k))
        child.text = str(v)
    return ET.tostring(root, encoding="utf-8")

def xml_bytes_to_dict(b: bytes) -> Dict:
    root = ET.fromstring(b.decode("utf-8"))
    out = {}
    for child in root:
        text = child.text or ""
        try:
            num = float(text)
            # Preserve integer semantics for whole-number values (e.g. 101.0 → 101)
            # so that a JSON round-trip does not change the Python type of an id field.
            out[child.tag] = int(num) if num.is_integer() else num
        except ValueError:
            out[child.tag] = text
    return out

def convert_bytes(src_fmt: str, dst_fmt: str, payload: bytes) -> bytes:
    src_fmt, dst_fmt = src_fmt.upper(), dst_fmt.upper()

    # Short-circuit: if source and destination are the same format, return the
    # payload unchanged without touching any codec logic. This avoids a
    # pointless decode → re-encode cycle that could alter whitespace or key order.
    if src_fmt == dst_fmt:
        return payload

    # Decode the incoming payload to an intermediate Python dict. All format
    # converters pass through this neutral representation, which means adding
    # support for a third format (e.g. MessagePack) only requires two new
    # functions (encode/decode) rather than N² converter pairs.
    if src_fmt == "JSON":
        d = json_bytes_to_dict(payload)
    elif src_fmt == "XML":
        d = xml_bytes_to_dict(payload)
    else:
        raise ValueError(f"Unsupported source format: {src_fmt}")

    if dst_fmt == "JSON":
        return dict_to_json_bytes(d)
    elif dst_fmt == "XML":
        return dict_to_xml_bytes(d)
    else:
        raise ValueError(f"Unsupported destination format: {dst_fmt}")


# Step 4: Define the Agent class
#
# Agent is a minimal model of a message-passing participant. It holds the
# agent's name and its format capabilities, and exposes two communication
# methods: encode() to produce a wire payload, and send() which runs the full
# NEGOTIATE → CONVERT → DELIVER (or FALLBACK) pipeline in a single call.
#
# Keeping all four pipeline stages inside send() makes the execution trace easy
# to follow in the printed output: each stage prints a labelled line so a reader
# can map the console output directly back to the algorithm described above.
#
# Constructor parameters:
#   name      — human-readable label used in log output to identify which agent
#               is speaking or receiving
#   supported — complete list of formats this agent can handle; used in the
#               intersection calculation inside negotiate()
#   pref      — ordered preference list; the first entry is the most preferred
#               format and will be selected whenever it falls inside the overlap

class Agent:
    def __init__(self, name: str, supported: List[str], pref: List[str]):
        self.name = name
        self.supported = supported
        self.pref = pref

    def encode(self, fmt: str, message: Dict[str, Any]) -> bytes:
        # encode() converts the application-level dict to bytes in the requested
        # format. It is called by send() to produce the pre-negotiation payload,
        # i.e. the payload as the sender would have produced it natively before
        # any format agreement is reached.
        if fmt.upper() == "JSON":
            return dict_to_json_bytes(message)
        elif fmt.upper() == "XML":
            return dict_to_xml_bytes(message)
        else:
            raise ValueError(f"Unsupported encode format: {fmt}")

    def receive(self, payload: bytes, fmt: str) -> bool:
        # receive() simulates the receiver accepting or rejecting a delivery.
        # In a real system this would deserialise the payload and hand it to
        # application logic; here it simply verifies that the agreed format is
        # in the receiver's supported list, which should always be true after a
        # successful negotiation but is checked explicitly for auditability.
        return fmt.upper() in [x.upper() for x in self.supported]

    def send(self, receiver: "Agent", message: Dict[str, Any], src_format: str) -> Dict[str, Any]:

        # --- STAGE 1: NEGOTIATE ---
        # Ask the negotiator which format both sides can agree on. The result
        # is None if there is no overlap, which triggers the FALLBACK path.
        agreed = negotiate(self.supported, receiver.supported, self.pref, receiver.pref)
        print(f"NEGOTIATE: sender={self.name}, receiver={receiver.name}, "
              f"sender_supported={self.supported}, receiver_supported={receiver.supported}, "
              f"agreed={agreed}")

        if not agreed:
            # --- STAGE 4 (FALLBACK) ---
            # No shared format exists. Rather than raising an exception (which
            # would crash the workflow), the method returns a structured error
            # dict. This allows the caller to log the failure, route the message
            # to a human operator, or retry with a different receiver — all
            # without catching an unexpected exception.
            print("FALLBACK: No common format. Action=abort_and_log")
            return {"ok": False, "reason": "no_common_format"}

        # --- STAGE 2: ENCODE (sender's native format) ---
        # Build the payload in the sender's chosen format before any conversion.
        # Capturing this "before" state lets the demo print both sides of the
        # transformation so the format difference is visible in the console.
        payload_before = self.encode(src_format, message)

        # --- STAGE 3: CONVERT (only if the agreed format differs) ---
        # If the sender's format already matches the agreed format, skip the
        # conversion entirely to avoid a pointless re-encode. When conversion is
        # needed, convert_bytes handles the intermediate dict representation.
        needs_conv = src_format.upper() != agreed.upper()
        if needs_conv:
            print(f"CONVERT: {src_format.upper()} -> {agreed.upper()}")
            payload_after = convert_bytes(src_format, agreed, payload_before)
        else:
            payload_after = payload_before

        # Print both payloads so the format difference is immediately visible.
        # The try/except keeps the demo robust if either payload cannot be
        # decoded to a UTF-8 string (e.g. if a binary format were added later).
        try:
            print(f"PAYLOAD BEFORE ({src_format.upper()}):")
            print(payload_before.decode("utf-8"))
            print(f"PAYLOAD AFTER  ({agreed.upper()}):")
            print(payload_after.decode("utf-8"))
        except Exception:
            pass

        # --- STAGE 4: DELIVER ---
        # Set the content-type header that would accompany the payload in an
        # HTTP or message-bus delivery. Passing the agreed format to receiver()
        # lets the receiver confirm it can handle the content before committing
        # to process it — a pattern borrowed from HTTP content negotiation.
        content_type = "application/json" if agreed.upper() == "JSON" else "application/xml"
        ok = receiver.receive(payload_after, agreed)
        print(f"DELIVER: content_type={content_type}, ok={ok}")

        return {"ok": ok, "content_type": content_type, "agreed_format": agreed}


# Step 5: Define agents, shared message payload, and run all scenarios
#
# The three scenario blocks below correspond to the three variations described
# in the demo instructions. Running all scenarios in sequence from a single
# __main__ block means the entire demo is self-contained: one file, one command,
# three observable outcomes. Each scenario is separated by a header line so the
# console output maps cleanly to the scenario name.
#
# Scenario 1 — XML → JSON (mismatch auto-heals):
#   Sender A supports XML and JSON, prefers XML.
#   Receiver B supports JSON only.
#   Negotiation agrees on JSON; the XML payload is auto-converted before delivery.
#
# Scenario 2 — JSON → XML conversion:
#   Sender A now prefers JSON; Receiver B supports XML only.
#   Negotiation agrees on XML; the JSON payload is auto-converted before delivery.
#
# Scenario 3 — No overlap (safe fallback):
#   Sender C supports XML only; Receiver D supports JSON only.
#   Negotiation finds no shared format and returns the fallback error dict without
#   raising an exception, leaving the system in a known good state.
#
# message is the shared application-level payload used in all three scenarios.
# Using the same dict across scenarios makes the before/after payload output
# directly comparable and highlights that the data content is preserved exactly
# through each conversion.

if __name__ == "__main__":

    message = {"id": 101, "title": "Quarterly Report", "amount": 123.45}

    # --- Scenario 1: XML → JSON (sender prefers XML, receiver supports JSON only) ---
    print("=== Scenario 1: XML -> JSON (mismatch auto-heals) ===")
    A = Agent("A", ["XML", "JSON"], ["XML", "JSON"])  # supports both, prefers XML
    B = Agent("B", ["JSON"],        ["JSON"])          # JSON only
    result1 = A.send(B, message, src_format="XML")
    print("RESULT:", result1)

    print()

    # --- Scenario 2: JSON → XML (sender prefers JSON, receiver supports XML only) ---
    print("=== Scenario 2: JSON -> XML conversion ===")
    A2 = Agent("A", ["XML", "JSON"], ["JSON", "XML"])  # supports both, now prefers JSON
    B2 = Agent("B", ["XML"],         ["XML"])           # XML only
    result2 = A2.send(B2, message, src_format="JSON")
    print("RESULT:", result2)

    print()

    # --- Scenario 3: No overlap — safe fallback ---
    # C and D share no common format. negotiate() returns None, send() returns
    # an error dict, and no conversion or delivery is attempted. The workflow
    # can continue normally after logging the failure.
    print("=== Scenario 3: No overlap (safe fallback) ===")
    C = Agent("C", ["XML"],  ["XML"])   # XML only
    D = Agent("D", ["JSON"], ["JSON"])  # JSON only
    result3 = C.send(D, message, src_format="XML")
    print("RESULT:", result3)
