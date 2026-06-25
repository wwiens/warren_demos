# ============================================================
# Demo 04 - File 1 of 2: A2A Server
# Fraud Detection Agent exposed via the A2A Protocol
# ------------------------------------------------------------
# Run this FIRST in a separate terminal:
#   pip install fastapi uvicorn python-dotenv
#   python a2a_server.py
# ============================================================
#
# WHAT IS A2A?
# ------------
# A2A (Agent-to-Agent) is an open protocol that lets AI agents
# communicate with each other over HTTP — regardless of which
# framework or company built them. Think of it like a handshake
# agreement between agents: "Here's what I can do, here's how
# to call me, and here's how to prove you're allowed to."
#
# This file creates a *server-side* agent — it waits for
# requests from other agents (like the AutoGen client in
# autogen_client.py) and responds with a fraud risk assessment.
#
# ARCHITECTURE OVERVIEW:
#
#   [AutoGen Client] ──(HTTP POST)──► [This FastAPI Server]
#        │                                     │
#        │  1. GET /.well-known/agent.json      │
#        │◄────────── Agent Card ──────────────│
#        │                                     │
#        │  2. POST /tasks/send + Bearer token  │
#        │────────── Task Payload ────────────►│
#        │◄────────── Task Result ─────────────│
#
# ============================================================

import os
import uuid
from datetime import datetime

# FastAPI is a modern Python web framework for building APIs.
# We import the core app class plus helpers for HTTP errors and request headers.
from fastapi import FastAPI, HTTPException, Header

# Pydantic lets us define the *shape* of expected request data.
# FastAPI uses these models to automatically validate incoming JSON.
from pydantic import BaseModel

# python-dotenv reads key=value pairs from a .env file and loads
# them as environment variables. This keeps secrets (like API keys)
# out of source code.
from dotenv import load_dotenv

# Load environment variables from .env (e.g., A2A_SECRET_KEY=my-secret)
load_dotenv()

# Read the shared secret from the environment. This token is how the
# server knows that a caller is authorized to send it tasks.
# The fallback "demo-secret-123" is only for local experimentation —
# always use a strong random value in any real deployment.
A2A_SECRET_KEY = os.environ.get("A2A_SECRET_KEY", "demo-secret-123")

# Create the FastAPI application. The title appears in the auto-generated
# API documentation at http://localhost:8000/docs.
app = FastAPI(title="FraudDetectorAgent A2A Server")


# -----------------------------------------------------------------
# CONCEPT 1: The Agent Card
# -----------------------------------------------------------------
# The Agent Card is the heart of A2A discovery. It is a JSON object
# served at the *standardized* URL path /.well-known/agent.json.
#
# Any A2A-compatible client can fetch this URL to learn:
#   - What this agent is called and what it does
#   - Where to send tasks (the "url" field)
#   - Which communication styles it supports (streaming, push, etc.)
#   - Which specific tasks ("skills") it can perform
#   - What authentication method callers must use
#
# By standardizing this discovery mechanism, A2A makes agents
# *interoperable* — an AutoGen agent, a LangChain agent, or a
# custom agent can all find and use this server the same way.
# -----------------------------------------------------------------
AGENT_CARD = {
    # Human-readable identity
    "name": "FraudDetectorAgent",
    "description": "Analyzes payment transactions and returns a fraud risk assessment.",

    # The base URL where this agent is reachable
    "url": "http://localhost:8000",
    "version": "1.0.0",

    # Capabilities tell clients which A2A features this agent supports.
    # "streaming": False means the agent returns a single response, not
    #   a real-time stream of tokens.
    # "pushNotifications": False means callers must poll for results;
    #   the agent won't proactively push updates.
    "capabilities": {
        "streaming": False,
        "pushNotifications": False,
    },

    # Skills describe the *specific tasks* this agent can perform.
    # A single agent can expose multiple skills. Clients use this list
    # to understand what inputs to send and what outputs to expect.
    "skills": [
        {
            "id": "fraud_risk_check",           # Machine-readable identifier
            "name": "Fraud Risk Check",          # Human-readable label
            "description": "Checks a transaction description for fraud risk indicators.",
            "inputModes": ["text"],              # This skill accepts plain text input
            "outputModes": ["text"],             # It returns plain text output
        }
    ],

    # Authentication tells clients which security schemes this agent requires.
    # "Bearer" means the caller must include an Authorization header like:
    #   Authorization: Bearer <secret-token>
    "authentication": {
        "schemes": ["Bearer"]
    }
}


# -----------------------------------------------------------------
# CONCEPT 2: The A2A Task Schema
# -----------------------------------------------------------------
# In the A2A protocol, a *Task* is the standard unit of work.
# Every interaction follows the same shape:
#   - A client sends a Task containing a message
#   - The server processes it and returns a Task result
#
# This consistent structure means any A2A-compatible tool can
# parse requests and responses without knowing the agent's domain.
#
# We use Pydantic BaseModel classes to describe the expected JSON
# shape. FastAPI will automatically:
#   1. Parse incoming JSON into these Python objects
#   2. Return a 422 error if the JSON doesn't match the schema
#   3. Show the schema in the /docs UI
# -----------------------------------------------------------------

class TaskMessage(BaseModel):
    """
    Represents a single message inside a Task.

    role  — who sent this message: "user" for the caller, "agent" for
            the server's reply. Mirrors the convention used in chat APIs.
    parts — a list of content blocks. A2A supports multiple content
            types (text, files, data), so "parts" is a list of dicts
            rather than a single string. Each dict has at minimum a
            "text" key for plain-text content.
    """
    role: str
    parts: list[dict]


class TaskRequest(BaseModel):
    """
    Represents a complete A2A Task sent by a client.

    id      — a unique identifier (typically a UUID) that the client
              generates. The server echoes it back so the client can
              match responses to requests, which matters in async flows.
    message — the TaskMessage containing the actual content to process.
    """
    id: str
    message: TaskMessage


# -----------------------------------------------------------------
# Endpoint 1: Agent Card Discovery
# -----------------------------------------------------------------
# The @app.get decorator registers this function as an HTTP GET
# handler. When any client sends GET /.well-known/agent.json, FastAPI
# calls this function and returns the AGENT_CARD dict as JSON.
#
# The path /.well-known/ is a web standard (RFC 5785) for publishing
# metadata about a server. A2A uses it so clients always know where
# to look for the Agent Card without any prior configuration.
# -----------------------------------------------------------------
@app.get("/.well-known/agent.json")
def get_agent_card():
    """Return this agent's capability manifest to any caller."""
    return AGENT_CARD


# -----------------------------------------------------------------
# Endpoint 2: Task Execution
# -----------------------------------------------------------------
# CONCEPT 3: Authentication with Bearer Tokens
#
# A Bearer token is a simple but effective security mechanism:
#   - The server and client share a secret string in advance
#     (here, via the A2A_SECRET_KEY environment variable)
#   - Every request must include the header:
#       Authorization: Bearer <secret>
#   - The server checks it before doing any work
#
# Why Bearer tokens? They are stateless (no session to manage),
# easy to rotate (just change the env var), and universally
# supported by HTTP clients. In production you'd use short-lived
# tokens from an OAuth2 server rather than a long-lived shared
# secret, but the pattern is the same.
# -----------------------------------------------------------------
@app.post("/tasks/send")
def handle_task(task: TaskRequest, authorization: str = Header(None)):
    """
    Receive an A2A Task, authenticate the caller, run fraud detection,
    and return the result in A2A Task format.

    Parameters
    ----------
    task          : Parsed from the JSON request body by FastAPI/Pydantic.
    authorization : Read automatically from the HTTP Authorization header.
                    FastAPI maps Header(None) to the incoming header value,
                    defaulting to None if the header is absent.
    """

    # ------------------------------------------------------------------
    # Step 1: Authenticate the caller
    # ------------------------------------------------------------------
    # Build the string we expect to see in the Authorization header.
    # The "Bearer " prefix is part of the HTTP spec for token auth.
    expected = f"Bearer {A2A_SECRET_KEY}"

    # Reject the request immediately if the header is missing or wrong.
    # HTTP 401 Unauthorized is the correct status code for auth failures.
    # We do this before any real work to avoid leaking information or
    # wasting compute on unauthorized callers.
    if not authorization or authorization != expected:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: invalid or missing Bearer token"
        )

    # ------------------------------------------------------------------
    # Step 2: Extract the transaction text from the A2A Task message
    # ------------------------------------------------------------------
    # A Task message contains a list of "parts". We iterate through them
    # looking for the first text part. This approach is forward-compatible:
    # future versions of the client could include additional parts (e.g.,
    # structured data or attachments) without breaking this extraction.
    transaction_text = ""
    for part in task.message.parts:
        if "text" in part:
            transaction_text = part["text"]
            break  # We only need the first text part for this skill

    # ------------------------------------------------------------------
    # Step 3: Run fraud detection logic (simulated rule-based check)
    # ------------------------------------------------------------------
    # In a real system this would call an ML model or a fraud-scoring API.
    # Here we use a simple keyword search to demonstrate the concept without
    # requiring additional infrastructure. Students should think about:
    #   - What are the limitations of keyword-based fraud detection?
    #   - How would you replace this with a trained classifier?
    #   - What data would you log for auditing purposes?

    # Keywords that correlate with high-risk transaction patterns
    HIGH_RISK_SIGNALS = [
        "foreign",    # Cross-border transfers carry higher risk
        "overnight",  # Off-hours activity is a common fraud indicator
        "unusual",    # Caller has already flagged it as anomalous
        "large",      # High-value transactions merit extra scrutiny
        "wire",       # Wire transfers are harder to reverse than card payments
        "2 am",       # Late-night activity is a common fraud indicator
        "3 am",       # Same reasoning as above
        "unknown",    # Unknown counterparties are a red flag
    ]

    # Determine risk level: HIGH if any keyword appears, LOW otherwise.
    # .lower() ensures the match is case-insensitive.
    risk_level = (
        "HIGH"
        if any(signal in transaction_text.lower() for signal in HIGH_RISK_SIGNALS)
        else "LOW"
    )

    # Confidence scores are also simulated here. A real model would
    # return a probability from its output layer.
    confidence = 0.93 if risk_level == "HIGH" else 0.88

    # Build a human-readable summary that the calling agent can relay
    # to a human reviewer or use in its own reasoning.
    summary = (
        f"Fraud Risk Assessment: {risk_level}\n"
        f"Confidence: {confidence:.0%}\n"
        # List which specific signals triggered the assessment (or "None")
        f"Signals detected: {[s for s in HIGH_RISK_SIGNALS if s in transaction_text.lower()] or 'None'}\n"
        # ISO 8601 UTC timestamp for audit trail purposes
        f"Assessed at: {datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}"
    )

    # ------------------------------------------------------------------
    # Step 4: Return the result in A2A Task format
    # ------------------------------------------------------------------
    # The response mirrors the request structure so clients have a
    # consistent format to parse. Key fields:
    #   id     — echo the original task ID so the client can match it
    #   status — "completed" signals the task finished successfully.
    #            Other states include "failed" and "in_progress".
    #   result — contains a message with role "agent" and the output parts
    return {
        "id": task.id,
        "status": {"state": "completed"},
        "result": {
            "message": {
                "role": "agent",       # Server-side responses use "agent" role
                "parts": [{"text": summary}]
            }
        }
    }


# -----------------------------------------------------------------
# Entry point: start the web server
# -----------------------------------------------------------------
# When this script is run directly (python a2a_server.py), uvicorn
# launches the FastAPI app on port 8000.
#
# host="0.0.0.0" means "accept connections from any network interface"
# which is required if other machines (or containers) need to reach
# this server. For local development only, you could use "127.0.0.1".
# -----------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("Starting FraudDetectorAgent A2A Server on http://localhost:8000")
    print(f"Agent Card available at: http://localhost:8000/.well-known/agent.json")
    print(f"Task endpoint:           http://localhost:8000/tasks/send")
    print(f"Interactive API docs:    http://localhost:8000/docs")
    print(f"Auth: Bearer token required (set A2A_SECRET_KEY in .env)\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
