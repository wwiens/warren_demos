# ============================================================
# Demo 04 - File 2 of 2: AutoGen A2A Client
# AutoGen agent that calls the FraudDetectorAgent via A2A
# ------------------------------------------------------------
# Run this SECOND (after a2a_server.py is running):
#   pip install pyautogen requests python-dotenv
#   python autogen_client.py
# ============================================================
#
# WHAT IS AUTOGEN?
# ----------------
# AutoGen is a Microsoft open-source framework for building
# *multi-agent* AI applications. Instead of one AI doing
# everything, AutoGen lets you create teams of specialized
# agents that talk to each other to solve a problem.
#
# In this demo, two AutoGen agents collaborate:
#   - PaymentReviewAgent  : an LLM-powered assistant that
#                           decides *when* to call external tools
#                           and interprets their results
#   - ComplianceOfficer   : a "user proxy" that initiates tasks,
#                           executes tool calls, and controls
#                           when the conversation ends
#
# WHAT IS A2A (AGENT-TO-AGENT)?
# ------------------------------
# A2A is an open protocol for AI agents to call *each other*
# over HTTP. This demo shows how an AutoGen agent can reach out
# to a separately running A2A server (a2a_server.py) without
# being tightly coupled to it — the server could be written in
# any language or framework.
#
# HOW THE PIECES FIT TOGETHER:
#
#   User calls run_demo()
#       │
#       ▼
#   ComplianceOfficer (UserProxyAgent)
#       │  initiates chat with a transaction description
#       ▼
#   PaymentReviewAgent (AssistantAgent / GPT-4)
#       │  decides to call the "call_fraud_detector" tool
#       ▼
#   ComplianceOfficer executes call_fraud_detector()
#       │  via function_map
#       ▼
#   call_fraud_detector() makes two HTTP calls:
#     1. GET  /.well-known/agent.json  (discovery)
#     2. POST /tasks/send              (task execution)
#       │
#       ▼
#   Result flows back → PaymentReviewAgent → final recommendation
#
# ============================================================

import os
import uuid         # Used to generate unique task IDs for each A2A call
import requests     # HTTP client library for calling the A2A server
import autogen      # Microsoft AutoGen multi-agent framework
from dotenv import load_dotenv  # Loads secrets from a .env file

# Load environment variables from a local .env file.
# The .env file should contain lines like:
#   OPENAI_API_KEY=sk-...
#   A2A_SECRET_KEY=my-shared-secret
# python-dotenv makes these available via os.environ without
# ever putting secrets in source code.
load_dotenv()


# -----------------------------------------------------------------
# LLM Configuration
# -----------------------------------------------------------------
# AutoGen needs to know *which* language model to use and *how*
# to authenticate with it. config_list is a list so AutoGen can
# fall back to alternative models if the primary one is unavailable.
#
# os.environ.get() reads from the environment (populated by .env
# above). If OPENAI_API_KEY is missing, the value will be None
# and API calls will fail — this is intentional so you get a clear
# error rather than a hardcoded key accidentally committed to git.
# -----------------------------------------------------------------
config_list = [
    {
        "model": "gpt-4",
        "api_key": os.environ.get("OPENAI_API_KEY"),
    }
]


# -----------------------------------------------------------------
# CONCEPT: Registering Tools in llm_config
# -----------------------------------------------------------------
# AutoGen uses OpenAI's "function calling" feature to let the LLM
# trigger Python functions. Here's how it works end-to-end:
#
#   1. We describe the tool in llm_config["functions"] using a
#      JSON Schema object. This description is sent to the LLM
#      with every prompt so it knows the tool exists.
#
#   2. When the LLM decides the tool is needed, it returns a
#      special "function_call" response instead of plain text.
#      AutoGen intercepts this response.
#
#   3. AutoGen looks up the function name in the UserProxyAgent's
#      function_map (see below) and calls the matching Python function.
#
#   4. The Python function's return value is sent back to the LLM
#      as the "function result", and the conversation continues.
#
# This keeps the LLM in control of *when* to use a tool while
# Python handles the actual execution — a clean separation of concerns.
# -----------------------------------------------------------------
llm_config = {
    "config_list": config_list,

    # temperature controls randomness in LLM responses.
    # 0.0 = fully deterministic, 1.0 = highly creative.
    # 0.3 is a good default for task-oriented agents where
    # we want consistent, reliable reasoning.
    "temperature": 0.3,

    # The "functions" list teaches the LLM what tools it can call.
    # Each entry follows the JSON Schema specification.
    "functions": [
        {
            # "name" must exactly match the key in function_map below
            "name": "call_fraud_detector",

            # "description" is read by the LLM to decide when to use
            # this tool. Write it in plain English as if explaining
            # to a colleague what the tool does.
            "description": (
                "Calls the external FraudDetectorAgent via the A2A protocol "
                "to get a fraud risk assessment for a transaction."
            ),

            # "parameters" uses JSON Schema to describe the function's inputs.
            # The LLM uses this to know what arguments to provide.
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_description": {
                        "type": "string",
                        "description": "A plain-text description of the transaction to analyze."
                    }
                },
                # "required" lists parameters the LLM must always provide
                "required": ["transaction_description"]
            }
        }
    ]
}


# -----------------------------------------------------------------
# A2A Connection Settings
# -----------------------------------------------------------------
# These values tell the client where to find the A2A server and
# how to prove it's authorized to send tasks.
#
# Reading from environment variables (rather than hardcoding) means:
#   - Developers can run the server on different ports without
#     changing code
#   - CI/CD pipelines can inject different secrets per environment
#   - Secrets never appear in git history
# -----------------------------------------------------------------

# The base URL of the A2A server — where to send HTTP requests.
# The default assumes the server is running locally (see a2a_server.py).
A2A_SERVER_URL = os.environ.get("A2A_SERVER_URL", "http://localhost:8000")

# The shared secret used to authenticate with the A2A server.
# Both this client and the server must have the same value in their .env files.
A2A_SECRET_KEY = os.environ.get("A2A_SECRET_KEY", "demo-secret-123")


# -----------------------------------------------------------------
# CONCEPT: A2A Client Function
# -----------------------------------------------------------------
# This function is the bridge between AutoGen and the A2A protocol.
# It implements the full A2A client flow in four steps:
#
#   1. DISCOVER  — fetch the Agent Card to confirm the server is
#                  reachable and learn its authentication requirements
#   2. BUILD     — construct a Task payload in A2A format
#   3. SEND      — POST the task with a Bearer token in the header
#   4. RETURN    — extract the result text and give it back to AutoGen
#
# AutoGen calls this function automatically whenever the LLM decides
# to use the "call_fraud_detector" tool (via function_map, below).
# -----------------------------------------------------------------
def call_fraud_detector(transaction_description: str) -> str:
    """
    Make an A2A call to the FraudDetectorAgent and return its assessment.

    Parameters
    ----------
    transaction_description : str
        Plain-text description of the transaction to analyze.
        The LLM in PaymentReviewAgent provides this argument.

    Returns
    -------
    str
        The fraud risk assessment text from the A2A server,
        or an error message if the call fails.
    """

    # ------------------------------------------------------------------
    # Step 1: Discover the agent via its Agent Card
    # ------------------------------------------------------------------
    # Before sending a task, we fetch the Agent Card to:
    #   a) Confirm the server is reachable (fail fast if it's down)
    #   b) Log what we're connecting to (useful for debugging)
    #   c) Confirm which auth scheme the server requires
    #
    # In production, you might cache the Agent Card rather than
    # fetching it on every call. Here we fetch it every time to keep
    # the code simple and show the full discovery flow.
    try:
        card_response = requests.get(
            f"{A2A_SERVER_URL}/.well-known/agent.json",
            timeout=5   # Don't wait more than 5 seconds — fail fast
        )
        agent_card = card_response.json()
        print(f"\n  [A2A] Agent discovered: {agent_card['name']} v{agent_card['version']}")
        print(f"  [A2A] Auth required: {agent_card['authentication']['schemes']}")
    except Exception:
        # Return a descriptive error string rather than raising an exception.
        # AutoGen will pass this string back to the LLM as the tool result,
        # allowing the LLM to handle the failure gracefully.
        return "Error: Could not reach the A2A server. Is a2a_server.py running?"

    # ------------------------------------------------------------------
    # Step 2: Build the A2A Task payload
    # ------------------------------------------------------------------
    # Every A2A task must have:
    #   id      — a UUID that uniquely identifies this task instance.
    #             uuid.uuid4() generates a random UUID like:
    #             "550e8400-e29b-41d4-a716-446655440000"
    #   message — contains the content to process, following the
    #             same role/parts structure as the server's schema
    task_payload = {
        "id": str(uuid.uuid4()),    # Generate a fresh UUID for each task
        "message": {
            "role": "user",         # This message comes from the calling agent
            "parts": [
                {"text": transaction_description}   # The actual content to analyze
            ]
        }
    }

    # ------------------------------------------------------------------
    # Step 3: Send the task with Bearer token authentication
    # ------------------------------------------------------------------
    # The Authorization header must match the format the server expects:
    #   Authorization: Bearer <secret>
    # Without this, the server returns HTTP 401 Unauthorized.
    #
    # Content-Type: application/json tells the server to parse the
    # body as JSON rather than form data or plain text.
    headers = {
        "Authorization": f"Bearer {A2A_SECRET_KEY}",
        "Content-Type": "application/json"
    }

    # Send the HTTP POST request. requests.post() will:
    #   - Serialize task_payload dict to JSON (because we pass json=...)
    #   - Attach the headers
    #   - Wait up to 10 seconds for a response
    response = requests.post(
        f"{A2A_SERVER_URL}/tasks/send",
        json=task_payload,
        headers=headers,
        timeout=10      # Fraud checks should complete quickly; fail if slow
    )

    # ------------------------------------------------------------------
    # Step 4: Handle the response
    # ------------------------------------------------------------------

    # Check for authentication failure specifically.
    # A 401 response means the secret key didn't match — this is a
    # configuration problem, not a transient network error, so we give
    # a targeted error message to help students diagnose it quickly.
    if response.status_code == 401:
        return "Error: A2A authentication failed. Check A2A_SECRET_KEY in your .env file."

    # raise_for_status() converts any other 4xx/5xx HTTP error code
    # into a Python exception. This is a requests library convenience
    # method — it's equivalent to writing:
    #   if response.status_code >= 400: raise requests.HTTPError(...)
    response.raise_for_status()

    # Parse the JSON response body into a Python dictionary.
    result = response.json()

    # Navigate the A2A response structure to extract the text result:
    #   result["result"]             — the Task result object
    #         ["message"]           — the agent's response message
    #                 ["parts"]     — list of content blocks
    #                         [0]   — first (and in this case only) block
    #                    ["text"]   — the plain-text content
    return result["result"]["message"]["parts"][0]["text"]


# -----------------------------------------------------------------
# AutoGen Agent Setup
# -----------------------------------------------------------------
# AutoGen conversations happen between at least two agents.
# The pattern used here is the classic "assistant + user proxy":
#
#   AssistantAgent  — powered by an LLM; reasons, plans, decides
#                     when to call tools, and produces final answers
#   UserProxyAgent  — represents the "user" side; initiates chats,
#                     runs tools, and decides when to stop
#
# Together they form a control loop:
#   UserProxy sends message → Assistant replies (possibly with
#   a tool call) → UserProxy executes tool → Assistant sees
#   the result → ... until a termination condition is met.
# -----------------------------------------------------------------

# PaymentReviewAgent is the "brain" — it uses GPT-4 to understand
# the transaction, decide to call the fraud detector, and formulate
# a final recommendation based on the result.
payment_reviewer = autogen.AssistantAgent(
    name="PaymentReviewAgent",
    llm_config=llm_config,      # Gives it access to GPT-4 and the tool description

    # The system_message sets the agent's persona and task instructions.
    # It is included at the start of every LLM call, so keep it concise
    # but specific — vague system messages lead to inconsistent behavior.
    system_message="""You are a Payment Review Agent for a financial institution.
    When given a transaction to review:
    1. Call the fraud detection tool to get an external risk assessment
    2. Interpret the tool's result in the context of the transaction
    3. Give a clear final recommendation: APPROVE, FLAG FOR REVIEW, or REJECT
    Always quote the fraud detection result in your response."""
)

# ComplianceOfficer is the "executor" — it starts conversations,
# runs Python tool functions when the LLM requests them, and
# decides when to stop the conversation.
compliance_officer = autogen.UserProxyAgent(
    name="ComplianceOfficer",

    # human_input_mode controls when AutoGen pauses and asks a human to type.
    # "TERMINATE" means: only ask for input when the conversation ends.
    # Other options: "ALWAYS" (pause every turn) or "NEVER" (fully automated).
    human_input_mode="TERMINATE",

    # max_consecutive_auto_reply limits how many times this agent can respond
    # automatically without human intervention. This prevents runaway loops.
    max_consecutive_auto_reply=3,

    # We're not executing code blocks in this demo — the agent only calls
    # our registered Python functions, not arbitrary code the LLM generates.
    code_execution_config=False,

    # function_map is the critical wiring: it connects the tool name the LLM
    # knows (from llm_config["functions"]) to the actual Python function.
    # When the LLM returns a function_call for "call_fraud_detector", AutoGen
    # looks it up here and calls the Python function below.
    function_map={"call_fraud_detector": call_fraud_detector}
)


# -----------------------------------------------------------------
# Sample Transactions
# -----------------------------------------------------------------
# These test cases cover two opposite ends of the risk spectrum.
# Students should consider:
#   - What other scenarios would you want to test?
#   - How would you handle edge cases (e.g., missing fields)?
#   - What would a real transaction data model look like?
# -----------------------------------------------------------------
TRANSACTIONS = [
    (
        "TXN-001",
        # High-risk: large wire transfer to unknown foreign account at 2 AM,
        # with no history of international transfers ("first" is a risk signal too)
        "Customer ID 5521: $14,800 wire transfer to an unknown foreign bank account "
        "initiated at 2 AM. This is the customer's first international transaction."
    ),
    (
        "TXN-002",
        # Low-risk: routine in-person purchase with no risk indicators
        "Customer ID 8834: $62.50 grocery store purchase in the customer's home city "
        "on a Saturday afternoon."
    ),
]


# -----------------------------------------------------------------
# Demo Runner
# -----------------------------------------------------------------
def run_demo():
    """
    Iterate through each sample transaction and run the full
    AutoGen + A2A fraud review pipeline for each one.

    This function orchestrates the high-level flow:
      1. Print a header for each transaction
      2. Start an AutoGen conversation via compliance_officer.initiate_chat()
      3. AutoGen handles all the back-and-forth internally
      4. Reset both agents so state doesn't bleed between transactions
    """
    print("=" * 60)
    print("  DEMO 04: AutoGen + A2A Secure Agent System")
    print("  Payment Fraud Detection Pipeline")
    print("=" * 60)
    print(f"\nAutoGen will call the FraudDetectorAgent at: {A2A_SERVER_URL}")
    print("Each call is authenticated with a Bearer token from .env\n")

    for txn_id, description in TRANSACTIONS:
        print(f"\n{'=' * 60}")
        print(f"  Reviewing Transaction: {txn_id}")
        print(f"{'=' * 60}")

        # initiate_chat() starts the AutoGen conversation.
        # The message is sent to payment_reviewer as the first user turn.
        # AutoGen then runs the agent loop until a termination condition:
        #   - The assistant replies with no function call (conversation ends)
        #   - max_consecutive_auto_reply is reached
        #   - The user proxy decides to terminate
        compliance_officer.initiate_chat(
            payment_reviewer,
            message=f"Please review this transaction:\n\n{description}"
        )

        # Reset both agents between transactions to clear conversation history.
        # Without this, the second transaction's chat would include context
        # from the first, which could confuse the LLM or skew the results.
        compliance_officer.reset()
        payment_reviewer.reset()


# -----------------------------------------------------------------
# Script Entry Point
# -----------------------------------------------------------------
# The `if __name__ == "__main__":` guard ensures run_demo() is only
# called when this file is executed directly (python autogen_client.py),
# not when it's imported as a module by another script.
#
# This is a Python best practice that makes code reusable and testable.
# -----------------------------------------------------------------
if __name__ == "__main__":
    try:
        run_demo()
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure OPENAI_API_KEY and A2A_SECRET_KEY are set in your .env file.")
