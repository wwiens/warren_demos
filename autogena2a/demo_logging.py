# ============================================================
# Demo 05 - AutoGen with Logging
# Shows how to add structured logging to an AutoGen workflow
# ------------------------------------------------------------
# Install:  pip install pyautogen python-dotenv
# Run:      python demo_logging.py
# ============================================================

import os
import logging
from datetime import datetime
import autogen
from dotenv import load_dotenv

load_dotenv()

# ── 1. LOGGING SETUP ──────────────────────────────────────────
#
# We configure two handlers:
#   - StreamHandler  → prints to the console in real time
#   - FileHandler    → writes every log line to a timestamped file
#
# This means you always have a permanent record of each run,
# even if the console output scrolls past.

LOG_FILE = f"autogen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),          # console
        logging.FileHandler(LOG_FILE),    # file
    ]
)

log = logging.getLogger("autogen_demo")

# ── 2. LLM CONFIGURATION ──────────────────────────────────────

config_list = [
    {
        "model": "gpt-4",
        "api_key": os.environ.get("OPENAI_API_KEY"),
    }
]

llm_config = {
    "config_list": config_list,
    "temperature": 0.5,
}

# ── 3. CREATE AGENTS ──────────────────────────────────────────

log.info("Initializing agents...")

assistant = autogen.AssistantAgent(
    name="Assistant",
    llm_config=llm_config,
    system_message=(
        "You are a knowledgeable assistant. "
        "Answer questions clearly in 3-4 sentences. "
        "End every reply with TERMINATE."
    ),
)

user_proxy = autogen.UserProxyAgent(
    name="User",
    human_input_mode="NEVER",           # fully automated — no keyboard input needed
    max_consecutive_auto_reply=1,
    code_execution_config=False,
    is_termination_msg=lambda m: "TERMINATE" in m.get("content", ""),
)

log.info("Agents ready: %s, %s", assistant.name, user_proxy.name)

# ── 4. RUN THE CONVERSATION WITH LOGGING ──────────────────────

QUESTION = "What are three key advantages of using multi-agent AI systems?"

log.info("Starting conversation")
log.info("Question: %s", QUESTION)

try:
    user_proxy.initiate_chat(assistant, message=QUESTION)

    # After the chat ends, log each message from the history
    messages = user_proxy.chat_messages[assistant]
    log.info("Conversation complete — %d messages exchanged", len(messages))

    for i, msg in enumerate(messages, 1):
        # Trim long content so log lines stay readable
        preview = msg["content"][:120].replace("\n", " ")
        log.info("  [%d] role=%-12s  %s", i, msg["role"], preview)

except Exception as e:
    log.error("Conversation failed: %s", e)
    raise

log.info("Log saved to: %s", LOG_FILE)
