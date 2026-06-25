# ============================================================
# Guided Practice 01 - Answer Key
# Customer Support Ticket Triage System
# ------------------------------------------------------------
# This file is the ANSWER KEY - do not distribute to students.
# ============================================================

# pip install pyautogen python-dotenv
# Create a .env file with: OPENAI_API_KEY=your-key-here

import os
import autogen
from dotenv import load_dotenv

# -----------------------------------------------------------------
# 1. Load environment variables from .env
# -----------------------------------------------------------------
load_dotenv()

# -----------------------------------------------------------------
# 2. LLM Configuration
# -----------------------------------------------------------------
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

# -----------------------------------------------------------------
# 3. Sample Support Ticket (the "input" students will process)
# -----------------------------------------------------------------
SAMPLE_TICKET = """
Customer: Sarah Chen
Account: #78341
Subject: Charged twice for last month's subscription

Hi, I noticed two identical charges of $49.99 on my credit card statement dated
June 3rd, both labeled as 'Monthly Plan - AutoFlow'. I only have one account and
one subscription. I've attached my bank statement as proof. I'd like a refund for
the duplicate charge as soon as possible. I've been a loyal customer for 2 years
and this has never happened before.
"""

# -----------------------------------------------------------------
# 4. Define the Three Specialized Agents
# -----------------------------------------------------------------

# Agent 1: Classifies the ticket type and urgency
ticket_classifier = autogen.AssistantAgent(
    name="TicketClassifier",
    llm_config=llm_config,
    system_message="""You are a Ticket Classifier agent. When given a support ticket, you must:
    1. Identify the issue category (Technical / Billing / Account / General Inquiry)
    2. Assign an urgency level (Critical / High / Medium / Low) with a brief reason
    3. Extract key facts: customer name, account number, the core problem, any evidence mentioned

    Always output a clean structured summary in this format:
    Category | Urgency | Customer | Account | Core Problem | Evidence/Notes""",
)

# Agent 2: Proposes a resolution based on the classification
solution_specialist = autogen.AssistantAgent(
    name="SolutionSpecialist",
    llm_config=llm_config,
    system_message="""You are a Solution Specialist agent. Based on the ticket classification provided,
    your job is to:
    1. Recommend 2-3 concrete steps the support team should take to resolve the issue
    2. Flag any policy considerations (e.g., refund eligibility, verification requirements)
    3. Specify whether the ticket should be escalated and to which team
    4. Provide an estimated resolution timeframe

    Be specific, actionable, and reference the customer's situation directly.""",
)

# Agent 3: Drafts the customer-facing response email
response_drafter = autogen.AssistantAgent(
    name="ResponseDrafter",
    llm_config=llm_config,
    system_message="""You are a Response Drafter agent. Using the classification and proposed solution,
    draft a professional and empathetic email response to the customer that:
    1. Addresses the customer by their first name
    2. Acknowledges the issue and validates their frustration
    3. Clearly explains the next steps being taken
    4. Provides a realistic resolution timeline
    5. Closes with a warm, professional sign-off

    Keep the tone human, warm, and solution-focused. Avoid jargon.""",
)

# -----------------------------------------------------------------
# 5. UserProxy Agent - represents the support manager kicking off the workflow
# -----------------------------------------------------------------
support_manager = autogen.UserProxyAgent(
    name="SupportManager",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=2,
    code_execution_config=False,
)

# -----------------------------------------------------------------
# 6. Group Chat - all agents collaborate in sequence
# -----------------------------------------------------------------
group_chat = autogen.GroupChat(
    agents=[ticket_classifier, solution_specialist, response_drafter, support_manager],
    messages=[],
    max_round=9,
)

manager = autogen.GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
)

# -----------------------------------------------------------------
# 7. Workflow Entry Point
# -----------------------------------------------------------------
def run_triage_workflow(ticket: str):
    print("=" * 60)
    print("  CUSTOMER SUPPORT TRIAGE SYSTEM")
    print("=" * 60)

    initial_message = f"""A new customer support ticket has arrived.
Please process it through the full triage workflow:

{ticket}

Steps:
1. TicketClassifier: Classify and extract key details from the ticket
2. SolutionSpecialist: Recommend a resolution based on the classification
3. ResponseDrafter: Draft the customer-facing email response
"""

    support_manager.initiate_chat(manager, message=initial_message)


if __name__ == "__main__":
    try:
        run_triage_workflow(SAMPLE_TICKET)
    except Exception as e:
        print(f"\nError: {e}")
        print("Ensure OPENAI_API_KEY is set in your .env file and pyautogen is installed.")
