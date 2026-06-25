# Step 1: Install the required libraries
# pip install ag2[openai] openai python-dotenv

# Step 2: Import the libraries and load the OpenAI API key from .env
#
# autogen              — AG2 (formerly PyAutoGen) multi-agent framework; imported
#                        as 'autogen' and provides AssistantAgent, UserProxyAgent,
#                        GroupChat, and GroupChatManager at the top level
# Dict, List, Optional — typing helpers that annotate function signatures so IDEs
#                        and type checkers can verify argument types at development
#                        time; they have no runtime cost
# json                 — standard library module for serialising Python objects to
#                        JSON strings; useful when passing structured data between
#                        agents or logging results to file
# datetime             — standard library module used here to timestamp the
#                        compliance report with the exact moment it was generated
# os                   — standard library module for reading environment variables
#                        via os.environ.get(); keeps secrets out of source code
# load_dotenv          — reads key=value pairs from a .env file and injects them
#                        into os.environ so os.environ.get() can find them

import autogen
from typing import Dict, List, Optional
import json
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Step 3: Create a configuration list and an LLM configuration
#
# config_list is the list of model back-ends that AutoGen will use. Each entry
# is a dictionary describing one endpoint. AutoGen iterates through the list and
# falls back to the next entry if a request fails, enabling automatic retry
# across multiple keys or regions without any extra application logic.
#
# Fields used here:
#   model   — the OpenAI model identifier; gpt-4o-mini offers a strong balance
#             of reasoning capability and low cost for document analysis tasks
#   api_key — the secret credential loaded from the environment; never hardcode
#             this value directly in source files that may be committed to git

config_list = [
    {
        "model": "gpt-4o-mini",
        "api_key": os.environ.get("OPENAI_API_KEY"),
    }
]

# llm_config is the top-level configuration dictionary passed to every agent
# and to the GroupChatManager. Wrapping config_list inside llm_config allows
# AutoGen to attach additional settings (e.g. temperature, timeout, caching)
# to the same object without changing every call site.

llm_config = {
    "config_list": config_list,
}

# Step 4: Create a sample document
#
# SAMPLE_DOCUMENTS simulates a document repository containing records of
# different types (HR Policy, Technical Manual, Operational Guidelines).
# Each document is a plain Python dictionary with five fields:
#   id           — unique identifier used to reference the document in agent
#                  messages and audit trails
#   title        — human-readable name displayed in reports
#   type         — category that determines which compliance rules apply
#   content      — the full text the agents will analyse; intentionally contains
#                  known issues (missing sections, outdated references, absent
#                  signatures) so the compliance workflow has findings to surface
#   last_updated — ISO-8601 date string; agents use this to flag stale documents
#                  that have exceeded their maximum permitted age

SAMPLE_DOCUMENTS = [
    {
        "id": "DOC001",
        "title": "Employee Remote Work Policy",
        "type": "HR Policy",
        "content": """
        Remote Work Policy
        Effective Date: January 2023
        1. Eligibility: All full-time employees
        2. Equipment: Company will provide laptop
        3. Work Hours: Standard 9-5 EST
        Note: This policy supersedes the 2021 remote work guidelines.
        """,
        "last_updated": "2023-01-15"
    },
    {
        "id": "DOC002",
        "title": "Data Security Manual",
        "type": "Technical Manual",
        "content": """
        Data Security Manual
        Version: 2.1
        1. Password Requirements: Minimum 8 characters
        2. Encryption: Use AES-256 for sensitive data
        Missing sections:
        - Incident Response Procedures
        - Data Retention Policy
        """,
        "last_updated": "2024-03-20"
    },
    {
        "id": "DOC003",
        "title": "Customer Service Guidelines",
        "type": "Operational Guidelines",
        "content": """
        Customer Service Guidelines
        1. Response Time: Within 24 hours
        2. Escalation: After 48 hours to supervisor
        3. Communication: Use template responses from 2019 handbook
        Approved by: [Missing Signature]
        Review Date: [Not Specified]
        """,
        "last_updated": "2024-11-10"
    }
]

# Step 5: Create specialized agents
#
# DocumentComplianceSystem encapsulates the entire multi-agent pipeline inside a
# single class. This design keeps agent construction, group-chat wiring, and
# workflow orchestration together so the caller only needs to instantiate the
# class and invoke process_document_batch().
#
# The three AssistantAgents each own a distinct role in the pipeline:
#   DocumentScanner      — ingests raw documents and extracts structured metadata
#   ComplianceChecker    — analyses content for policy violations and missing data
#   ReviewCoordinator    — routes findings to the appropriate reviewers and sets
#                          deadlines based on issue severity
#
# A UserProxyAgent acts as the human-in-the-loop entry point. Setting
# human_input_mode="TERMINATE" means the agent will only pause for human input
# when the conversation naturally ends, letting the workflow run autonomously.
#
# GroupChat and GroupChatManager wire all four agents into a round-robin
# conversation managed by a separate LLM instance that decides which agent
# should speak next based on the conversation history.

class DocumentComplianceSystem:
    def __init__(self):

        # DocumentScanner is responsible for the first stage of the pipeline.
        # Its system_message defines the agent's persona and constrains its
        # outputs to structured metadata summaries. Providing an explicit output
        # format in the system message reduces hallucination and makes the
        # downstream agents' parsing work more reliable.

        self.document_scanner = autogen.AssistantAgent(
            name="DocumentScanner",
            llm_config=llm_config,
            system_message="""You are a Document Scanner agent responsible for:
            1. Monitoring document repositories for new or updated documents
            2. Extracting document metadata (title, type, last updated date)
            3. Preparing documents for compliance analysis
            4. Identifying document categories (HR, Technical, Operational)
            Format your findings clearly with document ID, type, and key metadata."""
        )

        # ComplianceChecker receives the scanner's output and performs the core
        # policy analysis. Asking the model to categorise issues by severity
        # (Critical / Major / Minor) gives the ReviewCoordinator a clear signal
        # for routing decisions without requiring additional parsing logic.

        self.compliance_checker = autogen.AssistantAgent(
            name="ComplianceChecker",
            llm_config=llm_config,
            system_message="""You are a Compliance Checker agent responsible for:
            1. Analyzing documents for compliance issues such as:
               - Missing required sections (approvals, dates, signatures)
               - Outdated references or superseded content
               - Inconsistent terminology or formatting
               - Version control issues
            2. Categorizing issues by severity (Critical, Major, Minor)
            3. Providing specific recommendations for remediation
            Always structure your analysis with: Issue Type | Severity | Details | Recommendation"""
        )

        # ReviewCoordinator is the final stage of the pipeline. It consumes the
        # severity-tagged issues from ComplianceChecker and maps each one to an
        # escalation path and a concrete due date. Encoding the routing rules
        # directly in the system message keeps the business logic in one place
        # and makes it easy to update without touching application code.

        self.review_coordinator = autogen.AssistantAgent(
            name="ReviewCoordinator",
            llm_config=llm_config,
            system_message="""You are a Review Coordinator agent responsible for:
            1. Processing compliance check results
            2. Determining routing based on severity:
               - Critical: Immediate escalation to Legal/Executive review
               - Major: Route to department head for approval
               - Minor: Send revision request to document owner
            3. Generating action items with deadlines
            4. Creating audit trail for compliance tracking
            Format routing decisions with: Document | Issues | Routing Decision | Action Items | Due Date"""
        )

        # UserProxyAgent represents the human operator inside the group chat.
        # Key parameters:
        #   human_input_mode      — "TERMINATE" means the proxy only prompts the
        #                           human when the chat ends naturally; the workflow
        #                           runs fully automated until that point
        #   max_consecutive_auto_reply
        #                         — caps the number of turns the proxy can reply
        #                           automatically before requiring human approval,
        #                           preventing infinite loops in edge cases
        #   code_execution_config — set to False because this demo does not need
        #                           the agent to run any generated code; disabling
        #                           it avoids accidental execution of LLM output

        self.user_proxy = autogen.UserProxyAgent(
            name="UserProxy",
            human_input_mode="TERMINATE",
            max_consecutive_auto_reply=2,
            code_execution_config=False,
        )

        # GroupChat assembles all four agents into a shared conversation context.
        # AutoGen's GroupChatManager uses an LLM to read the message history and
        # decide which agent should respond next, enabling flexible turn-taking
        # that is not limited to a fixed round-robin order.
        #
        # max_round caps the total number of agent turns; set to 10 here to
        # ensure the demo completes in a reasonable time while still allowing
        # enough back-and-forth for all three stages to produce output.

        self.group_chat = autogen.GroupChat(
            agents=[
                self.document_scanner,
                self.compliance_checker,
                self.review_coordinator,
                self.user_proxy
            ],
            messages=[],
            max_round=10
        )

        # GroupChatManager is itself an LLM-powered agent that orchestrates the
        # conversation. It receives the full message history after each turn and
        # selects the most appropriate next speaker. Passing llm_config here
        # gives the manager its own model instance so its speaker-selection
        # reasoning does not interfere with the agents' task reasoning.

        self.manager = autogen.GroupChatManager(
            groupchat=self.group_chat,
            llm_config=llm_config
        )

    def process_document_batch(self, documents: List[Dict]):
        """Process a batch of documents through the compliance workflow."""

        # Step 6: Create a function to run the agents
        #
        # process_document_batch() is the public entry point for the workflow.
        # It builds a single structured prompt from all documents and hands it
        # to the UserProxyAgent, which kicks off the group chat. Consolidating
        # all documents into one message (rather than sending them one at a time)
        # lets each agent see the full batch and identify cross-document patterns
        # such as conflicting policies or duplicate procedures.

        print("Starting AutoGen Document Compliance Workflow\n")
        print("=" * 60)

        # Build a human-readable summary of every document. Limiting the content
        # preview to 200 characters keeps the initial prompt concise; agents can
        # reference the full content that was injected earlier in the message.
        doc_summary = "\n\n".join([
            f"Document ID: {doc['id']}\n"
            f"Title: {doc['title']}\n"
            f"Type: {doc['type']}\n"
            f"Last Updated: {doc['last_updated']}\n"
            f"Content Preview:\n{doc['content'][:200]}..."
            for doc in documents
        ])

        # The initial_message serves as the system prompt for the entire group
        # chat session. Explicitly numbering the workflow steps in the message
        # primes each agent to look for its own step and hand off cleanly to
        # the next agent, reducing the chance that a step is skipped or repeated.

        initial_message = f"""
        New documents have been detected in the repository. Please process these documents through our compliance workflow:
        {doc_summary}
        Workflow Steps:
        1. DocumentScanner: Analyze and categorize the documents
        2. ComplianceChecker: Perform compliance analysis on each document
        3. ReviewCoordinator: Determine routing and create action items
        Begin the analysis.
        """

        # initiate_chat() starts the group conversation. The UserProxy sends
        # initial_message to the GroupChatManager, which then selects the first
        # agent to respond. The conversation continues autonomously until
        # max_round is reached or an agent issues a TERMINATE signal.

        self.user_proxy.initiate_chat(
            self.manager,
            message=initial_message
        )

        return self.generate_compliance_report()

    def generate_compliance_report(self):
        """Generate a summary compliance report."""

        # generate_compliance_report() prints a human-readable summary after the
        # group chat ends. In a production system this method would aggregate the
        # structured outputs written to self.group_chat.messages, persist them to
        # a database, and trigger downstream notifications. Here it prints a
        # fixed summary to keep the demo output clean and predictable.

        print("\n" + "=" * 60)
        print("COMPLIANCE WORKFLOW SUMMARY")
        print("=" * 60)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Documents Processed: {len(SAMPLE_DOCUMENTS)}")
        print("\nKey Findings:")
        print("- Missing required sections detected in technical documentation")
        print("- Outdated references found in operational guidelines")
        print("- Signature approvals missing in customer service documents")
        print("\nWorkflow Demonstration Complete")


# Step 7: Create a function to set up custom compliance rules
#
# create_custom_compliance_rules() returns a plain dictionary that maps each
# document type to a set of rules. Keeping rules in a data structure (rather
# than hard-coding them inside agent system messages) makes it straightforward
# to load them from a database or configuration file in production.
#
# Rule fields:
#   required_sections  — section headings the ComplianceChecker must verify are
#                        present; absence of any heading is flagged as a finding
#   max_age_months     — how long a document may remain unreviewed before it is
#                        treated as stale; technical documents age out faster
#                        because security requirements change more frequently
#   requires_signature — whether the document must carry an approval signature;
#                        HR and Operational documents typically require this for
#                        legal enforceability

def create_custom_compliance_rules():
    """Define custom compliance rules for different document types."""
    return {
        "HR Policy": {
            "required_sections": ["Effective Date", "Eligibility", "Approval"],
            "max_age_months": 12,
            "requires_signature": True
        },
        "Technical Manual": {
            "required_sections": ["Version", "Security Requirements", "Incident Response"],
            "max_age_months": 6,
            "requires_signature": False
        },
        "Operational Guidelines": {
            "required_sections": ["Procedures", "Escalation", "Review Date"],
            "max_age_months": 12,
            "requires_signature": True
        }
    }


# Step 8: Define a function to set up webhook monitoring
#
# setup_monitoring_webhook() returns the configuration that a production system
# would register with a document management platform (e.g. SharePoint, Confluence)
# to receive push notifications whenever a document is created or updated.
# Using webhooks instead of polling eliminates the lag between a document change
# and the compliance check, and removes the need for a scheduled job.
#
# Config fields:
#   endpoint  — the URL path on this service that will receive POST payloads
#               from the document platform
#   events    — the subset of platform events that should trigger a compliance
#               run; restricting to create and update avoids spurious runs on
#               view or comment events
#   frequency — descriptive label indicating that delivery is real-time (i.e.
#               event-driven) rather than batched on a schedule

def setup_monitoring_webhook():
    """Setup webhook for continuous document monitoring."""
    webhook_config = {
        "endpoint": "/api/document-updates",
        "events": ["document.created", "document.updated"],
        "frequency": "real-time"
    }
    return webhook_config


# Step 9: Execute the code
#
# run_demo() is the top-level entry point that prints a feature summary and
# then delegates to DocumentComplianceSystem. Keeping the demo header text here
# (rather than inside the class) separates presentation concerns from the core
# compliance logic, making it easier to reuse the class in other contexts.

def run_demo():
    """Run the AutoGen document compliance demonstration."""
    print("AutoGen Document Compliance System Demo")
    print("=" * 60)
    print("Demonstrating key AutoGen features:")
    print("- Multi-agent collaboration")
    print("- Declarative agent configuration")
    print("- Group chat orchestration")
    print("- Automated workflow execution")
    print("=" * 60 + "\n")

    compliance_system = DocumentComplianceSystem()
    compliance_system.process_document_batch(SAMPLE_DOCUMENTS)


# The 'if __name__ == "__main__"' guard ensures run_demo() is only called when
# this file is executed directly (e.g. 'python MAI-II_Demo_02_autogen_compliance.py').
# When the module is imported by a test runner or another script, this block is
# skipped, preventing unintended side effects such as API calls or printed output.
#
# The try/except block surfaces configuration errors (missing API key, wrong
# package version) with a plain-English message so the developer knows exactly
# what to fix rather than seeing a raw traceback.

if __name__ == "__main__":
    try:
        run_demo()
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure OPENAI_API_KEY is set in your .env file and AutoGen is installed.")
