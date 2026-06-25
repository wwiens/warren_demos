# Step 1: Import required libraries
#
# asyncio              — Python's built-in async I/O framework; AutoGen's model
#                        clients expose async methods, so all calls must run
#                        inside an async function driven by an event loop
# os                   — standard library module for reading environment variables
#                        via os.getenv(); keeps secrets out of source code
# load_dotenv          — reads key=value pairs from a .env file and injects them
#                        into os.environ so os.getenv() can find them
# UserMessage          — AutoGen Core data class that wraps a single user turn;
#                        the 'source' field identifies who authored the message
# OpenAIChatCompletionClient
#                      — AutoGen Extension that adapts OpenAI's chat-completion
#                        API to AutoGen's model-client interface; handles
#                        authentication, serialisation, and async calls

import asyncio
import os
from dotenv import load_dotenv
from autogen_core.models import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Step 2: Load API credentials from .env
#
# Keeping secrets in a .env file (rather than hardcoded in the script) means
# the file can be shared or committed to version control safely.
# load_dotenv() scans the current directory (and parent directories) for a .env
# file and copies its key=value pairs into os.environ.
#
# Key loaded here:
#   OPENAI_API_KEY — authenticates every request to the OpenAI API; read below
#                    by os.getenv() when the client is constructed

load_dotenv()


async def main():
    # Step 3: Create the OpenAI model client
    #
    # OpenAIChatCompletionClient is AutoGen's adapter for OpenAI's
    # chat-completion endpoint. Wrapping the raw API in a typed client object
    # lets AutoGen manage message history, retry logic, and token counting
    # without coupling the application code to HTTP details.
    #
    # Parameters:
    #   model   — the OpenAI model to use; gpt-4o-mini balances cost and
    #             capability for short, factual queries
    #   api_key — the OpenAI secret key; loaded from the environment rather
    #             than hardcoded so the same script works in any deployment
    #             without source changes

    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Step 4: Send a message to the model and await the response
    #
    # KEY CONCEPT — async/await:
    #   create() is a coroutine, meaning it suspends the current function while
    #   the HTTP request travels to OpenAI and the response streams back. Using
    #   'await' here lets the event loop run other work in the meantime instead
    #   of blocking the entire process.
    #
    # create() accepts a list of messages representing the conversation history.
    # For a single-turn request we pass a list with one UserMessage.
    #
    # UserMessage fields:
    #   content — the text prompt sent to the model
    #   source  — a label identifying the author of this turn; AutoGen uses
    #             this when building multi-agent message logs so each
    #             participant's contributions can be traced

    response = await model_client.create([
        UserMessage(
            content="What is Microsoft AutoGen?",
            source="user"
        )
    ])

    # Step 5: Print the model's reply
    #
    # response.content holds the plain-text string returned by the model.
    # In a production application you would typically process or route this
    # content further; here we print it directly to verify the integration.

    print(response.content)

    # Step 6: Close the model client
    #
    # close() releases any underlying HTTP connections held open by the client.
    # Calling it explicitly (rather than relying on garbage collection) ensures
    # connections are returned to the pool promptly, which matters when the
    # client is used inside a long-running service or when many clients are
    # created in a loop.

    await model_client.close()


# Step 7: Run the async entry point
#
# Python scripts execute synchronously by default. asyncio.run() creates a new
# event loop, runs the main() coroutine to completion, and then closes the loop.
# This is the standard pattern for running async code from a script's top level.
#
# The 'if __name__ == "__main__"' guard ensures this block is skipped when the
# module is imported by another script or a test runner, preventing unintended
# execution.

if __name__ == "__main__":
    asyncio.run(main())
