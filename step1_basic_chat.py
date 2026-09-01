"""
STEP 1: The most basic possible LLM call — no tools, no loop.

WHY THIS STEP FIRST:
Before building anything "agentic", it's worth seeing the plain
request -> response pattern with your own eyes. Every agent framework
is ultimately built on top of exactly this call, repeated and made
smarter.

HOW IT WORKS:
DeepSeek's API is "OpenAI-compatible" — meaning it accepts the exact
same request format as OpenAI's API. That's why we install the
`openai` Python package even though we're talking to DeepSeek: we just
point it at DeepSeek's server address (`base_url`) instead of OpenAI's.

INSTALL FIRST:
    pip install -r requirements.txt

SETUP FIRST:
    1. Copy .env.example to a new file named exactly ".env"
    2. Open .env and paste your real DeepSeek API key after the "="
    3. Never share that .env file or paste its contents anywhere
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

# load_dotenv() reads the ".env" file in this folder and copies its
# key=value pairs into the environment variables of this running
# script — that's how the API key gets from the file into our code
# without ever being typed directly into a .py file.
load_dotenv()

# os.getenv() reads an environment variable by name. If the ".env"
# file is missing or the key wasn't set, this will be None — the
# check right after catches that case with a clear error message
# instead of a confusing crash deep inside the API call.
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError(
        "DEEPSEEK_API_KEY not found. Did you create a .env file "
        "(copied from .env.example) with your real key inside it?"
    )

# The OpenAI client is a general-purpose HTTP client for "OpenAI-style"
# APIs. `base_url` is what redirects it to DeepSeek's servers instead
# of OpenAI's — everything else about how we call it stays the same.
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)


def ask(question: str) -> str:
    """Send one question to the model and return its text answer."""
    response = client.chat.completions.create(
        # "deepseek-chat" is DeepSeek's general-purpose chat model.
        model="deepseek-chat",
        # `messages` is the conversation history sent on every call —
        # the model has no memory of its own between requests, so this
        # list IS the model's entire "memory" for this call.
        messages=[
            {
                "role": "system",
                # The "system" message sets the model's behavior/persona
                # — it's not something the user typed, it's instructions
                # from the developer (you) about how to behave.
                "content": "You are a concise, helpful assistant.",
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    # The response contains multiple possible "choices" (usually just
    # one). Each choice has a `.message.content` — the actual text
    # the model generated.
    return response.choices[0].message.content


if __name__ == "__main__":
    # A tiny interactive loop so you can ask several questions in a row
    # without restarting the script each time.
    print("Basic chat (no tools). Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        answer = ask(user_input)
        print(f"Model: {answer}\n")
