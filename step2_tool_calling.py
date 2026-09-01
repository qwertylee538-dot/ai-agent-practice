"""
STEP 2: Tool calling — give the model ONE tool it can decide to use.

WHAT'S NEW COMPARED TO STEP 1:
In step1, the model could only answer from what it already "knows"
(text it was trained on) — it had no way to get fresh, real-world data.
Here we give it a `get_exchange_rate` tool. The model itself decides,
based on the user's question, whether it needs to call this tool at
all, and if so, with which arguments.

IMPORTANT MENTAL MODEL:
The model does NOT run Python code. It only ever produces TEXT — in
this case, a structured piece of text that says "please call function
X with arguments Y". Our own Python code is what notices that request,
actually runs the real `get_exchange_rate()` function below, and sends
the result back to the model as a new message. The model never touches
the network, the filesystem, or anything else directly — it just talks,
and our code decides what "talking about calling a tool" means.

INSTALL / SETUP:
Same as step1 — requirements.txt already installed, .env already has
your key from step1, nothing new to configure.
"""

import json
import os

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError("DEEPSEEK_API_KEY not found — check your .env file.")

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


# ---------------------------------------------------------------------------
# PART 1: The real Python function that does actual work.
# This is just an ordinary function — nothing "AI" about it at all.
# It's the same idea as tracker.py from your scraper-toolkit portfolio,
# simplified to a single lookup.
# ---------------------------------------------------------------------------
def get_exchange_rate(base_currency: str, target_currency: str) -> str:
    """Look up a live exchange rate via a free public API.

    Returns a short text description of the result (or the error) —
    we return a STRING here because that's what gets fed back to the
    model as a chat message, and chat messages are always text.

    NOTE: we use open.er-api.com here instead of Frankfurter — Frankfurter
    sources its rates from the European Central Bank, which does not
    publish a RUB rate, so RUB lookups always failed with "not found".
    open.er-api.com is free, needs no API key, and does include RUB.
    """
    try:
        # This API's URL pattern puts the base currency directly in the
        # path and returns ALL target currencies' rates in one response
        # (unlike Frankfurter, there's no `symbols` filter parameter) —
        # so we fetch everything and then pick out the one we need.
        response = requests.get(
            f"https://open.er-api.com/v6/latest/{base_currency.upper()}",
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        if data.get("result") != "success":
            return f"Exchange rate service returned an error: {data.get('error-type', 'unknown error')}"

        rate = data.get("rates", {}).get(target_currency.upper())
        if rate is None:
            return f"Could not find a rate for {target_currency} (check the currency code)."

        return (
            f"1 {base_currency.upper()} = {rate} {target_currency.upper()} "
            f"(last updated: {data.get('time_last_update_utc')})"
        )

    except requests.exceptions.RequestException as exc:
        # If the tool itself fails, we still return a normal string —
        # this lets the model see "the tool failed" and react sensibly
        # (e.g. apologize, suggest trying again) instead of crashing.
        return f"Error while fetching exchange rate: {exc}"


# ---------------------------------------------------------------------------
# PART 2: Describe the tool to the model in its expected JSON format.
# This is NOT code the model runs — it's a *description* (a kind of
# menu) telling the model: "this function exists, here's its name,
# what it does in plain English, and what arguments it needs."
# The model reads this description and decides for itself whether and
# when to ask for it, based purely on the user's question.
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": (
                "Get the current exchange rate between two currencies. "
                "Use this whenever the user asks about a currency rate, "
                "conversion, or 'how much is X in Y'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "base_currency": {
                        "type": "string",
                        "description": "3-letter currency code to convert FROM, e.g. USD, EUR, RUB",
                    },
                    "target_currency": {
                        "type": "string",
                        "description": "3-letter currency code to convert TO, e.g. USD, EUR, RUB",
                    },
                },
                # "required" tells the model both fields must be filled
                # in before it's allowed to call this tool.
                "required": ["base_currency", "target_currency"],
            },
        },
    }
]

# A lookup table mapping a tool's name (as a string) to the real Python
# function that implements it. When the model asks to call
# "get_exchange_rate", we look it up here to find the actual function
# to run. With more tools later, you just add more entries here.
AVAILABLE_FUNCTIONS = {
    "get_exchange_rate": get_exchange_rate,
}


def ask_with_tools(question: str) -> str:
    """Ask the model a question, letting it optionally call one tool."""

    # This list holds the whole conversation so far — it grows as we
    # go: user question -> model's tool request -> our tool result ->
    # model's final answer. All of it gets sent on every API call.
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools."},
        {"role": "user", "content": question},
    ]

    # --- First call: give the model the question AND the tool menu ---
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=TOOLS,  # <-- this is the only new thing vs. step1
    )

    response_message = response.choices[0].message

    # If the model decided it needs a tool, `tool_calls` will be a
    # non-empty list. If it could answer directly (e.g. "hello"), it
    # will be empty/None, and we can just return its normal answer.
    if not response_message.tool_calls:
        return response_message.content

    # --- The model wants to call one or more tools. Let's honor that. ---
    # We must append the model's own message (its tool request) to the
    # conversation history before adding our tool's result — otherwise
    # the model won't understand what its own request was replying to.
    messages.append(response_message)

    for tool_call in response_message.tool_calls:
        function_name = tool_call.function.name
        # The model sends arguments back as a JSON string — we parse it
        # into a real Python dict so we can use it as **kwargs.
        function_args = json.loads(tool_call.function.arguments)

        print(f"[tool call] {function_name}({function_args})")

        function_to_call = AVAILABLE_FUNCTIONS[function_name]
        function_result = function_to_call(**function_args)

        # We feed the result back as a special "tool" role message,
        # tagged with `tool_call_id` so the model knows exactly which
        # of its requests this result answers (relevant when it asks
        # for several tools at once).
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": function_result,
            }
        )

    # --- Second call: now the model sees the tool's result and can
    # write a normal, human-readable final answer using it. ---
    second_response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
    )

    return second_response.choices[0].message.content


if __name__ == "__main__":
    print("Chat with ONE tool (currency rates). Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        answer = ask_with_tools(user_input)
        print(f"Model: {answer}\n")
