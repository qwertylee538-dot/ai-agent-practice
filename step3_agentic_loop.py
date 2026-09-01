"""
STEP 3: The agentic loop — let the model call tools MULTIPLE times in
a row, deciding each time what to do next, until it's ready to give a
final answer.

WHY STEP 2 WASN'T QUITE ENOUGH:
In step2, our code assumed the model would need AT MOST one round of
tool calls: ask -> maybe call a tool once -> get final answer. That
works for simple questions, but falls apart for anything that needs
several steps chained together — e.g. "how much would 250 dollars be
in rubles?" needs TWO things: (1) look up the exchange rate, then
(2) actually multiply 250 by that rate. That's two separate tool calls,
with the second one depending on the result of the first.

THE FIX — A LOOP INSTEAD OF A FIXED SEQUENCE:
Instead of hardcoding "call once, then answer", we now just keep
asking the model "what next?" in a loop. Each time:
  1. Send the full conversation so far to the model.
  2. If the model asks for a tool call -> run it, append the result,
     go back to step 1.
  3. If the model instead returns a normal text answer -> we're done,
     return it.
This is genuinely the core idea behind every "AI agent" framework you
will ever see (LangChain, CrewAI, custom agents, etc.) — they're all
elaborate versions of this exact loop.

NEW TOOL ADDED:
We add a `calculate` tool alongside `get_exchange_rate`, specifically
so a single question can require BOTH tools, back to back — that's
what makes the loop actually loop instead of running just once.
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
# Tool implementations (real Python functions)
# ---------------------------------------------------------------------------
def get_exchange_rate(base_currency: str, target_currency: str) -> str:
    """Look up a live exchange rate via a free public API (open.er-api.com)."""
    try:
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
        return f"Error while fetching exchange rate: {exc}"


def calculate(expression: str) -> str:
    """Safely evaluate a simple arithmetic expression, e.g. "250 * 86.41".

    We deliberately do NOT use Python's built-in eval() here — eval()
    would happily run ANY Python code the model (or a malicious prompt
    hidden in some webpage the model read) sends us, which is a real
    security risk. Instead we only allow digits, spaces, and the four
    basic math operators — anything else gets rejected before it's
    evaluated at all.
    """
    import re

    allowed_pattern = r"^[0-9\.\s\+\-\*/\(\)]+$"
    if not re.match(allowed_pattern, expression):
        return "Error: expression contains characters that are not allowed."

    try:
        # eval() is safe HERE specifically because the regex above has
        # already guaranteed the string can only contain numbers,
        # whitespace, parentheses, and + - * / — nothing else, including
        # no letters, so no function calls or variable access are possible.
        result = eval(expression)
        return str(result)
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": "Get the current exchange rate between two currencies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "base_currency": {"type": "string", "description": "3-letter code, e.g. USD"},
                    "target_currency": {"type": "string", "description": "3-letter code, e.g. RUB"},
                },
                "required": ["base_currency", "target_currency"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate a simple arithmetic expression (numbers and + - * / ( ) only). "
                "Use this for any multiplication, division, or other math the user's "
                "question requires — do not do arithmetic yourself, always call this tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "e.g. '250 * 86.41'",
                    },
                },
                "required": ["expression"],
            },
        },
    },
]

AVAILABLE_FUNCTIONS = {
    "get_exchange_rate": get_exchange_rate,
    "calculate": calculate,
}


def run_agent(question: str, max_iterations: int = 5) -> str:
    """Run the full agentic loop for one user question.

    `max_iterations` is a safety limit — without it, a model stuck in
    a confused loop (calling tools forever without ever answering)
    could run indefinitely and burn through your API balance. Real
    agent frameworks all have some version of this same safeguard.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to tools. "
                "Always use the calculate tool for any arithmetic — "
                "never compute numbers yourself."
            ),
        },
        {"role": "user", "content": question},
    ]

    for iteration in range(1, max_iterations + 1):
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS,
        )
        response_message = response.choices[0].message

        # No tool calls this round -> the model is giving its final
        # answer. This is the loop's exit condition.
        if not response_message.tool_calls:
            return response_message.content

        print(f"[loop iteration {iteration}] model requested {len(response_message.tool_calls)} tool call(s)")

        # Record the model's own tool-request message before we can
        # add the tool results that answer it.
        messages.append(response_message)

        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"  -> {function_name}({function_args})")

            function_to_call = AVAILABLE_FUNCTIONS[function_name]
            function_result = function_to_call(**function_args)

            print(f"  <- {function_result}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": function_result,
                }
            )
        # Loop continues: we go back to the top and ask the model
        # again, now with the tool result added to the conversation.
        # It might ask for another tool (e.g. calculate, after
        # get_exchange_rate), or it might be ready to answer.

    # If we got here, we hit max_iterations without a final answer —
    # this is a safety net, not something that should normally happen.
    return "Sorry, I couldn't complete this request within the allowed number of steps."


if __name__ == "__main__":
    print("Agentic loop demo (currency rates + calculator). Type 'exit' to quit.\n")
    print("Try: 'Сколько будет 250 долларов в рублях по текущему курсу?'\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        answer = run_agent(user_input)
        print(f"Model: {answer}\n")
