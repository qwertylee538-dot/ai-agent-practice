"""
STEP 5: Give the agent PERSISTENT MEMORY and a new "read_file" tool.

WHY THIS IS THE NATURAL NEXT STEP AFTER STEP 4:
Every agent so far (steps 1-4) forgets everything the moment you close
the terminal. Each time you run the script, `messages` starts over as
an empty list — the agent has no idea what you talked about last time.
Real assistants (and real client projects) need memory that survives
between runs: you close your laptop, come back tomorrow, and the agent
still remembers what you told it.

THE FIX — SAVE THE CONVERSATION TO A FILE:
Instead of keeping `messages` only in memory (RAM), we now:
  1. On startup, try to load messages.json — if it exists, continue
     that conversation; if not, start a fresh one.
  2. After every question is answered, save the whole message list
     back to messages.json.
This is the simplest possible form of memory — no database, no vector
search, just a JSON file — but it's exactly how a lot of real small
tools/bots actually do it under the hood.

IMPORTANT DIFFERENCE FROM STEP 3:
In step3, when the model asked for a tool call, we did
`messages.append(response_message)` — appending the raw object the
OpenAI SDK gave us. That worked there because the list was only ever
sent back into the API, never saved to a file. Here we DO save to a
file, and that raw object can't be turned into JSON text directly. So
instead we build a plain, ordinary Python dictionary out of it first —
see `build_assistant_message()` below.

NEW TOOL ADDED: `read_file`
So the agent can actually be useful with local files (a very common
real request from clients — "read my file and tell me X"), we add a
tool that reads a text file from disk and returns its contents. We
keep it deliberately simple and safe:
  - only reads files inside this project's own folder (no reading
    random files elsewhere on the computer)
  - refuses files that are too large, so we don't blow the API's
    context window or the DeepSeek token budget on one giant file
"""

import json
import os
import re

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError("DEEPSEEK_API_KEY not found — check your .env file.")

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

MEMORY_FILE = "messages.json"
MAX_FILE_READ_CHARS = 8000  # safety limit so one huge file can't eat the whole context


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

    Same safety approach as step3: a regex allow-list is checked BEFORE
    eval() ever runs, so only digits/spaces/+-*/() can reach eval() —
    no letters means no function calls or variable access are possible.
    """
    allowed_pattern = r"^[0-9\.\s\+\-\*/\(\)]+$"
    if not re.match(allowed_pattern, expression):
        return "Error: expression contains characters that are not allowed."

    try:
        result = eval(expression)
        return str(result)
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


def read_file(file_name: str) -> str:
    """Read a text file that lives in THIS project's folder and return its
    contents, so the model can answer questions about it.

    Safety rules (deliberately strict, this is what a real client-facing
    tool needs to think about):
      - os.path.basename() strips away any folder path the model might
        try to sneak in (e.g. "../../secrets.txt" becomes "secrets.txt"),
        so the tool can ONLY ever read a file sitting directly in this
        project folder — never anywhere else on the computer.
      - We refuse anything that isn't a plain existing file.
      - We cut off the content at MAX_FILE_READ_CHARS characters so a
        huge file can't blow up the conversation sent to the API.
    """
    safe_name = os.path.basename(file_name)
    if not os.path.isfile(safe_name):
        return f"Error: file '{safe_name}' was not found in the project folder."

    try:
        with open(safe_name, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as exc:
        return f"Error reading file: {exc}"

    if len(content) > MAX_FILE_READ_CHARS:
        content = content[:MAX_FILE_READ_CHARS] + "\n...[file truncated, too long]"

    return content


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
                    "expression": {"type": "string", "description": "e.g. '250 * 86.41'"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the contents of a text file that lives in this project's folder "
                "(for example a .txt or .csv the user placed there), so you can answer "
                "questions about it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {"type": "string", "description": "e.g. 'notes.txt'"},
                },
                "required": ["file_name"],
            },
        },
    },
]

AVAILABLE_FUNCTIONS = {
    "get_exchange_rate": get_exchange_rate,
    "calculate": calculate,
    "read_file": read_file,
}


# ---------------------------------------------------------------------------
# Memory: load / save the conversation to a JSON file on disk
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a helpful assistant with access to tools. "
        "Always use the calculate tool for any arithmetic — never compute "
        "numbers yourself. Use read_file when the user asks about a local file."
    ),
}


def load_memory() -> list:
    """Load the saved conversation from disk, or start a fresh one."""
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                messages = json.load(f)
            print(f"(loaded {len(messages)} saved messages from {MEMORY_FILE})")
            return messages
        except Exception as exc:
            print(f"(could not load saved memory, starting fresh: {exc})")

    return [SYSTEM_PROMPT]


def save_memory(messages: list) -> None:
    """Save the whole conversation so far back to disk, as plain JSON text."""
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)


def build_assistant_message(response_message) -> dict:
    """Turn the SDK's response object into a plain dict we can both send
    back to the API and save to a JSON file (see the note at the top of
    this file for why this step is necessary here but wasn't in step3).
    """
    return {
        "role": "assistant",
        "content": response_message.content,
        "tool_calls": [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in response_message.tool_calls
        ],
    }


# ---------------------------------------------------------------------------
# The agentic loop itself — same idea as step3, but now it receives and
# mutates the SAME `messages` list across multiple questions in a row,
# instead of starting a brand new list for every single question.
# ---------------------------------------------------------------------------
def run_agent(messages: list, max_iterations: int = 5) -> str:
    for iteration in range(1, max_iterations + 1):
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS,
        )
        response_message = response.choices[0].message

        # No tool calls -> final answer. Save it as a plain dict and stop.
        if not response_message.tool_calls:
            messages.append({"role": "assistant", "content": response_message.content})
            return response_message.content

        print(f"[loop iteration {iteration}] model requested {len(response_message.tool_calls)} tool call(s)")
        messages.append(build_assistant_message(response_message))

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

    return "Sorry, I couldn't complete this request within the allowed number of steps."


if __name__ == "__main__":
    print("Memory + file-reading agent demo. Type 'exit' to quit.\n")
    print("Conversation is saved to messages.json — close and reopen this")
    print("program and it will remember what you talked about.\n")

    conversation = load_memory()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        conversation.append({"role": "user", "content": user_input})
        answer = run_agent(conversation)
        print(f"Model: {answer}\n")

        save_memory(conversation)