"""
STEP 4: Multi-agent system — an orchestrator that delegates to
specialized sub-agents.

THE KEY INSIGHT (read this before the code):
From the orchestrator's point of view, "call another agent" looks
EXACTLY like "call a tool" — same TOOLS list format, same tool_calls
handling loop we already built in step3. The only difference is what
happens INSIDE the function: instead of a plain calculation or an API
request, the function itself makes another full LLM call (sometimes
with its own tools and its own loop, like our currency_agent below).

So a multi-agent system is not a new technology — it's the exact same
agentic loop from step3, just with some of the "tools" being other
agents instead of plain functions. Agents nested inside agents.

WHAT WE'RE BUILDING:
- `currency_agent(question)` — a narrow specialist. Its OWN system
  prompt only talks about currency/math, and it has its own two tools
  (reused from step3) and its own mini agentic loop.
- `writer_agent(topic)` — an even simpler specialist. No tools at all,
  its only job is to write nicely phrased text on a given topic.
- `orchestrator(user_question)` — decides, using tool calling, whether
  to hand the question to `currency_agent`, to `writer_agent`, or (if
  neither fits) answer directly itself.
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
# SPECIALIST 1: currency_agent — reuses the exact tools + loop pattern
# from step3, but packaged as a single function the orchestrator can
# call like any other tool.
# ---------------------------------------------------------------------------
def _get_exchange_rate(base_currency: str, target_currency: str) -> str:
    try:
        response = requests.get(
            f"https://open.er-api.com/v6/latest/{base_currency.upper()}",
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("result") != "success":
            return f"Exchange rate service error: {data.get('error-type', 'unknown')}"
        rate = data.get("rates", {}).get(target_currency.upper())
        if rate is None:
            return f"Could not find a rate for {target_currency}."
        return f"1 {base_currency.upper()} = {rate} {target_currency.upper()}"
    except requests.exceptions.RequestException as exc:
        return f"Error fetching exchange rate: {exc}"


def _calculate(expression: str) -> str:
    import re

    if not re.match(r"^[0-9\.\s\+\-\*/\(\)]+$", expression):
        return "Error: expression contains characters that are not allowed."
    try:
        return str(eval(expression))
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


_CURRENCY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": "Get the current exchange rate between two currencies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "base_currency": {"type": "string"},
                    "target_currency": {"type": "string"},
                },
                "required": ["base_currency", "target_currency"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a simple arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]
_CURRENCY_FUNCTIONS = {"get_exchange_rate": _get_exchange_rate, "calculate": _calculate}


def currency_agent(question: str) -> str:
    """A narrow specialist agent: only currency conversion and math.

    This function is a COMPLETE mini agentic loop on its own — notice
    it has its own `messages`, its own system prompt, and its own
    while-style loop over iterations. From the orchestrator's side,
    none of that complexity is visible — it just sees "I called
    currency_agent and got a string back", same as any other tool.
    """
    messages = [
        {
            "role": "system",
            "content": "You only handle currency conversion and arithmetic. Use the tools provided.",
        },
        {"role": "user", "content": question},
    ]

    for _ in range(5):
        response = client.chat.completions.create(
            model="deepseek-chat", messages=messages, tools=_CURRENCY_TOOLS
        )
        msg = response.choices[0].message
        if not msg.tool_calls:
            return msg.content
        messages.append(msg)
        for tool_call in msg.tool_calls:
            fn = _CURRENCY_FUNCTIONS[tool_call.function.name]
            args = json.loads(tool_call.function.arguments)
            result = fn(**args)
            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": result}
            )
    return "currency_agent: too many steps, giving up."


# ---------------------------------------------------------------------------
# SPECIALIST 2: writer_agent — no tools at all, just a narrow system
# prompt. This shows that a "specialist agent" doesn't need tools to
# be useful — sometimes the specialization is purely in the prompt.
# ---------------------------------------------------------------------------
def writer_agent(topic: str) -> str:
    """A narrow specialist: writes a short, well-phrased piece of text."""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a skilled copywriter. Write a short, engaging "
                    "piece of text (2-4 sentences) on the given topic. "
                    "No preamble, just the text itself."
                ),
            },
            {"role": "user", "content": topic},
        ],
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# ORCHESTRATOR: decides which specialist(s) to call, using the exact
# same tool-calling mechanism as step2/step3 — except here, each
# "tool" is actually one of the specialist agents above.
# ---------------------------------------------------------------------------
_ORCHESTRATOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "currency_agent",
            "description": (
                "Delegate to a specialist for currency conversion and math "
                "questions (exchange rates, converting amounts, arithmetic)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The user's question, verbatim or rephrased"},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "writer_agent",
            "description": (
                "Delegate to a specialist copywriter for requests to write, "
                "phrase, or draft a short piece of text on some topic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "What to write about"},
                },
                "required": ["topic"],
            },
        },
    },
]

_ORCHESTRATOR_AGENTS = {
    "currency_agent": currency_agent,
    "writer_agent": writer_agent,
}


def orchestrator(user_question: str, max_iterations: int = 5) -> str:
    """The top-level agent. Decides whether to delegate, to which
    specialist(s), or to just answer directly if neither fits.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a dispatcher. For currency/math questions, delegate "
                "to currency_agent. For writing requests, delegate to "
                "writer_agent. For anything else, answer directly yourself."
            ),
        },
        {"role": "user", "content": user_question},
    ]

    for iteration in range(1, max_iterations + 1):
        response = client.chat.completions.create(
            model="deepseek-chat", messages=messages, tools=_ORCHESTRATOR_TOOLS
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            return msg.content

        messages.append(msg)
        for tool_call in msg.tool_calls:
            agent_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            print(f"[orchestrator] delegating to {agent_name}({args})")

            agent_function = _ORCHESTRATOR_AGENTS[agent_name]
            # This call can itself take several seconds and involve
            # multiple LLM round-trips inside it (e.g. currency_agent's
            # own loop) — the orchestrator doesn't know or care.
            result = agent_function(**args)

            print(f"[orchestrator] {agent_name} returned: {result[:120]}")

            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": result}
            )

    return "orchestrator: too many steps, giving up."


if __name__ == "__main__":
    print("Multi-agent demo (orchestrator + currency_agent + writer_agent).")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        answer = orchestrator(user_input)
        print(f"Model: {answer}\n")
