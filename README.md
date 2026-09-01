# AI Agent Practice

A progressive, hands-on walkthrough of building AI agents from
scratch with the DeepSeek API — no framework, just plain Python, so
every moving part is visible. Four steps, each building on the last.

## Steps

| File | What it demonstrates |
|---|---|
| [`step1_basic_chat.py`](./step1_basic_chat.py) | Plain LLM request/response — no tools. The baseline every agent builds on. |
| [`step2_tool_calling.py`](./step2_tool_calling.py) | The model decides, on its own, whether to call a `get_exchange_rate` tool and with which arguments. |
| [`step3_agentic_loop.py`](./step3_agentic_loop.py) | A real agentic loop: the model can chain multiple tool calls (exchange rate lookup + a calculator) across several iterations to answer one question. |
| [`step4_multi_agent.py`](./step4_multi_agent.py) | Multi-agent system: an orchestrator delegates to specialist sub-agents (`currency_agent`, `writer_agent`) — from the orchestrator's side, calling another agent looks exactly like calling a tool. |

## Key idea

A "tool" the model calls doesn't have to be a plain function — it can
itself be a full LLM call, with its own tools and its own loop. That's
the entire trick behind multi-agent systems: agents nested inside
agents, using the exact same tool-calling mechanism throughout.

## Install

```bash
pip install -r requirements.txt
```

## Setup

1. Copy `.env.example` to `.env`.
2. Get a free API key at [platform.deepseek.com](https://platform.deepseek.com) (API Keys section).
3. Paste it into `.env` as `DEEPSEEK_API_KEY=...`.

`.env` is listed in `.gitignore` and is never committed — keep your
key out of version control.

## Run

```bash
python step1_basic_chat.py
python step2_tool_calling.py
python step3_agentic_loop.py
python step4_multi_agent.py
```

Each is an interactive terminal chat — type a question, see the
model's reasoning (tool calls are printed as they happen), type
`exit` to quit.

## Tech stack

`openai` Python SDK (DeepSeek's API is OpenAI-compatible — we just
point `base_url` at DeepSeek's servers), `requests` for the exchange
rate lookup, `python-dotenv` for safe API key loading. No agent
framework — every loop and tool-dispatch table is hand-written, on
purpose, to make the mechanics visible.

## License

MIT — see [LICENSE](./LICENSE).
