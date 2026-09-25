# AI Agent Practice

A progressive, hands-on walkthrough of building AI agents from
scratch with the DeepSeek API — no framework, just plain Python, so
every moving part is visible. Six steps, each building on the last.

## Steps

| File | What it demonstrates |
|---|---|
| [`step1_basic_chat.py`](./step1_basic_chat.py) | Plain LLM request/response — no tools. The baseline every agent builds on. |
| [`step2_tool_calling.py`](./step2_tool_calling.py) | The model decides, on its own, whether to call a `get_exchange_rate` tool and with which arguments. |
| [`step3_agentic_loop.py`](./step3_agentic_loop.py) | A real agentic loop: the model can chain multiple tool calls (exchange rate lookup + a calculator) across several iterations to answer one question. |
| [`step4_multi_agent.py`](./step4_multi_agent.py) | Multi-agent system: an orchestrator delegates to specialist sub-agents (`currency_agent`, `writer_agent`) — from the orchestrator's side, calling another agent looks exactly like calling a tool. |
| [`step5_memory.py`](./step5_memory.py) | Persistent memory: the conversation is saved to `messages.json` after every turn and reloaded on startup, so the agent remembers past sessions. Also adds a `read_file` tool, so it can read and answer questions about a local text file. |
| [`step6_rag.py`](./step6_rag.py) | RAG (Retrieval-Augmented Generation): documents in `documents/` are chunked and embedded with a free local model (`sentence-transformers`), then searched by semantic similarity so the agent can answer questions about documents far too large to fit in one prompt — and honestly says "I don't know" when the answer isn't in the documents. |

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
python step5_memory.py
python step6_rag.py
```

Each is an interactive terminal chat — type a question, see the
model's reasoning (tool calls are printed as they happen), type
`exit` to quit.

## Memory

`step5_memory.py` saves the running conversation to `messages.json`
after every answer. Close the program and run it again — it picks up
right where it left off, with no memory of your questions lost.
`messages.json` is listed in `.gitignore` (it's personal conversation
history, not code) so it's never committed to the repo.

## RAG / document search

`step6_rag.py` indexes every `.txt` file in the `documents/` folder:
each file is split into chunks, and each chunk is turned into an
embedding by a small free local model (no API key or cost — it runs
on your own computer, downloading its weights once on first run).
Questions are answered by finding the most semantically similar
chunks and handing only those to the model, instead of the whole
document. Drop your own `.txt` files into `documents/` to try it on
different content.

## Tech stack

`openai` Python SDK (DeepSeek's API is OpenAI-compatible — we just
point `base_url` at DeepSeek's servers), `requests` for the exchange
rate lookup, `python-dotenv` for safe API key loading, `sentence-transformers`
and `numpy` for local embeddings and similarity search in step6. No
agent framework — every loop and tool-dispatch table is hand-written,
on purpose, to make the mechanics visible.

## License

MIT — see [LICENSE](./LICENSE).