# AI Agent Practice

A progressive, hands-on walkthrough of building AI agents from
scratch with the DeepSeek API — no framework, just plain Python, so
every moving part is visible. Five steps, each building on the last.

## Steps

| File | What it demonstrates |
|---|---|
| [`step1_basic_chat.py`](./step1_basic_chat.py) | Plain LLM request/response — no tools. The baseline every agent builds on. |
| [`step2_tool_calling.py`](./step2_tool_calling.py) | The model decides, on its own, whether to call a `get_exchange_rate` tool and with which arguments. |
| [`step3_agentic_loop.py`](./step3_agentic_loop.py) | A real agentic loop: the model can chain multiple tool calls (exchange rate lookup + a calculator) across several iterations to answer one question. |
| [`step4_multi_agent.py`](./step4_multi_agent.py) | Multi-agent system: an orchestrator delegates to specialist sub-agents (`currency_agent`, `writer_agent`) — from the orchestrator's side, calling another agent looks exactly like calling a tool. |
| [`step5_memory.py`](./step5_memory.py) | Persistent memory: the conversation is saved to `messages.json` after every turn and reloaded on startup, so the agent remembers past sessions. Also adds a