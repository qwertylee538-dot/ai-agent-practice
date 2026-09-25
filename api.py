"""
STEP 7: Wrap the RAG agent from step6_rag.py as a real web API using
FastAPI.

WHY THIS MATTERS FOR CLIENTS:
Everything so far (steps 1-6) only runs in YOUR terminal, on YOUR
computer. A real client can't use that — they need an address on the
internet (a URL) they can send a question to from their own website,
app, or another program, and get an answer back as JSON. That's what
a web API is: a program that listens for HTTP requests and replies
with data instead of printing to a terminal.

FastAPI is the most popular Python framework for exactly this. Two
things make it especially good to learn:
  1. It's extremely common in real freelance job postings — "I need
     an API for my chatbot/app/website" is one of the most frequent
     requests you'll see.
  2. It comes with automatic, interactive documentation for free (the
     "/docs" page below) — you get a working demo page to show a
     client with almost no extra work.

HOW THIS REUSES step6_rag.py INSTEAD OF COPYING IT:
Rather than copy-pasting the whole RAG agent again, we `import
step6_rag` as a module and call its functions directly. This is a
good habit: the actual agent logic (chunking, embeddings, the
agentic loop) lives in exactly one place, and this file is only
responsible for the "web" part — turning HTTP requests into calls to
that existing logic, and turning the result back into an HTTP
response.
"""

from fastapi import FastAPI
from pydantic import BaseModel

import step6_rag

app = FastAPI(
    title="AI Agent Practice API",
    description="A small RAG-powered support agent, exposed as a web API.",
    version="1.0.0",
)


# Pydantic models describe the SHAPE of the JSON that comes in and goes
# out. FastAPI uses these to validate incoming requests automatically
# (a request missing "question", or with the wrong type, is rejected
# with a clear error BEFORE our code even runs) and to generate the
# interactive docs at /docs.
class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


@app.on_event("startup")
def load_documents() -> None:
    """Build the document index ONCE when the server starts, not on
    every single request — re-embedding all documents on every question
    would be extremely slow and pointless, since the documents don't
    change between requests.

    (Newer FastAPI code often uses a "lifespan" context manager instead
    of @app.on_event("startup") — both do the same job; on_event is
    simpler to read for a first web API.)
    """
    print("Building document index...")
    step6_rag.DOCUMENT_INDEX = step6_rag.build_index()
    print("Ready.")


@app.get("/")
def root() -> dict:
    """A simple landing endpoint, mostly so visiting the base URL in a
    browser shows something useful instead of an error."""
    return {
        "message": "AI Agent Practice API is running.",
        "docs": "/docs",
        "ask_endpoint": "POST /ask",
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """The main endpoint: send a question, get an answer grounded in the
    documents indexed at startup (see step6_rag.py for how retrieval
    and the agentic loop work).
    """
    answer = step6_rag.run_agent(request.question)
    return AskResponse(answer=answer)


if __name__ == "__main__":
    import uvicorn

    # uvicorn is the actual web server that runs our FastAPI app and
    # handles incoming HTTP connections. reload=True restarts the server
    # automatically whenever we save a code change — convenient while
    # developing, but should be turned off in production.
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)