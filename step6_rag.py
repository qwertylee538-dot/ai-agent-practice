"""
STEP 6: RAG (Retrieval-Augmented Generation) — let the agent answer
questions about LARGE documents it could never fit into one prompt.

THE PROBLEM WITH step5's read_file TOOL:
read_file dumps the ENTIRE file into the conversation, then lets the
model read all of it. That's fine for a short notes.txt, but falls
apart fast:
  - A real company knowledge base might be hundreds of pages. That
    won't fit in the model's context window at all.
  - Even if it did fit, you'd be paying (in API cost and speed) to
    send the whole document on EVERY question, even one that only
    needs a single paragraph's worth of information.

THE FIX — RETRIEVAL:
Instead of sending the whole document, we:
  1. Split every document into small chunks (a few sentences each).
  2. Convert each chunk into an "embedding" — a list of numbers that
     represents its MEANING (not just its exact words). This is done
     by a small, free, local AI model (sentence-transformers), so
     there's no extra API cost and no API key needed for this part.
  3. When a question comes in, we embed the question the same way,
     and compare it against every chunk's embedding using cosine
     similarity — a standard way to measure "how close are these two
     meanings" (1.0 = identical meaning, 0 = unrelated).
  4. We hand the model only the TOP FEW most relevant chunks, not the
     whole document, as the result of a `search_documents` tool call.
     The model then writes a normal-language answer using just that.

This is "Retrieval-Augmented Generation": the model's answer is
"augmented" with retrieved information it didn't already know, fetched
just-in-time instead of stuffed into every prompt. It's the technique
behind basically every "chat with your documents" / "company knowledge
base assistant" product on the market.

WHY A HAND-ROLLED VECTOR STORE, NOT A REAL DATABASE:
Production systems use a dedicated vector database (Pinecone, Chroma,
Qdrant, etc.) that can hold millions of chunks and search them fast.
Here we just keep a plain Python list of (chunk, embedding) pairs and
compare against every single one — that's O(n), too slow for millions
of documents, but perfectly fine for a folder of a few files, and it
makes the actual mechanism (embed -> compare -> pick top matches)
completely visible, which is the whole point of this series.
"""

import glob
import json
import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError("DEEPSEEK_API_KEY not found — check your .env file.")

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# This downloads a small (~90 MB) free embedding model the first time
# it runs, then reuses the cached copy on every run after that. No API
# key needed — it runs entirely on this computer.
print("Loading embedding model (first run downloads it, ~90 MB)...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

DOCUMENTS_FOLDER = "documents"
CHUNK_SIZE = 500  # characters per chunk — small enough to be a focused answer, big enough for context


# ---------------------------------------------------------------------------
# Building the "vector store": chunk every document, embed every chunk.
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list:
    """Split text into paragraph-based chunks of roughly chunk_size
    characters. We split on blank lines (paragraph breaks) first, so we
    don't cut a sentence in half — then group small paragraphs together
    until we hit the size limit.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current_chunk = ""
    for paragraph in paragraphs:
        if current_chunk and len(current_chunk) + len(paragraph) > chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            current_chunk = f"{current_chunk}\n\n{paragraph}" if current_chunk else paragraph

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def build_index() -> list:
    """Read every .txt file in DOCUMENTS_FOLDER, split each into chunks,
    and embed every chunk. Returns a list of dicts: {text, source, embedding}.

    This runs once at startup. A real system would save this to disk (or
    a proper vector database) so it doesn't have to re-embed everything
    on every run — we keep it simple here since our test document is small.
    """
    index = []
    file_paths = glob.glob(os.path.join(DOCUMENTS_FOLDER, "*.txt"))

    if not file_paths:
        print(f"(no .txt files found in '{DOCUMENTS_FOLDER}/' — nothing to index)")
        return index

    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)
        # encode() can take a whole list of chunks at once — much faster
        # than embedding them one at a time in a loop.
        embeddings = embedding_model.encode(chunks)

        for chunk_text_value, embedding in zip(chunks, embeddings):
            index.append(
                {
                    "text": chunk_text_value,
                    "source": os.path.basename(file_path),
                    "embedding": embedding,
                }
            )

        print(f"Indexed {len(chunks)} chunk(s) from {os.path.basename(file_path)}")

    return index


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Standard cosine similarity: how close two vectors point in the
    same direction, ignoring their length. 1.0 = same direction (same
    meaning), 0.0 = unrelated, -1.0 = opposite meaning.
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# DOCUMENT_INDEX is built once when the script starts (see the bottom of
# this file) and reused for every question in this session.
DOCUMENT_INDEX: list = []


def search_documents(query: str, top_k: int = 3) -> str:
    """Embed the query, compare it against every indexed chunk, and
    return the top_k most similar chunks as one text block, each
    labelled with its source file and similarity score. This is the
    function the agent calls as a tool.
    """
    if not DOCUMENT_INDEX:
        return "No documents have been indexed (the documents/ folder is empty)."

    query_embedding = embedding_model.encode(query)

    scored_chunks = [
        (cosine_similarity(query_embedding, entry["embedding"]), entry)
        for entry in DOCUMENT_INDEX
    ]
    scored_chunks.sort(key=lambda pair: pair[0], reverse=True)

    top_results = scored_chunks[:top_k]

    result_parts = []
    for score, entry in top_results:
        result_parts.append(
            f"[from {entry['source']}, relevance {score:.2f}]\n{entry['text']}"
        )

    return "\n\n---\n\n".join(result_parts)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the indexed documents for text relevant to a question. "
                "Always use this before answering any question that could be "
                "about the company's policies, products, or procedures — never "
                "guess or make up an answer. Returns the most relevant excerpts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for — usually the user's question itself.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

AVAILABLE_FUNCTIONS = {"search_documents": search_documents}


# ---------------------------------------------------------------------------
# The agentic loop — same shape as step3/step5, but with one tool:
# search_documents. The model decides on its own when to search and can
# search again with a different query if the first result wasn't useful.
# ---------------------------------------------------------------------------
def run_agent(question: str, max_iterations: int = 5) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a support assistant that answers ONLY using information "
                "found via the search_documents tool. If the documents don't "
                "contain the answer, say so honestly instead of guessing. "
                "Answer in the same language the user asked in."
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

        if not response_message.tool_calls:
            return response_message.content

        print(f"[loop iteration {iteration}] model requested {len(response_message.tool_calls)} tool call(s)")

        messages.append(
            {
                "role": "assistant",
                "content": response_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in response_message.tool_calls
                ],
            }
        )

        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"  -> {function_name}({function_args})")
            function_to_call = AVAILABLE_FUNCTIONS[function_name]
            function_result = function_to_call(**function_args)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": function_result,
                }
            )

    return "Sorry, I couldn't complete this request within the allowed number of steps."


if __name__ == "__main__":
    print("RAG demo — ask questions about the documents in documents/. Type 'exit' to quit.\n")

    DOCUMENT_INDEX = build_index()
    print()

    if not DOCUMENT_INDEX:
        print("Add some .txt files to the documents/ folder and run this again.")
    else:
        print("Try asking: 'Сколько дней на возврат бракованного товара?'\n")

        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ("exit", "quit"):
                break
            if not user_input:
                continue

            answer = run_agent(user_input)
            print(f"Model: {answer}\n")