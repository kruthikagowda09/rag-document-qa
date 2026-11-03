import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from . import vector_store

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash")

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered context passages provided. "
    "Cite the passages you use inline like [1] or [2][3]. "
    "If the context does not contain the answer, say you could not find it in the "
    "uploaded documents. Do not use outside knowledge."
)

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY"),
            http_options=types.HttpOptions(timeout=20000),
        )
    return _client


def _generate(prompt: str):
    """Try the main model twice, then the backup model twice."""
    last_error = None
    for model in (MODEL, FALLBACK_MODEL, FALLBACK_MODEL):
        try:
            return _get_client().models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.2,
                ),
            )
        except Exception as e:
            last_error = e
            print(f"ASK: {model} failed: {e}", flush=True)
            time.sleep(2)
    raise last_error


def ask(question: str, k: int = 5) -> dict:
    t0 = time.time()
    print("ASK: searching...", flush=True)
    hits = vector_store.search(question, k)
    print(f"ASK: search done in {time.time() - t0:.1f}s", flush=True)

    if not hits:
        return {"answer": "No documents found. Please upload a PDF first.", "sources": []}

    context = "\n\n".join(
        f"[{i}] (Source: {h['source']}, page {h['page']})\n{h['text']}"
        for i, h in enumerate(hits, start=1)
    )

    print("ASK: calling Gemini...", flush=True)
    response = _generate(f"Context:\n{context}\n\nQuestion: {question}")
    print(f"ASK: Gemini done in {time.time() - t0:.1f}s", flush=True)
    answer = response.text or "The model returned no answer."

    sources = [
        {
            "id": i,
            "source": h["source"],
            "page": h["page"],
            "score": h["score"],
            "snippet": h["text"][:250],
        }
        for i, h in enumerate(hits, start=1)
    ]
    return {"answer": answer, "sources": sources}