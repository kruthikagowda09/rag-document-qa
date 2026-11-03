from collections import Counter
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = str(Path(__file__).resolve().parent.parent / "data" / "chroma")
EMBED_MODEL = "all-MiniLM-L6-v2"

_model = None
_collection = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=DB_PATH)
        _collection = client.get_or_create_collection(
            name="documents", metadata={"hnsw:space": "cosine"}
        )
    return _collection


def add_chunks(chunks: list[dict]) -> int:
    if not chunks:
        return 0
    col = _get_collection()
    texts = [c["text"] for c in chunks]
    embeddings = _get_model().encode(texts, show_progress_bar=False).tolist()
    col.upsert(
        ids=[c["id"] for c in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
    )
    return len(chunks)


def search(query: str, k: int = 5) -> list[dict]:
    col = _get_collection()
    total = col.count()
    if total == 0:
        return []
    q_emb = _get_model().encode([query], show_progress_bar=False).tolist()
    res = col.query(query_embeddings=q_emb, n_results=min(k, total))
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({
            "text": doc,
            "source": meta["source"],
            "page": meta["page"],
            "score": round(1 - dist, 3),
        })
    return hits


def list_documents() -> list[dict]:
    col = _get_collection()
    metas = col.get(include=["metadatas"])["metadatas"]
    counts = Counter(m["source"] for m in metas)
    return [{"source": s, "chunks": n} for s, n in sorted(counts.items())]


def delete_document(source: str) -> None:
    _get_collection().delete(where={"source": source})