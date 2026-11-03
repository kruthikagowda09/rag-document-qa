from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

load_dotenv()

from . import rag, vector_store  # noqa: E402
from .chunker import chunk_pages  # noqa: E402
from .pdf_loader import load_pdf  # noqa: E402

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app):
    vector_store._get_model()
    vector_store._get_collection()
    vector_store.search("warmup", 1)
    yield


app = FastAPI(title="RAG Document Q&A", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    k: int = 5


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    results = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            results.append({"file": f.filename, "error": "Only PDF files are supported"})
            continue

        name = Path(f.filename).name
        dest = UPLOAD_DIR / name
        dest.write_bytes(await f.read())

        pages = load_pdf(str(dest))
        if not pages:
            results.append({"file": name, "error": "No extractable text (scanned PDF?)"})
            continue

        vector_store.delete_document(name)
        chunks = chunk_pages(pages)
        vector_store.add_chunks(chunks)
        results.append({"file": name, "pages": len(pages), "chunks": len(chunks)})
    return {"results": results}


@app.post("/ask")
async def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty")
    try:
        return rag.ask(req.question, req.k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def documents():
    return {"documents": vector_store.list_documents()}


@app.delete("/documents/{name}")
async def delete_document(name: str):
    vector_store.delete_document(name)
    path = UPLOAD_DIR / name
    if path.exists():
        path.unlink()
    return {"deleted": name}