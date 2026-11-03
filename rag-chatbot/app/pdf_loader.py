import fitz  # PyMuPDF
from pathlib import Path


def load_pdf(path: str) -> list[dict]:
    pages = []
    filename = Path(path).name
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:
                pages.append({"source": filename, "page": i, "text": text})
    return pages