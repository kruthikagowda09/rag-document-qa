import sys

from app.chunker import chunk_pages
from app.pdf_loader import load_pdf

path = sys.argv[1] if len(sys.argv) > 1 else "data/uploads/sample.pdf"

pages = load_pdf(path)
print(f"Extracted {len(pages)} pages")
if pages:
    print(pages[0]["source"], "- page", pages[0]["page"])
    print(pages[0]["text"][:500])

    chunks = chunk_pages(pages)
    print(f"\nCreated {len(chunks)} chunks")
    print(chunks[0]["id"])
    print(chunks[0]["text"][:300])