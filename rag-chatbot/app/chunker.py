def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = " ".join(text.split())
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start + chunk_size // 2, end)
            if cut != -1:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_pages(pages: list[dict], chunk_size: int = 1000, overlap: int = 200) -> list[dict]:
    chunks = []
    for p in pages:
        for idx, piece in enumerate(_split_text(p["text"], chunk_size, overlap)):
            chunks.append({
                "id": f"{p['source']}-p{p['page']}-c{idx}",
                "text": piece,
                "source": p["source"],
                "page": p["page"],
            })
    return chunks