"""
Turns a PDF into a list of overlapping text chunks ready for embedding.

Chunking is the single highest-leverage knob in a RAG system:
- too big -> retrieval pulls in irrelevant text, answers get vague
- too small -> chunks lose context, answers get fragmented / wrong
- overlap prevents an answer-relevant sentence from being cut in half
  right at a chunk boundary
"""
from dataclasses import dataclass
from pypdf import PdfReader

from app.config import settings


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int


def load_pdf_text(path: str) -> list[tuple[int, str]]:
    """Returns a list of (page_number, page_text)."""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i + 1, text))
    return pages


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Simple sliding-window chunker over characters.

    Good enough to learn on. A real system would chunk on sentence/paragraph
    boundaries (e.g. with a text splitter library) so chunks don't cut words
    or sentences awkwardly.
    """
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
    return chunks


def ingest_pdf(path: str, source_name: str) -> list[Chunk]:
    """Loads a PDF and returns chunk objects ready to embed + store."""
    pages = load_pdf_text(path)
    chunks: list[Chunk] = []
    for page_num, page_text in pages:
        for i, piece in enumerate(
            chunk_text(page_text, settings.chunk_size, settings.chunk_overlap)
        ):
            chunk_id = f"{source_name}-p{page_num}-c{i}"
            chunks.append(
                Chunk(id=chunk_id, text=piece, source=source_name, page=page_num)
            )
    return chunks
