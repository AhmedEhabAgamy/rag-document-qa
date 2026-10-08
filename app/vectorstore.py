"""
Thin wrapper around ChromaDB so the rest of the app never talks to the
vector DB client directly. If you swap Chroma for Qdrant/pgvector/Pinecone
later, this is the only file that should need to change.
"""
from sentence_transformers import SentenceTransformer
import chromadb

from app.config import settings
from app.ingest import Chunk

_embedder = None
_client = None
_collection = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(settings.embedding_model)
    return _embedder


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=settings.chroma_dir)
        _collection = _client.get_or_create_collection(name="documents")
    return _collection


def add_chunks(chunks: list[Chunk]) -> int:
    if not chunks:
        return 0
    embedder = get_embedder()
    collection = get_collection()

    texts = [c.text for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

    collection.add(
        ids=[c.id for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"source": c.source, "page": c.page} for c in chunks],
    )
    return len(chunks)


def query(question: str, top_k: int | None = None) -> list[dict]:
    """Returns the top_k most relevant chunks with their source metadata."""
    embedder = get_embedder()
    collection = get_collection()
    k = top_k or settings.top_k

    q_embedding = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=q_embedding, n_results=k)

    hits = []
    if results["ids"] and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            hits.append(
                {
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "source": results["metadatas"][0][i]["source"],
                    "page": results["metadatas"][0][i]["page"],
                    "distance": results["distances"][0][i],
                }
            )
    return hits


def collection_count() -> int:
    return get_collection().count()
