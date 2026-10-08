"""
Ties retrieval (vectorstore) and generation (llm) together, and logs
each query. This is the function the API route calls — it's also the
function the eval harness calls, so eval and production use the exact
same code path (no silent drift between "what we test" and "what we ship").
"""
import time

from app import vectorstore, llm, logger


def answer_question(question: str, top_k: int | None = None) -> dict:
    start = time.perf_counter()

    hits = vectorstore.query(question, top_k=top_k)

    if not hits:
        answer = "I don't have any documents to search yet — upload one first."
    else:
        answer = llm.generate_answer(question, hits)

    latency_ms = (time.perf_counter() - start) * 1000
    logger.log_query(question, hits, answer, latency_ms)

    seen = set()
    unique_sources = []
    for h in hits:
        key = (h["source"], h["page"])
        if key not in seen:
            seen.add(key)
            unique_sources.append({"source": h["source"], "page": h["page"]})

    return {
        "answer": answer,
        "sources": unique_sources,
        "latency_ms": round(latency_ms, 1),
    }
