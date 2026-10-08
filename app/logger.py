"""
The seed of observability. Every query gets one JSON line: what was asked,
what was retrieved, what came back, and how long it took. In a real
production system this would flow into something like a logging pipeline
or a monitoring dashboard instead of a flat file — but the shape of the
data you'd want to capture doesn't change.
"""
import json
import os
import time
from datetime import datetime, timezone

from app.config import settings


def log_query(question: str, hits: list[dict], answer: str, latency_ms: float) -> None:
    os.makedirs(settings.log_dir, exist_ok=True)
    log_path = os.path.join(settings.log_dir, "queries.jsonl")

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "retrieved_chunk_ids": [h["id"] for h in hits],
        "retrieved_sources": list({h["source"] for h in hits}),
        "answer": answer,
        "latency_ms": round(latency_ms, 1),
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")
