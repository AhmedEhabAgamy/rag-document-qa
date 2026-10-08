FROM python:3.11-slim

WORKDIR /app

# System deps needed by some pypdf / sentence-transformers wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only torch first, from PyTorch's own CPU wheel index. This skips
# the multi-gigabyte NVIDIA CUDA libraries entirely — we only run embeddings
# on CPU here, so they'd be wasted bandwidth and wasted disk space.
RUN pip install --no-cache-dir --default-timeout=120 --retries 10 \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --default-timeout=120 --retries 10 -r requirements.txt

COPY app/ ./app/
COPY static/ ./static/

# data/ and logs/ are meant to be mounted as volumes (see docker-compose.yml)
# so the vector DB and query logs survive container restarts/rebuilds.
RUN mkdir -p data/uploads data/chroma logs

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
