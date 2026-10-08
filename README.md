# RAG Document Q&A — Learning Project

A minimal but *real* RAG service: upload PDFs, ask questions, get grounded answers
with citations back to the source chunks. Built to be a vehicle for learning
Docker, deployment, LLMs/RAG, and basic MLOps — not a toy notebook.

## Stack
- **FastAPI** — the API layer (`/upload`, `/ask`, `/health`)
- **ChromaDB** — local vector store (persisted to disk / a volume)
- **sentence-transformers** — embeddings, runs locally, no API key needed
- **Ollama** — generation step, runs an open-source LLM locally on your machine, completely free, no API key or billing (swap providers easily, see `app/llm.py`)
- **Docker + docker-compose** — containerization
- **A tiny eval harness** — the MLOps piece: a fixed question set scored on every change

## Project layout
```
rag-project/
├── app/
│   ├── main.py         # FastAPI app: routes
│   ├── config.py       # settings (env vars)
│   ├── ingest.py        # PDF loading + chunking
│   ├── vectorstore.py   # Chroma wrapper (add / query)
│   ├── llm.py           # LLM call (Claude), swappable
│   ├── rag.py           # ties retrieval + generation together
│   └── logger.py        # structured query logging (MLOps step 1)
├── eval/
│   ├── eval_set.json    # fixed Q/A pairs to test against
│   └── run_eval.py      # scores current pipeline against eval_set.json
├── data/
│   ├── uploads/         # uploaded PDFs land here
│   └── chroma/          # persisted vector DB
├── logs/                # query logs (created at runtime)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

## Quickstart (local, no Docker yet)

**0. Install Ollama first** (this is what runs the LLM, for free, on your own machine):
- Download from https://ollama.com and install it for your OS
- Pull a small model: `ollama pull llama3.2` (about 2GB, runs fine on CPU)
- Leave it running — Ollama runs as a background service on `localhost:11434`

**1. Set up the project:**
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# the defaults in .env already point at Ollama — no key needed

uvicorn app.main:app --reload --port 8000
```

If `/ask` ever returns a message saying it couldn't reach Ollama, it means
Ollama isn't running or the model hasn't been pulled yet — run `ollama serve`
and `ollama pull llama3.2` in a terminal and try again.

Then in another terminal:

```bash
# upload a PDF
curl -X POST http://localhost:8000/upload -F "file=@/path/to/some.pdf"

# ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

## Run with Docker

```bash
docker compose up --build
```

Same endpoints, now containerized. `data/` and `logs/` are mounted as volumes so
your vector DB and logs survive container restarts — this is the first real
"deployment" lesson: state has to live outside the container.

## Run the eval

```bash
python eval/run_eval.py
```

This is your first MLOps habit: every time you change the chunk size, the
embedding model, or the prompt, run this and see if your score moves. Add
real questions/answers from your own documents to `eval/eval_set.json` as you go.

## Suggested next steps (in order)
1. Get it running locally, upload a real PDF, ask real questions. Read the logs.
2. Break something on purpose (tiny chunk size, wrong top-k) and see how answers degrade.
3. Add a `/metrics` endpoint (latency, token count, cost per query) — real MLOps.
4. Swap `sentence-transformers` for a different embedding model and re-run the eval — this is what "versioning your embeddings" means in practice.
5. Deploy the Docker container to a free-tier VM (Render, Railway, an Oracle free VM, whatever) so it's reachable from outside your laptop.
6. Add basic auth + rate limiting so it isn't a wide-open endpoint once it's public.

## Notes for learning, not just running
- `app/llm.py` is intentionally isolated — swapping models/providers should
  only ever touch this one file. That's a deployment/MLOps discipline, not
  just tidiness.
- `app/logger.py` writes one JSON line per query (question, retrieved chunk
  IDs, answer, latency, timestamp). This is the seed of real observability —
  in production this would go to a proper logging/monitoring stack instead
  of a local file.
- Chunk size and top-k are read from `.env`, not hardcoded, so you can
  experiment without touching code — every RAG project lives or dies on
  these two numbers.
