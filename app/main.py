import os
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import settings
from app.ingest import ingest_pdf
from app import vectorstore, rag

app = FastAPI(title="RAG Document Q&A", version="0.1.0")


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "documents_indexed": vectorstore.collection_count(),
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported right now.")

    os.makedirs(settings.upload_dir, exist_ok=True)
    dest_path = os.path.join(settings.upload_dir, file.filename)
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks = ingest_pdf(dest_path, source_name=file.filename)
    added = vectorstore.add_chunks(chunks)

    return {
        "filename": file.filename,
        "chunks_added": added,
        "total_chunks_indexed": vectorstore.collection_count(),
    }


@app.post("/ask")
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    return rag.answer_question(req.question, top_k=req.top_k)


@app.get("/documents")
def list_documents():
    return {"documents": vectorstore.list_sources()}


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    deleted_count = vectorstore.delete_by_source(filename)
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail=f"No chunks found for '{filename}'.")

    file_path = os.path.join(settings.upload_dir, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    return {"filename": filename, "chunks_removed": deleted_count}

# Serves static/index.html at "/" — must come after the API routes above,
# since FastAPI matches routes in definition order.
app.mount("/", StaticFiles(directory="static", html=True), name="static")