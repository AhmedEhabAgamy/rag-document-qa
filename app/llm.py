"""
The only file that should need to change if you swap LLM providers.
Keeping the generation call isolated like this is a real deployment
discipline: your API layer and retrieval logic shouldn't care whether
the answer came from a local model, Claude, GPT, or anything else.

This version uses Ollama (https://ollama.com) — a tool that runs open-source
LLMs locally on your own machine, completely free, with no API key and no
billing. Ollama must be installed and running separately (see README), and
it exposes a local HTTP API on port 11434 that this file calls.
"""
import requests

from app.config import settings

SYSTEM_PROMPT = """You are a precise assistant that answers questions using ONLY \
the provided context excerpts. If the context does not contain the answer, say \
you don't have enough information — never make something up. When you use a \
piece of context, mention which source/page it came from."""


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    context_block = "\n\n".join(
        f"[Source: {c['source']}, page {c['page']}]\n{c['text']}"
        for c in context_chunks
    )

    user_message = f"""Context excerpts:
{context_block}

Question: {question}

Answer using only the context above. Cite sources like (source, page)."""

    try:
        response = requests.post(
            f"{settings.ollama_host}/api/chat",
            json={
                "model": settings.llm_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        return (
            "Could not reach Ollama. Make sure it's installed and running "
            f"(`ollama serve`), and that the model '{settings.llm_model}' has "
            f"been pulled (`ollama pull {settings.llm_model}`)."
        )