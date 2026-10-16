"""
The only file that should need to change if you swap LLM providers.
Keeping the generation call isolated like this is a real deployment
discipline: your API layer and retrieval logic shouldn't care whether
the answer came from a local model, Groq, Claude, or anything else.

This version uses Groq (https://console.groq.com) — a free, hosted API
that runs open-source models (Llama, etc.) on very fast custom hardware.
Free tier: no credit card, 30K tokens/minute, 14,400 requests/day.
Groq's API is OpenAI-compatible, so this is a plain HTTP POST — no special
SDK needed.
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
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.groq_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.llm_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        if response.status_code == 401:
            return "Groq API key is missing or invalid. Check GROQ_API_KEY in your .env file."
        if response.status_code == 429:
            return "Hit Groq's free tier rate limit — wait a moment and try again."
        return f"Groq API error: {e}"
    except requests.exceptions.RequestException as e:
        return f"Could not reach Groq: {e}"