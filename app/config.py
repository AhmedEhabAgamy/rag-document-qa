"""
All tunable knobs live here and are read from environment variables.
Changing chunk size / top_k / model without touching code is a small
but real MLOps habit: it makes experiments reproducible and diffable
(you can log exactly which .env produced which eval score).
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_host: str = "http://localhost:11434"
    llm_model: str = "llama3.2"

    embedding_model: str = "all-MiniLM-L6-v2"

    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4

    chroma_dir: str = "data/chroma"
    upload_dir: str = "data/uploads"
    log_dir: str = "logs"


settings = Settings()
