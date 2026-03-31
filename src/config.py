from __future__ import annotations

import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


@dataclass(frozen=True)
class Settings:
    pinecone_api_key: str
    pinecone_index_name: str

    openai_api_key: str
    openai_model: str
    embedding_model: str
    embedding_dimension: int

    telegram_bot_token: str

    # Pinecone serverless (optional but recommended)
    pinecone_cloud: str
    pinecone_region: str

    # RAG behavior
    top_k: int
    last_messages_context: int


def _embedding_dim_from_model(model: str) -> int:
    # OpenAI known dimensions (as of 2025/2026)
    if model == "text-embedding-3-small":
        return 1536
    if model == "text-embedding-3-large":
        return 3072
    # Fallback for custom embedding models
    return _int_env("EMBEDDING_DIMENSION", 1536)

def _validate_pinecone_index_name(name: str) -> str:
    """
    Pinecone index name constraints:
    must consist of lower case alphanumeric characters or '-'.
    """
    name = (name or "").strip()
    if not name:
        raise RuntimeError("Missing PINECONE_INDEX_NAME")
    if not re.fullmatch(r"[a-z0-9-]+", name):
        raise RuntimeError(
            "Invalid PINECONE_INDEX_NAME. Pinecone requires only lower-case letters, digits, or '-'. "
            "Example: 'team-chat'."
        )
    return name


def load_settings() -> Settings:
    if not os.getenv("DOTENV_DISABLE"):
        load_dotenv()

    pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

    if not pinecone_api_key:
        raise RuntimeError("Missing PINECONE_API_KEY")
    if not openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    if not telegram_bot_token:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")

    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip()
    pinecone_index_name = _validate_pinecone_index_name(
        os.getenv("PINECONE_INDEX_NAME", "team_chat").strip() or "team_chat"
    )
    return Settings(
        pinecone_api_key=pinecone_api_key,
        pinecone_index_name=pinecone_index_name,
        openai_api_key=openai_api_key,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini",
        embedding_model=embedding_model,
        embedding_dimension=_embedding_dim_from_model(embedding_model),
        telegram_bot_token=telegram_bot_token,
        pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws").strip() or "aws",
        # Pinecone Starter/Free commonly supports aws us-east-1 only.
        pinecone_region=os.getenv("PINECONE_REGION", "us-east-1").strip() or "us-east-1",
        top_k=_int_env("TOP_K", 8),
        last_messages_context=_int_env("LAST_MESSAGES_CONTEXT", 100),
    )

