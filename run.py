from __future__ import annotations

import logging
import os

import urllib3

from src.bot import TeamAssistantBot
from src.config import load_settings
from src.logging_setup import setup_logging
from src.pinecone_setup import ensure_pinecone_index
from src.rag import build_services
from src.state import StateStore


def main() -> None:
    setup_logging()

    # Pinecone's OpenAPI client relies on urllib3 behavior.
    # System urllib3 1.26.x can cause UnicodeEncodeError on non-latin text bodies.
    if tuple(int(p) for p in urllib3.__version__.split(".")[:2]) < (2, 0):
        raise RuntimeError(
            f"Unsupported urllib3 version: {urllib3.__version__} ({urllib3.__file__}). "
            "Install dependencies into a virtualenv and run from it, or `pip install -r requirements.txt` "
            "so urllib3>=2 is used."
        )

    settings = load_settings()

    # Ensure env for Haystack integrations
    os.environ["PINECONE_API_KEY"] = settings.pinecone_api_key
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key

    logging.info(
        "starting with pinecone_index=%s cloud=%s region=%s openai_model=%s embedding_model=%s",
        settings.pinecone_index_name,
        settings.pinecone_cloud,
        settings.pinecone_region,
        settings.openai_model,
        settings.embedding_model,
    )

    try:
        ensure_pinecone_index(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index_name,
            dimension=settings.embedding_dimension,
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
        )
    except Exception:
        logging.exception("failed to ensure Pinecone index")
        raise

    services = build_services(
        pinecone_index_name=settings.pinecone_index_name,
        pinecone_namespace="team_chat",
        embedding_dimension=settings.embedding_dimension,
        openai_model=settings.openai_model,
        embedding_model=settings.embedding_model,
        top_k=settings.top_k,
    )

    bot = TeamAssistantBot(settings=settings, services=services, state=StateStore())
    bot.run_polling()


if __name__ == "__main__":
    main()

