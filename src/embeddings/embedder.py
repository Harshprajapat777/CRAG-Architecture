"""OpenAI embedding model factory.

Centralizes construction of the `OpenAIEmbeddings` client so the rest of the
pipeline doesn't have to know about credentials or model names.
"""
from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from config.settings import settings


_PLACEHOLDER_PREFIX = "sk-..."


def get_embedder() -> OpenAIEmbeddings:
    """Return a configured `OpenAIEmbeddings` instance.

    Raises a `RuntimeError` with an actionable message if the API key is
    missing or still set to the placeholder value from `.env.example`.
    """
    api_key = (settings.openai_api_key or "").strip()
    if not api_key or api_key.startswith(_PLACEHOLDER_PREFIX):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Edit your .env file and provide a real "
            "OpenAI API key (the value must not be empty or the 'sk-...' "
            "placeholder from .env.example)."
        )

    return OpenAIEmbeddings(model=settings.embedding_model, api_key=api_key)
