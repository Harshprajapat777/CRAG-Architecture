"""ChromaDB persistence layer for the CRAG knowledge base.

Thin wrapper around `langchain_chroma.Chroma` that ensures the persist
directory exists, lazily constructs an embedder when one isn't supplied,
and exposes deterministic-id upsert + similarity search helpers.
"""
from __future__ import annotations

from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import settings
from src.embeddings.embedder import get_embedder
from src.utils.io import ensure_dirs
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_store(embedder: Optional[object] = None) -> Chroma:
    """Return a `Chroma` store backed by `settings.vector_store_dir`.

    When `embedder` is omitted, a new `OpenAIEmbeddings` is constructed via
    `get_embedder()`. The persist directory is created if it does not exist.
    """
    if embedder is None:
        embedder = get_embedder()

    ensure_dirs(settings.vector_store_dir)

    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embedder,
        persist_directory=str(settings.vector_store_dir),
    )


def upsert_documents(store: Chroma, docs: list[Document]) -> int:
    """Add (or replace, by id) `docs` in `store`.

    IDs are derived deterministically from `source_slug` + `chunk_index` so
    repeated runs upsert rather than duplicate.
    """
    if not docs:
        logger.info("No documents to upsert.")
        return 0

    ids = [
        f"{doc.metadata['source_slug']}::{doc.metadata['chunk_index']}"
        for doc in docs
    ]
    store.add_documents(docs, ids=ids)
    logger.info("Upserted %d chunks into collection '%s'", len(docs), settings.chroma_collection_name)
    return len(docs)


def similarity_search(store: Chroma, query: str, k: int = 4) -> list[Document]:
    """Return the top-`k` most similar chunks for `query`."""
    return store.similarity_search(query, k=k)
