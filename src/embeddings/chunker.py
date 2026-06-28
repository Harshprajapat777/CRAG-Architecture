"""Chunk processed documents into LangChain `Document` objects.

Uses `RecursiveCharacterTextSplitter` with `settings.chunk_size` and
`settings.chunk_overlap`. Each emitted chunk carries metadata sufficient to
trace back to the source page and chunk position.
"""
from __future__ import annotations

from typing import Any

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - older langchain layouts
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore[no-redef]

from langchain_core.documents import Document

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Documents with fewer than this many characters of body text are skipped.
_MIN_DOC_CHARS = 50


def chunk_documents(processed_docs: list[dict[str, Any]]) -> list[Document]:
    """Split processed docs into a flat list of LangChain `Document` chunks.

    Each input dict is expected to have at least `url`, `slug`, `title`,
    `lastmod`, and `text` keys (as produced by the scraping pipeline).
    Docs with missing or very short text are logged and skipped.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    all_chunks: list[Document] = []
    for doc in processed_docs:
        text: str = (doc.get("text") or "").strip()
        url: str = doc.get("url", "")
        slug: str = doc.get("slug", "")

        if not text or len(text) < _MIN_DOC_CHARS:
            logger.warning(
                "Skipping doc with insufficient text (chars=%d): %s",
                len(text),
                url or slug or "<unknown>",
            )
            continue

        title: str = doc.get("title") or ""
        lastmod: str = doc.get("lastmod") or ""

        pieces = splitter.split_text(text)
        for i, piece in enumerate(pieces):
            metadata = {
                "source": url,
                "title": title,
                "lastmod": lastmod,
                "source_slug": slug,
                "chunk_index": i,
            }
            all_chunks.append(Document(page_content=piece, metadata=metadata))

    logger.info(
        "Chunked %d docs into %d chunks (size=%d, overlap=%d)",
        len(processed_docs),
        len(all_chunks),
        settings.chunk_size,
        settings.chunk_overlap,
    )
    return all_chunks
