"""Build the ChromaDB vector store from processed documents.

Reads every `*.json` (excluding `_failures.json`) under
`settings.processed_dir`, chunks them, prints a rough OpenAI embedding cost
estimate, then upserts the chunks into the persistent Chroma collection.

This script issues paid OpenAI API calls when it embeds; the cost estimate
is logged BEFORE any embedding request is made.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running this script directly: ensure project root is importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings  # noqa: E402
from src.embeddings.chunker import chunk_documents  # noqa: E402
from src.embeddings.embedder import get_embedder  # noqa: E402
from src.utils.io import read_json  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.vector_store.chroma_store import get_store, upsert_documents  # noqa: E402

logger = get_logger(__name__)

# Pricing assumption for text-embedding-3-small: $0.02 per 1M tokens.
# Tokenization heuristic: ~4 characters per token.
_USD_PER_MILLION_TOKENS = 0.02
_CHARS_PER_TOKEN = 4


def _collect_processed_files() -> list[Path]:
    processed_dir = settings.processed_dir
    if not processed_dir.exists():
        return []
    return sorted(
        p for p in processed_dir.glob("*.json") if p.name != "_failures.json"
    )


def main() -> None:
    files = _collect_processed_files()
    if not files:
        logger.error(
            "No processed documents found in %s. Run scripts/01 and scripts/02 first.",
            settings.processed_dir,
        )
        sys.exit(1)

    processed_docs: list[dict] = [read_json(path) for path in files]
    total_chars = sum(len(doc.get("text") or "") for doc in processed_docs)
    logger.info(
        "Loaded %d processed docs (%d total characters)",
        len(processed_docs),
        total_chars,
    )

    chunks = chunk_documents(processed_docs)
    logger.info("Produced %d chunks", len(chunks))

    if not chunks:
        logger.error("No chunks produced; aborting before embedding.")
        sys.exit(1)

    chunk_chars = sum(len(chunk.page_content) for chunk in chunks)
    est_tokens = chunk_chars // _CHARS_PER_TOKEN
    est_cost_usd = (est_tokens / 1_000_000) * _USD_PER_MILLION_TOKENS
    logger.info(
        "Estimated embedding cost: ~%d tokens (~$%.4f USD) using %s",
        est_tokens,
        est_cost_usd,
        settings.embedding_model,
    )

    embedder = get_embedder()
    store = get_store(embedder)
    added = upsert_documents(store, chunks)

    logger.info(
        "Done: %s",
        {
            "added": added,
            "collection": settings.chroma_collection_name,
            "persist_dir": str(settings.vector_store_dir),
        },
    )


if __name__ == "__main__":
    main()
