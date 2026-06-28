"""KB coverage eval: end-to-end check that every sitemap URL is represented in ChromaDB.

Three independent stages are measured:

  Stage 1  Sitemap   -> Scrape          (data/sitemaps/evangelist_urls.txt vs data/processed/*.json)
  Stage 2  Scrape    -> Vector store    (data/processed/*.json    vs Chroma collection)
  Stage 3  Retrieval -> Vector store    (sentence from each doc returns the doc on similarity_search)

Stages 1 and 2 are pure filesystem / metadata inspection (free, instant).
Stage 3 is opt-in (`--retrieve`) and uses OpenAI embeddings to validate the query path.

Run directly for a human-readable report:
    python tests/kb_coverage.py
    python tests/kb_coverage.py --retrieve --sample 10

Or via pytest for threshold assertions:
    pytest tests/test_kb_coverage.py
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from src.utils.io import read_json, url_to_slug
from src.utils.logger import get_logger

logger = get_logger(__name__)

REQUIRED_CHUNK_METADATA_KEYS = {"source", "title", "lastmod", "source_slug", "chunk_index"}


@dataclass
class CoverageReport:
    sitemap_urls: list[str]
    sitemap_slugs: set[str]
    processed_docs: dict[str, dict]
    processed_slugs: set[str]
    failed_urls: list[dict]
    kb_ids: list[str]
    kb_metadatas: list[dict]
    kb_documents: list[str]
    kb_slugs: set[str]
    chunks_per_slug: dict[str, int]
    chars_source_per_slug: dict[str, int]
    chars_chunks_per_slug: dict[str, int]
    metadata_keys_per_chunk: list[set[str]]
    retrievability: dict[str, bool] = field(default_factory=dict)
    retrieval_sample_size: int = 0

    @property
    def missing_from_processed(self) -> set[str]:
        return self.sitemap_slugs - self.processed_slugs

    @property
    def orphan_processed(self) -> set[str]:
        return self.processed_slugs - self.sitemap_slugs

    @property
    def missing_from_kb(self) -> set[str]:
        return self.processed_slugs - self.kb_slugs

    @property
    def orphan_kb(self) -> set[str]:
        return self.kb_slugs - self.processed_slugs

    @property
    def sitemap_coverage_pct(self) -> float:
        if not self.sitemap_slugs:
            return 0.0
        return 100 * len(self.sitemap_slugs & self.processed_slugs) / len(self.sitemap_slugs)

    @property
    def kb_coverage_pct(self) -> float:
        if not self.processed_slugs:
            return 0.0
        return 100 * len(self.processed_slugs & self.kb_slugs) / len(self.processed_slugs)

    @property
    def retrievability_pct(self) -> float:
        if not self.retrievability:
            return 0.0
        return 100 * sum(self.retrievability.values()) / len(self.retrievability)

    @property
    def total_chunks(self) -> int:
        return len(self.kb_ids)

    @property
    def docs_with_zero_chunks(self) -> set[str]:
        return {s for s in self.processed_slugs if self.chunks_per_slug.get(s, 0) == 0}

    @property
    def total_source_chars(self) -> int:
        return sum(self.chars_source_per_slug.values())

    @property
    def total_chunk_chars(self) -> int:
        return sum(self.chars_chunks_per_slug.values())

    @property
    def char_ratio(self) -> float:
        if self.total_source_chars == 0:
            return 0.0
        return self.total_chunk_chars / self.total_source_chars

    @property
    def missing_metadata_chunks(self) -> int:
        return sum(1 for keys in self.metadata_keys_per_chunk if not REQUIRED_CHUNK_METADATA_KEYS.issubset(keys))


def _load_sitemap_urls() -> list[str]:
    txt = settings.sitemaps_dir / "evangelist_urls.txt"
    if not txt.exists():
        raise FileNotFoundError(
            f"URL list not found at {txt}. Run scripts/01_extract_sitemap.py first."
        )
    return [line.strip() for line in txt.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_processed_docs() -> tuple[dict[str, dict], list[dict]]:
    if not settings.processed_dir.exists():
        raise FileNotFoundError(
            f"Processed dir not found at {settings.processed_dir}. Run scripts/02_scrape_pages.py first."
        )

    docs: dict[str, dict] = {}
    for p in sorted(settings.processed_dir.glob("*.json")):
        if p.name == "_failures.json":
            continue
        doc = read_json(p)
        slug = doc.get("slug") or url_to_slug(doc.get("url", ""))
        docs[slug] = doc

    failures_path = settings.processed_dir / "_failures.json"
    failures = read_json(failures_path) if failures_path.exists() else []
    return docs, failures


def _load_kb_chunks() -> dict[str, Any]:
    """Read the entire Chroma collection via chromadb directly (no embedder needed)."""
    import chromadb

    if not settings.vector_store_dir.exists():
        raise FileNotFoundError(
            f"Vector store dir not found at {settings.vector_store_dir}. Run scripts/03_build_vector_store.py first."
        )

    client = chromadb.PersistentClient(path=str(settings.vector_store_dir))
    collection = client.get_collection(settings.chroma_collection_name)
    return collection.get(include=["metadatas", "documents"])


def _probe_retrievability(report: CoverageReport, sample_size: int, k: int = 5) -> None:
    """For a random sample of indexed docs, query with the doc's own first chunk text
    (which is guaranteed to exist in the KB) and confirm the doc's slug appears in
    the top-k similarity_search results. A miss here means the chunk -> doc mapping
    is broken, not that the embeddings are bad."""
    from src.vector_store.chroma_store import get_store, similarity_search

    store = get_store()
    indexed_slugs = sorted(report.processed_slugs & report.kb_slugs)
    if not indexed_slugs:
        logger.warning("No indexed slugs available for retrievability probe.")
        return

    # Index a slug -> [(chunk_index, chunk_text), ...] map from kb data so we can
    # query each sampled doc with one of its own indexed chunks.
    chunks_by_slug: dict[str, list[tuple[int, str]]] = {}
    for meta, text in zip(report.kb_metadatas, report.kb_documents):
        slug = meta.get("source_slug", "")
        idx = int(meta.get("chunk_index", 0))
        chunks_by_slug.setdefault(slug, []).append((idx, text or ""))

    sample = random.sample(indexed_slugs, min(sample_size, len(indexed_slugs)))
    report.retrieval_sample_size = len(sample)

    for slug in sample:
        chunks = sorted(chunks_by_slug.get(slug, []), key=lambda kv: kv[0])
        if not chunks:
            report.retrievability[slug] = False
            continue
        # Query with the first indexed chunk's text.
        query = chunks[0][1]
        if len(query) < 50:
            report.retrievability[slug] = False
            continue
        try:
            results = similarity_search(store, query, k=k)
            hit_slugs = {r.metadata.get("source_slug", "") for r in results}
            report.retrievability[slug] = slug in hit_slugs
        except Exception as exc:  # pragma: no cover - network/API issues
            logger.error("Retrieval probe failed for %s: %s", slug, exc)
            report.retrievability[slug] = False


def compute_coverage(run_retrieval: bool = False, sample_size: int = 10) -> CoverageReport:
    sitemap_urls = _load_sitemap_urls()
    sitemap_slugs = {url_to_slug(u) for u in sitemap_urls}

    processed_docs, failures = _load_processed_docs()
    processed_slugs = set(processed_docs.keys())

    kb = _load_kb_chunks()
    kb_ids: list[str] = kb["ids"]
    kb_metadatas: list[dict] = [m or {} for m in kb["metadatas"]]
    kb_documents: list[str] = kb["documents"]

    kb_slugs: set[str] = set()
    chunks_per_slug: dict[str, int] = {}
    chars_chunks_per_slug: dict[str, int] = {}
    metadata_keys_per_chunk: list[set[str]] = []

    for meta, doc_text in zip(kb_metadatas, kb_documents):
        slug = meta.get("source_slug", "")
        kb_slugs.add(slug)
        chunks_per_slug[slug] = chunks_per_slug.get(slug, 0) + 1
        chars_chunks_per_slug[slug] = chars_chunks_per_slug.get(slug, 0) + len(doc_text or "")
        metadata_keys_per_chunk.append(set(meta.keys()))

    chars_source_per_slug = {
        slug: int(doc.get("char_count") or len(doc.get("text") or ""))
        for slug, doc in processed_docs.items()
    }

    report = CoverageReport(
        sitemap_urls=sitemap_urls,
        sitemap_slugs=sitemap_slugs,
        processed_docs=processed_docs,
        processed_slugs=processed_slugs,
        failed_urls=failures,
        kb_ids=kb_ids,
        kb_metadatas=kb_metadatas,
        kb_documents=kb_documents,
        kb_slugs=kb_slugs,
        chunks_per_slug=chunks_per_slug,
        chars_source_per_slug=chars_source_per_slug,
        chars_chunks_per_slug=chars_chunks_per_slug,
        metadata_keys_per_chunk=metadata_keys_per_chunk,
    )

    if run_retrieval:
        _probe_retrievability(report, sample_size=sample_size)

    return report


def render_report(report: CoverageReport) -> None:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()

    # Stage 1 -- Sitemap -> Scrape
    t1 = Table(show_header=False, box=None, pad_edge=False)
    t1.add_column(style="bold")
    t1.add_column(justify="right")
    t1.add_row("Sitemap URLs", str(len(report.sitemap_urls)))
    t1.add_row("Unique sitemap slugs", str(len(report.sitemap_slugs)))
    t1.add_row("Processed JSON docs", str(len(report.processed_slugs)))
    t1.add_row("Coverage", f"{report.sitemap_coverage_pct:.2f}%")
    t1.add_row("Missing (sitemap, no JSON)", str(len(report.missing_from_processed)))
    t1.add_row("Orphan (JSON, no sitemap entry)", str(len(report.orphan_processed)))
    t1.add_row("Scrape failures (_failures.json)", str(len(report.failed_urls)))
    console.print(Panel(t1, title="Stage 1: Sitemap -> Scrape", border_style="cyan"))

    # Stage 2 -- Scrape -> KB
    t2 = Table(show_header=False, box=None, pad_edge=False)
    t2.add_column(style="bold")
    t2.add_column(justify="right")
    t2.add_row("Processed docs", str(len(report.processed_slugs)))
    t2.add_row("Distinct docs in KB", str(len(report.kb_slugs)))
    t2.add_row("Coverage", f"{report.kb_coverage_pct:.2f}%")
    t2.add_row("Missing (processed, not in KB)", str(len(report.missing_from_kb)))
    t2.add_row("Orphan (in KB, no source doc)", str(len(report.orphan_kb)))
    t2.add_row("Total chunks indexed", str(report.total_chunks))
    t2.add_row("Docs with zero chunks", str(len(report.docs_with_zero_chunks)))
    console.print(Panel(t2, title="Stage 2: Scrape -> Vector Store", border_style="cyan"))

    # Chunk distribution
    counts = list(report.chunks_per_slug.values())
    if counts:
        t3 = Table(show_header=False, box=None, pad_edge=False)
        t3.add_column(style="bold")
        t3.add_column(justify="right")
        t3.add_row("Min chunks per doc", str(min(counts)))
        t3.add_row("Mean", f"{statistics.mean(counts):.2f}")
        t3.add_row("Median", f"{statistics.median(counts):.1f}")
        t3.add_row("Max", str(max(counts)))
        t3.add_row("Std dev", f"{statistics.pstdev(counts):.2f}" if len(counts) > 1 else "n/a")
        console.print(Panel(t3, title="Chunk distribution per doc", border_style="magenta"))

        top5 = sorted(report.chunks_per_slug.items(), key=lambda kv: -kv[1])[:5]
        if top5:
            tt = Table(box=None, pad_edge=False)
            tt.add_column("slug", style="bold")
            tt.add_column("chunks", justify="right")
            for slug, c in top5:
                tt.add_row(slug, str(c))
            console.print(Panel(tt, title="Top 5 docs by chunk count", border_style="magenta"))

    # Content integrity
    t4 = Table(show_header=False, box=None, pad_edge=False)
    t4.add_column(style="bold")
    t4.add_column(justify="right")
    t4.add_row("Total source chars", f"{report.total_source_chars:,}")
    t4.add_row("Total chunk chars", f"{report.total_chunk_chars:,}")
    t4.add_row("Chunk/source ratio", f"{report.char_ratio:.3f}")
    t4.add_row("Chunks missing required metadata", str(report.missing_metadata_chunks))
    console.print(Panel(t4, title="Content integrity", border_style="green"))

    # Stage 3 -- Retrievability (only if probed)
    if report.retrieval_sample_size:
        t5 = Table(show_header=False, box=None, pad_edge=False)
        t5.add_column(style="bold")
        t5.add_column(justify="right")
        hits = sum(report.retrievability.values())
        t5.add_row("Sampled docs", str(report.retrieval_sample_size))
        t5.add_row("Retrieved in top-5", f"{hits}/{report.retrieval_sample_size}")
        t5.add_row("Retrievability", f"{report.retrievability_pct:.2f}%")
        misses = [s for s, ok in report.retrievability.items() if not ok]
        if misses:
            t5.add_row("Misses", ", ".join(misses[:5]) + ("..." if len(misses) > 5 else ""))
        console.print(Panel(t5, title="Stage 3: Retrievability spot-check", border_style="yellow"))
    else:
        console.print(
            Panel(
                "Skipped. Re-run with --retrieve to probe live similarity_search (uses OpenAI).",
                title="Stage 3: Retrievability spot-check",
                border_style="yellow",
            )
        )

    # Overall verdict
    verdict_ok = (
        report.sitemap_coverage_pct >= 95.0
        and report.kb_coverage_pct >= 99.0
        and report.missing_metadata_chunks == 0
        and not report.docs_with_zero_chunks
    )
    style = "bold green" if verdict_ok else "bold red"
    summary = (
        f"Sitemap->Scrape {report.sitemap_coverage_pct:.2f}%  |  "
        f"Scrape->KB {report.kb_coverage_pct:.2f}%  |  "
        f"chunks={report.total_chunks}  |  "
        f"missing-metadata={report.missing_metadata_chunks}"
    )
    console.print(Panel(summary, title="VERDICT: OK" if verdict_ok else "VERDICT: ISSUES", border_style=style))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CRAG-BOT KB coverage report")
    parser.add_argument("--retrieve", action="store_true", help="Run retrievability probe (uses OpenAI).")
    parser.add_argument("--sample", type=int, default=10, help="Retrievability sample size (default 10).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the retrievability sample.")
    args = parser.parse_args(argv)

    random.seed(args.seed)
    report = compute_coverage(run_retrieval=args.retrieve, sample_size=args.sample)
    render_report(report)

    issues = (
        report.sitemap_coverage_pct < 95.0
        or report.kb_coverage_pct < 99.0
        or report.missing_metadata_chunks > 0
        or bool(report.docs_with_zero_chunks)
    )
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
