"""Threshold assertions on the KB coverage report.

Run:
    pytest tests/test_kb_coverage.py -v

These tests inspect the existing artifacts in `data/` and the persistent
ChromaDB collection. They do NOT re-run the ingestion pipeline.
"""
from __future__ import annotations

import pytest

from tests.kb_coverage import CoverageReport


def test_no_scrape_failures(coverage_report: CoverageReport) -> None:
    failed = coverage_report.failed_urls
    assert not failed, f"{len(failed)} scrape failures recorded in _failures.json: {[f.get('url') for f in failed[:5]]}"


def test_sitemap_coverage_at_least_95_percent(coverage_report: CoverageReport) -> None:
    pct = coverage_report.sitemap_coverage_pct
    missing = sorted(coverage_report.missing_from_processed)
    assert pct >= 95.0, (
        f"Only {pct:.2f}% of sitemap URLs have processed JSONs. "
        f"Missing slugs ({len(missing)}): {missing[:10]}"
    )


def test_every_processed_doc_indexed_in_kb(coverage_report: CoverageReport) -> None:
    missing = sorted(coverage_report.missing_from_kb)
    assert not missing, (
        f"{len(missing)} processed docs are absent from the vector store. "
        f"First few: {missing[:10]}"
    )


def test_no_orphan_chunks_in_kb(coverage_report: CoverageReport) -> None:
    orphans = sorted(coverage_report.orphan_kb)
    assert not orphans, (
        f"{len(orphans)} KB slugs have no corresponding processed JSON: {orphans[:10]}"
    )


def test_every_doc_has_at_least_one_chunk(coverage_report: CoverageReport) -> None:
    zero = sorted(coverage_report.docs_with_zero_chunks)
    assert not zero, f"{len(zero)} processed docs produced zero chunks: {zero[:10]}"


def test_total_chunks_positive(coverage_report: CoverageReport) -> None:
    assert coverage_report.total_chunks > 0, "Chroma collection is empty."


def test_chunk_metadata_complete(coverage_report: CoverageReport) -> None:
    missing = coverage_report.missing_metadata_chunks
    assert missing == 0, f"{missing} chunks are missing one or more required metadata keys."


def test_chunk_char_ratio_within_expected_bounds(coverage_report: CoverageReport) -> None:
    # With chunk_size=1000, overlap=150 we expect chunk chars to be roughly
    # 1.0x-1.3x source chars (overlap inflates total). Generous bounds catch
    # truncation bugs (ratio << 1) and runaway duplication (ratio >> 1.5).
    ratio = coverage_report.char_ratio
    assert 0.85 <= ratio <= 1.5, (
        f"Chunk/source char ratio {ratio:.3f} is outside expected bounds [0.85, 1.5]. "
        f"source={coverage_report.total_source_chars:,} chunks={coverage_report.total_chunk_chars:,}"
    )


@pytest.mark.slow
def test_retrievability_at_least_80_percent() -> None:
    """Opt-in: probes live OpenAI similarity_search. Run with `pytest -m slow`."""
    from tests.kb_coverage import compute_coverage

    report = compute_coverage(run_retrieval=True, sample_size=10)
    assert report.retrieval_sample_size > 0, "No docs available to probe."
    pct = report.retrievability_pct
    misses = sorted(s for s, ok in report.retrievability.items() if not ok)
    assert pct >= 80.0, (
        f"Retrievability {pct:.2f}% below 80% threshold. Misses: {misses[:5]}"
    )
