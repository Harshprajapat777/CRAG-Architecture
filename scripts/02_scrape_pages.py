"""Scrape every URL listed in the consolidated sitemap output.

Prefers the consolidated XML (so `lastmod` is preserved); falls back to the
plain URL list if the XML is missing. Writes raw HTML + processed JSON
under `data/raw` and `data/processed`.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running this script directly: ensure project root is importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings  # noqa: E402
from src.ingestion.pipeline import run_scrape  # noqa: E402
from src.ingestion.sitemap_parser import UrlEntry, read_consolidated_xml  # noqa: E402
from src.utils.io import ensure_dirs  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def _load_entries() -> list[UrlEntry]:
    xml_path = settings.sitemaps_dir / "evangelist_urls.xml"
    txt_path = settings.sitemaps_dir / "evangelist_urls.txt"

    if xml_path.exists():
        logger.info("Loading URLs from consolidated XML: %s", xml_path)
        return read_consolidated_xml(xml_path)

    if txt_path.exists():
        logger.info("Consolidated XML missing; loading URL list: %s", txt_path)
        urls = [line.strip() for line in txt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [UrlEntry(loc=url, lastmod=None, source_sitemap="") for url in urls]

    raise FileNotFoundError(
        f"No URL source found. Run scripts/01_extract_sitemap.py first. "
        f"Looked for {xml_path} and {txt_path}."
    )


def main() -> None:
    ensure_dirs(settings.raw_dir, settings.processed_dir)
    entries = _load_entries()
    logger.info("Starting scrape over %d URLs", len(entries))

    report = run_scrape(entries, settings.raw_dir, settings.processed_dir)

    logger.info(
        "Report: total=%d succeeded=%d skipped=%d failed=%d",
        report.total,
        report.succeeded,
        report.skipped,
        len(report.failed),
    )


if __name__ == "__main__":
    main()
