"""End-to-end scraping pipeline.

Iterates URL entries, persists raw HTML + processed JSON, and produces a
`ScrapeReport` summarizing the run. Resumable: existing processed files are
skipped so re-runs are cheap.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from config.settings import settings
from src.ingestion.scraper import extract_text, fetch_html
from src.ingestion.sitemap_parser import UrlEntry
from src.utils.io import ensure_dirs, url_to_slug, write_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ScrapeReport:
    total: int = 0
    succeeded: int = 0
    skipped: int = 0
    failed: list[dict] = field(default_factory=list)


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_scrape(
    entries: list[UrlEntry],
    raw_dir: Path,
    processed_dir: Path,
) -> ScrapeReport:
    """Scrape each URL entry, persisting raw HTML and processed JSON.

    Skips URLs whose processed JSON already exists. Failures are aggregated
    into the returned `ScrapeReport` and also written to
    `processed_dir / "_failures.json"`.
    """
    ensure_dirs(raw_dir, processed_dir)

    report = ScrapeReport(total=len(entries))
    delay = settings.scrape_request_delay_sec

    progress = tqdm(entries, desc="scraping", unit="page")
    for entry in progress:
        slug = url_to_slug(entry.loc)
        processed_path = processed_dir / f"{slug}.json"
        raw_path = raw_dir / f"{slug}.html"

        if processed_path.exists():
            report.skipped += 1
            continue

        try:
            html = fetch_html(entry.loc)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(html, encoding="utf-8")

            extracted = extract_text(html, entry.loc)
            record = {
                "url": entry.loc,
                "slug": slug,
                "title": extracted["title"],
                "lastmod": entry.lastmod,
                "text": extracted["text"],
                "char_count": extracted["char_count"],
                "scraped_at": _iso_utc_now(),
            }
            write_json(processed_path, record)
            report.succeeded += 1
        except Exception as exc:  # noqa: BLE001 - aggregated, not swallowed
            logger.warning("Failed to scrape %s: %s", entry.loc, exc)
            report.failed.append(
                {
                    "url": entry.loc,
                    "slug": slug,
                    "error": f"{type(exc).__name__}: {exc}",
                    "at": _iso_utc_now(),
                }
            )

        if delay > 0:
            time.sleep(delay)

    failures_path = processed_dir / "_failures.json"
    write_json(failures_path, report.failed)
    logger.info(
        "Scrape complete: total=%d succeeded=%d skipped=%d failed=%d",
        report.total,
        report.succeeded,
        report.skipped,
        len(report.failed),
    )
    return report
