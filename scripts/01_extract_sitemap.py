"""Extract all URLs from the configured sitemap index.

Downloads the sitemap index, walks every sub-sitemap, and writes the
deduplicated URL set as both XML (with `lastmod` + source) and a plain
text list.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running this script directly: ensure project root is importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings  # noqa: E402
from src.ingestion.sitemap_parser import (  # noqa: E402
    fetch_all_urls,
    write_consolidated_xml,
    write_url_list,
)
from src.utils.io import ensure_dirs  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    ensure_dirs(settings.sitemaps_dir)

    entries = fetch_all_urls(settings.sitemap_index_url)

    xml_path = settings.sitemaps_dir / "evangelist_urls.xml"
    txt_path = settings.sitemaps_dir / "evangelist_urls.txt"
    write_consolidated_xml(entries, xml_path)
    write_url_list(entries, txt_path)

    logger.info("Extracted %d unique URLs", len(entries))
    logger.info("  XML -> %s", xml_path)
    logger.info("  TXT -> %s", txt_path)


if __name__ == "__main__":
    main()
