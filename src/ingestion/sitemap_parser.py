"""Sitemap index parsing and URL extraction.

Downloads the top-level sitemap index, walks each sub-sitemap, and produces
a deduplicated list of `UrlEntry` records. Helpers are provided to persist
the consolidated set as both XML and a plain URL list.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

import requests

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

SITEMAP_NS = "http://www.sitemaps.org/schemas/sitemap/0.9"
_NS = {"sm": SITEMAP_NS}


@dataclass
class UrlEntry:
    """A single URL discovered in a sitemap."""

    loc: str
    lastmod: Optional[str]
    source_sitemap: str


def _http_get(url: str) -> bytes:
    """Fetch raw bytes for `url` using the project's UA + timeout."""
    headers = {"User-Agent": settings.scrape_user_agent}
    response = requests.get(url, headers=headers, timeout=settings.scrape_timeout_sec)
    response.raise_for_status()
    return response.content


def _parse_sitemap_index(xml_bytes: bytes) -> list[str]:
    """Return the list of sub-sitemap URLs from a sitemap index document."""
    root = ET.fromstring(xml_bytes)
    locs: list[str] = []
    for sitemap_el in root.findall("sm:sitemap", _NS):
        loc_el = sitemap_el.find("sm:loc", _NS)
        if loc_el is not None and loc_el.text:
            locs.append(loc_el.text.strip())
    return locs


def _parse_urlset(xml_bytes: bytes, source_sitemap: str) -> list[UrlEntry]:
    """Return URL entries from a `<urlset>` sitemap document."""
    root = ET.fromstring(xml_bytes)
    entries: list[UrlEntry] = []
    for url_el in root.findall("sm:url", _NS):
        loc_el = url_el.find("sm:loc", _NS)
        if loc_el is None or not loc_el.text:
            continue
        lastmod_el = url_el.find("sm:lastmod", _NS)
        lastmod = lastmod_el.text.strip() if (lastmod_el is not None and lastmod_el.text) else None
        entries.append(
            UrlEntry(
                loc=loc_el.text.strip(),
                lastmod=lastmod,
                source_sitemap=source_sitemap,
            )
        )
    return entries


def fetch_all_urls(index_url: str) -> list[UrlEntry]:
    """Download a sitemap index and return all deduplicated URL entries.

    Deduplication is by `loc`; first occurrence wins.
    """
    logger.info("Fetching sitemap index: %s", index_url)
    index_bytes = _http_get(index_url)
    sub_sitemaps = _parse_sitemap_index(index_bytes)
    logger.info("Found %d sub-sitemaps in index", len(sub_sitemaps))

    seen: set[str] = set()
    all_entries: list[UrlEntry] = []
    for sub in sub_sitemaps:
        logger.info("Fetching sub-sitemap: %s", sub)
        sub_bytes = _http_get(sub)
        entries = _parse_urlset(sub_bytes, source_sitemap=sub)
        new_count = 0
        for entry in entries:
            if entry.loc in seen:
                continue
            seen.add(entry.loc)
            all_entries.append(entry)
            new_count += 1
        logger.info("  -> %d urls (%d new) from %s", len(entries), new_count, sub)

    logger.info("Total unique URLs collected: %d", len(all_entries))
    return all_entries


def write_consolidated_xml(entries: list[UrlEntry], path: Path) -> None:
    """Write all entries as a single `<urlset>` XML document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.register_namespace("", SITEMAP_NS)
    urlset = ET.Element(f"{{{SITEMAP_NS}}}urlset")
    for entry in entries:
        url_el = ET.SubElement(urlset, f"{{{SITEMAP_NS}}}url")
        loc_el = ET.SubElement(url_el, f"{{{SITEMAP_NS}}}loc")
        loc_el.text = entry.loc
        if entry.lastmod:
            lastmod_el = ET.SubElement(url_el, f"{{{SITEMAP_NS}}}lastmod")
            lastmod_el.text = entry.lastmod
        source_el = ET.SubElement(url_el, f"{{{SITEMAP_NS}}}source-sitemap")
        source_el.text = entry.source_sitemap

    tree = ET.ElementTree(urlset)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    logger.info("Wrote consolidated XML: %s", path)


def write_url_list(entries: list[UrlEntry], path: Path) -> None:
    """Write one URL per line to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(entry.loc for entry in entries) + ("\n" if entries else "")
    path.write_text(content, encoding="utf-8")
    logger.info("Wrote URL list: %s", path)


def read_consolidated_xml(path: Path) -> list[UrlEntry]:
    """Read a consolidated `<urlset>` XML produced by `write_consolidated_xml`."""
    xml_bytes = path.read_bytes()
    root = ET.fromstring(xml_bytes)
    entries: list[UrlEntry] = []
    for url_el in root.findall("sm:url", _NS):
        loc_el = url_el.find("sm:loc", _NS)
        if loc_el is None or not loc_el.text:
            continue
        lastmod_el = url_el.find("sm:lastmod", _NS)
        lastmod = lastmod_el.text.strip() if (lastmod_el is not None and lastmod_el.text) else None
        source_el = url_el.find("sm:source-sitemap", _NS)
        source = source_el.text.strip() if (source_el is not None and source_el.text) else ""
        entries.append(UrlEntry(loc=loc_el.text.strip(), lastmod=lastmod, source_sitemap=source))
    return entries
