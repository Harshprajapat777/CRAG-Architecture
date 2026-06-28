"""HTTP fetching and HTML -> text extraction.

`fetch_html` retries with exponential backoff via tenacity.
`extract_text` prefers trafilatura, with a BeautifulSoup fallback for thin
or missed extractions, and always returns the page title separately.
"""
from __future__ import annotations

import requests
import trafilatura
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Minimum length (chars) before we accept the trafilatura output instead of
# falling back to BeautifulSoup.
_MIN_TRAFILATURA_CHARS = 100

# Tags we strip wholesale when we have to fall back to BeautifulSoup.
_FALLBACK_STRIP_TAGS = ("script", "style", "nav", "footer", "aside")


@retry(
    reraise=True,
    stop=stop_after_attempt(settings.scrape_max_retries),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(requests.RequestException),
)
def fetch_html(url: str) -> str:
    """Fetch `url` and return decoded HTML. Retries transient HTTP errors."""
    headers = {"User-Agent": settings.scrape_user_agent}
    response = requests.get(url, headers=headers, timeout=settings.scrape_timeout_sec)
    response.raise_for_status()
    # `response.text` already decodes via the charset inferred by requests.
    return response.text


def _extract_title(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    return ""


def _bs_fallback_text(soup: BeautifulSoup) -> str:
    """Heuristic body-text extraction when trafilatura returns nothing useful."""
    # Operate on a copy so we don't mutate the caller's soup (used for title).
    working = BeautifulSoup(str(soup), "lxml")
    for tag_name in _FALLBACK_STRIP_TAGS:
        for tag in working.find_all(tag_name):
            tag.decompose()

    container = (
        working.find("main")
        or working.find("article")
        or working.find("div", id="content")
        or working.body
    )
    if container is None:
        return ""
    return container.get_text(separator="\n", strip=True)


def extract_text(html: str, url: str) -> dict:
    """Extract a page's main text and title.

    Returns a dict with `title`, `text`, and `char_count` keys.
    """
    soup = BeautifulSoup(html, "lxml")
    title = _extract_title(soup)

    extracted = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        favor_recall=True,
    )

    if not extracted or len(extracted) < _MIN_TRAFILATURA_CHARS:
        logger.debug(
            "trafilatura output thin (%s chars) for %s; using BS4 fallback",
            len(extracted) if extracted else 0,
            url,
        )
        text = _bs_fallback_text(soup)
    else:
        text = extracted

    text = text.strip()
    return {"title": title, "text": text, "char_count": len(text)}
