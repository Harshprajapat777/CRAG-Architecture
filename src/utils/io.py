import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def url_to_slug(url: str) -> str:
    """Stable filesystem-safe slug for a URL. Example:
    https://evangelistsoftware.com/services/ai-ml -> services_ai-ml
    Homepage -> 'index'.
    """
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if not path:
        return "index"
    slug = re.sub(r"[^a-zA-Z0-9._/-]", "_", path).replace("/", "_")
    return slug[:180] or "index"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
