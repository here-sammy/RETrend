"""Downloading big public data files once and reusing them.

Zillow and Redfin publish whole-country files. We download each file to disk (atomically,
with a refresh interval), then stream through it to pull out one region, so a run for
three cities downloads each file once.
"""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable


class SourceError(RuntimeError):
    """A data file could not be downloaded, read, or matched to a location."""


def _default_open(url: str) -> Any:
    request = urllib.request.Request(
        url, headers={"User-Agent": "retrend/0.0.2 (research project)"}
    )
    return urllib.request.urlopen(request, timeout=120)


def download_cached(
    url: str,
    dest_dir: str | Path,
    *,
    max_age: timedelta,
    now: Callable[[], datetime] | None = None,
    open_url: Callable[[str], Any] | None = None,
    hint: str = "",
) -> Path:
    """Return a local copy of ``url``, downloading it if absent or older than ``max_age``.

    The download goes to a temporary file and is renamed into place, so an interrupted
    download never leaves a truncated file behind.
    """
    now = now or (lambda: datetime.now(timezone.utc))
    open_url = open_url or _default_open
    directory = Path(dest_dir)
    directory.mkdir(parents=True, exist_ok=True)
    name = urllib.parse.urlparse(url).path.rsplit("/", 1)[-1]
    path = directory / (re.sub(r"[^A-Za-z0-9._-]+", "_", name) or "download")

    if path.exists():
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if now() - modified <= max_age:
            return path

    partial = path.with_name(path.name + ".part")
    try:
        with open_url(url) as response, open(partial, "wb") as out:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        partial.replace(path)
    except Exception as exc:
        partial.unlink(missing_ok=True)
        raise SourceError(
            f"could not download {url}: {type(exc).__name__}: {exc}. {hint}".strip()
        ) from None
    return path
