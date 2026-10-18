"""News coverage from the GDELT DOC 2.0 API (free, no key).

For a city it pulls three things in one go: how much of all monitored coverage matches (volume), the
average tone of that coverage (negative numbers are gloomier), and the newest headlines.

    https://api.gdeltproject.org/api/v2/doc/doc?query=...&mode=timelinevol|timelinetone|artlist&format=json

IMPORTANT: written from memory of the documented API and NOT run against the live service (the build
environment cannot reach it). GDELT's DOC API only searches roughly the last three months, so this signal
cannot be replayed for past dates; see workers/news.py. Assumed JSON shapes:
  timeline: {"timeline": [{"series": "...", "data": [{"date": "20260901T000000Z", "value": 1.2}, ...]}]}
  artlist:  {"articles": [{"url", "title", "seendate", "domain", "language", "sourcecountry"}, ...]}
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
from datetime import date
from typing import Callable

from retrend.location import Location
from retrend.sources.files import SourceError

BASE = "https://api.gdeltproject.org/api/v2/doc/doc"
TOPICS = '(housing OR "home prices" OR "real estate" OR rent OR rents OR mortgage OR "home sales")'
TIMESPAN = "3months"
MAX_HEADLINES = 8
PAUSE = 6.0  # GDELT asks for about one request per five seconds; a little more is safer
RETRIES = 3
BACKOFF = 15.0  # seconds added per retry after a 'too many requests' reply


class NewsError(SourceError):
    pass


def build_query(loc: Location) -> str:
    return f'"{loc.city}" {TOPICS} sourcecountry:US sourcelang:english'


def url_for(mode: str, loc: Location, **extra) -> str:
    params = {"query": build_query(loc), "mode": mode, "format": "json", "timespan": TIMESPAN, **extra}
    return BASE + "?" + urllib.parse.urlencode(params)


_last_request = 0.0


def _wait_turn(pause: float, sleep: Callable[[float], None]) -> None:
    """Keep at least ``pause`` seconds between ANY two GDELT requests in this process (several cities too)."""
    global _last_request
    gap = _last_request + pause - time.monotonic()
    if gap > 0:
        sleep(gap)
    _last_request = time.monotonic()


def _get_json(open_url: Callable, url: str, *, pause: float = 0.0, sleep: Callable[[float], None] = time.sleep,
              retries: int = RETRIES) -> dict:
    """GET and parse. A 'too many requests' reply (HTTP 429, or GDELT's plain-text 'please limit requests')
    waits and tries again, longer each time, up to ``retries`` times."""
    for attempt in range(retries + 1):
        _wait_turn(pause, sleep)
        wait, problem = 0.0, None
        try:
            raw = open_url(url).read()
        except urllib.error.HTTPError as exc:
            if exc.code != 429:
                raise NewsError(f"GDELT request failed: HTTP {exc.code}") from None
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            wait = float(retry_after) if retry_after and retry_after.isdigit() else BACKOFF * (attempt + 1)
            problem = "GDELT says too many requests (HTTP 429)"
        except OSError as exc:
            raise NewsError(f"GDELT request failed: {type(exc).__name__}: {exc}") from None
        else:
            text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
            try:
                return json.loads(text)
            except ValueError:
                snippet = " ".join(text.split())[:160]
                if "limit" not in snippet.lower() and "too many" not in snippet.lower():
                    # other plain-text replies are bad queries, not rate limits
                    raise NewsError("GDELT did not return JSON: " + snippet) from None
                wait, problem = BACKOFF * (attempt + 1), "GDELT says too many requests: " + snippet
        if attempt == retries:
            raise NewsError(f"{problem}. Tried {retries + 1} times; wait a few minutes and ask again.")
        sleep(wait)
    raise AssertionError("unreachable")


def _points(data: dict) -> list[dict]:
    try:
        series = data["timeline"][0]["data"]
        out = []
        for p in series:
            d = re.match(r"(\d{4})(\d{2})(\d{2})", str(p["date"]))
            if d and p.get("value") is not None:
                out.append({"date": date(int(d[1]), int(d[2]), int(d[3])).isoformat(), "value": float(p["value"])})
        return sorted(out, key=lambda r: r["date"])
    except (KeyError, IndexError, TypeError, ValueError):
        raise NewsError("GDELT returned a timeline in an unexpected shape") from None


def fetch_news(loc: Location, open_url: Callable, *, pause: float = PAUSE,
               sleep: Callable[[float], None] = time.sleep, retries: int = RETRIES) -> dict:
    """-> {"volume": [...], "tone": [...], "headlines": [...]}; an empty timeline means no matching coverage."""
    out: dict = {}
    for i, (key, mode, extra) in enumerate((("volume", "timelinevol", {}), ("tone", "timelinetone", {}),
                                            ("articles", "artlist", {"maxrecords": 50, "sort": "datedesc"}))):
        data = _get_json(open_url, url_for(mode, loc, **extra), pause=pause, sleep=sleep, retries=retries)
        if key == "articles":
            try:
                arts = data.get("articles") or []
                out["headlines"] = [
                    {"title": " ".join(str(a.get("title", "")).split())[:200], "url": str(a.get("url", "")),
                     "domain": str(a.get("domain", ""))[:80], "seen": str(a.get("seendate", ""))[:8]}
                    for a in arts if a.get("title") and str(a.get("url", "")).startswith(("http://", "https://"))
                ][:MAX_HEADLINES]
            except AttributeError:
                raise NewsError("GDELT returned articles in an unexpected shape") from None
        else:
            out[key] = _points(data) if data else []
    return out
