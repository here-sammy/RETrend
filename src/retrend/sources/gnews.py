"""Housing headlines from Google News RSS (no key), scored for tone in code.

    https://news.google.com/rss/search?q="Seattle" (housing OR ...) after:2026-09-01 before:2026-10-01&hl=en-US&gl=US&ceid=US:en

Three 30-day windows are fetched (recent, and the two before) and turned into the same daily tone and
volume series the GDELT source produces, so the news worker treats both alike.

Tone is NOT a model's reading: each headline gets (up-words - down-words) / (up-words + down-words) from
the word lists below, so it measures price-momentum wording ("prices fall", "rents surge") on a -1 to +1
scale. It is crude and says nothing about facts.

IMPORTANT: written from memory of the feed format and search operators (after:/before:), and NOT run
against the live service (the build environment cannot reach it). This is an unofficial feed, so Google can
change or block it; check Google's terms for your use. Each feed returns at most about 100 items, so a busy
window is capped and volume is then not reported.
"""

from __future__ import annotations

import re
import time
import urllib.error
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from email.utils import parsedate_to_datetime
from typing import Callable

from retrend.location import Location
from retrend.sources.files import SourceError

BASE = "https://news.google.com/rss/search"
TOPICS = '(housing OR "home prices" OR "real estate" OR rent OR rents OR mortgage OR "home sales")'
WINDOW_DAYS = 30
N_WINDOWS = 3
CAP = 95            # a window with this many items is treated as capped (feeds stop near 100)
PAUSE = 2.0
RETRIES = 3
BACKOFF = 15.0
MAX_HEADLINES = 8

UP = {"rise", "rises", "rising", "rose", "surge", "surges", "surging", "jump", "jumps", "soar", "soars", "climb",
      "climbs", "gain", "gains", "growth", "grow", "grows", "boom", "booming", "hot", "strong", "stronger",
      "rebound", "rebounds", "recover", "recovers", "recovery", "record", "up", "higher", "increase", "increases",
      "improve", "improves", "improving", "tight", "bidding", "demand", "frenzy", "heats"}
DOWN = {"fall", "falls", "falling", "fell", "drop", "drops", "dropped", "plunge", "plunges", "slump", "slumps",
        "decline", "declines", "declining", "crash", "crashes", "cut", "cuts", "cooling", "cools", "cooled",
        "slow", "slows", "slowing", "slowdown", "weak", "weaker", "down", "lower", "decrease", "decreases",
        "sink", "sinks", "foreclosure", "foreclosures", "layoffs", "glut", "oversupply", "bubble", "downturn",
        "stall", "stalls", "stalled", "softening", "softens", "slide", "slides", "tumble", "tumbles"}


class NewsError(SourceError):
    pass


def score_headline(title: str) -> float | None:
    """(up - down) / (up + down) over the headline's words, or None when it has neither kind of word."""
    words = re.findall(r"[a-z]+", title.lower())
    up = sum(w in UP for w in words)
    down = sum(w in DOWN for w in words)
    return None if up + down == 0 else round((up - down) / (up + down), 3)


def build_query(loc: Location, start: date, end: date) -> str:
    return f'"{loc.city}" {TOPICS} after:{start.isoformat()} before:{end.isoformat()}'


def url_for(loc: Location, start: date, end: date) -> str:
    return BASE + "?" + urllib.parse.urlencode(
        {"q": build_query(loc, start, end), "hl": "en-US", "gl": "US", "ceid": "US:en"})


_last_request = 0.0


def _wait_turn(pause: float, sleep: Callable[[float], None]) -> None:
    global _last_request
    gap = _last_request + pause - time.monotonic()
    if gap > 0:
        sleep(gap)
    _last_request = time.monotonic()


def _get_text(open_url: Callable, url: str, *, pause: float, sleep: Callable[[float], None], retries: int) -> str:
    for attempt in range(retries + 1):
        _wait_turn(pause, sleep)
        try:
            raw = open_url(url).read()
            return raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 503):
                raise NewsError(f"Google News request failed: HTTP {exc.code}") from None
            if attempt == retries:
                raise NewsError(f"Google News says too many requests (HTTP {exc.code}). Tried {retries + 1} "
                                "times; wait a few minutes and ask again.") from None
            sleep(BACKOFF * (attempt + 1))
        except OSError as exc:
            raise NewsError(f"Google News request failed: {type(exc).__name__}: {exc}") from None
    raise AssertionError("unreachable")


def parse_feed(text: str) -> list[dict]:
    """RSS text -> [{"title", "url", "domain", "date"}]. Rejects documents that declare entities."""
    head = text[:2000].upper()
    if "<!DOCTYPE" in head or "<!ENTITY" in head:
        raise NewsError("the feed contained a document type declaration; refusing to parse it")
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        raise NewsError("Google News did not return an RSS feed: " + " ".join(text.split())[:160]) from None
    items = []
    for it in root.iter("item"):
        title = " ".join((it.findtext("title") or "").split())
        link = (it.findtext("link") or "").strip()
        src = it.find("source")
        domain = " ".join(((src.text if src is not None else "") or "").split())[:80]
        try:
            day = parsedate_to_datetime(it.findtext("pubDate") or "").date()
        except (TypeError, ValueError):
            continue
        if title and link.startswith(("http://", "https://")):
            if domain and title.endswith(" - " + domain):
                title = title[: -len(domain) - 3]
            items.append({"title": title[:200], "url": link, "domain": domain, "date": day})
    return items


def fetch_news(loc: Location, open_url: Callable, today: date, *, pause: float = PAUSE,
               sleep: Callable[[float], None] = time.sleep, retries: int = RETRIES) -> dict:
    """-> {"tone": [{date, value}], "volume": [{date, value}], "headlines": [...], "capped": bool}"""
    items: dict[str, dict] = {}
    capped = False
    end = today + timedelta(days=1)
    for _ in range(N_WINDOWS):
        start = end - timedelta(days=WINDOW_DAYS)
        found = parse_feed(_get_text(open_url, url_for(loc, start, end), pause=pause, sleep=sleep, retries=retries))
        capped |= len(found) >= CAP
        for it in found:
            if start <= it["date"] < end:
                items.setdefault(it["url"], it)
        end = start
    by_day: dict[date, list[dict]] = {}
    for it in items.values():
        by_day.setdefault(it["date"], []).append(it)
    tone, volume = [], []
    for day in sorted(by_day):
        volume.append({"date": day.isoformat(), "value": float(len(by_day[day]))})
        scores = [s for s in (score_headline(i["title"]) for i in by_day[day]) if s is not None]
        if scores:
            tone.append({"date": day.isoformat(), "value": round(sum(scores) / len(scores), 3)})
    newest = sorted(items.values(), key=lambda i: i["date"], reverse=True)[:MAX_HEADLINES]
    return {"tone": tone, "volume": volume, "capped": capped, "source": "gnews",
            "headlines": [{"title": i["title"], "url": i["url"], "domain": i["domain"],
                           "seen": i["date"].strftime("%Y%m%d")} for i in newest]}
