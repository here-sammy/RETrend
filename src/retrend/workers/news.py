"""News worker: tone and volume of local housing coverage (GDELT). Context only, live data only.

Not point-in-time: GDELT searches only about the last three months, so a past date cannot be rebuilt.
An as-of run uses a snapshot this project stored at or before that date (the monthly refresh makes
them) and otherwise raises NoDataAsOf, so the backtest never sees news from the future.
Headlines are shown to the person but never given to a model (they are untrusted text).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta, timezone

from retrend.llm import JsonJudge
from retrend.location import Location
from retrend.schema import Source, WorkerReport
from retrend.sources.gdelt import fetch_news as fetch_gdelt
from retrend.sources.gnews import fetch_news as fetch_gnews
from retrend.store import RawStore
from retrend.workers.base import BaseWorker, NoDataAsOf
from retrend.workers.judging import judge_with_fallback
from retrend.workers.news_metrics import PROMPT_METRIC_KEYS, TONE_MOVE, compute_metrics

LIVE_MAX_AGE = timedelta(days=1)
MAX_CONFIDENCE = 0.3
STALE_AFTER_DAYS = 10
GNEWS_TONE_MOVE = 0.15  # placeholder, untuned: headline-wording scores run from -1 to +1

SYSTEM_PROMPT = """\
You are one component of a real-estate price-trend system. Your only input is a summary of how news \
coverage about housing in one city has changed: average tone (negative is gloomier) and volume. Judge what \
this implies for home prices over roughly the next 6 to 12 months, using ONLY these numbers. News tone is a \
noisy, partly lagging signal, so keep confidence low.

Rules:
- direction: "up" if tone improved clearly, "down" if it worsened clearly, "flat" if small or mixed. A big \
rise in volume with worse tone is stronger evidence for "down" than either alone.
- confidence: 0 to 1, your own estimate; rarely above 0.3.
- summary: 1 to 3 sentences citing only numbers in the input.
- caveats: up to 3 short strings.
- Values may be null. Do not use outside knowledge about places or events.

Reply with ONLY a JSON object, no markdown fences:
{"direction": "up|flat|down", "confidence": 0.0, "summary": "...", "caveats": ["..."]}
"""


def build_user_prompt(metrics: dict) -> str:
    shown = {k: metrics[k] for k in PROMPT_METRIC_KEYS if metrics[k] is not None}
    return ("Housing news coverage for one city. tone_recent is the average tone over the last 30 days, "
            "tone_base over the 60 days before that; tone comes from automated scoring of headline wording, "
            "so judge the change rather than the level. Volume is coverage per day.\n\n"
            + json.dumps(shown, indent=2))


def _latest(payload: dict) -> date | None:
    t = payload.get("tone") or []
    return date.fromisoformat(t[-1]["date"]) if t else None


class NewsWorker(BaseWorker):
    name = "news"

    def __init__(self, store: RawStore, logger: logging.Logger | None = None, *, open_url,
                 judge: JsonJudge | None, fallback_to_baseline: bool = True, pause: float | None = None,
                 source: str | None = None) -> None:
        super().__init__(store, logger)
        self._open_url, self._judge, self._fallback = open_url, judge, fallback_to_baseline
        self._pause = pause
        self._source = (source or os.environ.get("RETREND_NEWS_SOURCE") or "gnews").strip().lower()
        if self._source not in ("gnews", "gdelt"):
            raise ValueError(f"unknown news source {self._source!r}; use 'gnews' or 'gdelt'")

    def _fetch(self, location: Location) -> dict:
        from retrend.sources import gdelt, gnews
        if self._open_url is None:
            import urllib.request

            def opener(url):
                req = urllib.request.Request(url, headers={"User-Agent": "retrend/0.1 (news tone)"})
                return urllib.request.urlopen(req, timeout=60)
        else:
            opener = self._open_url
        if self._source == "gdelt":
            return fetch_gdelt(location, opener, pause=gdelt.PAUSE if self._pause is None else self._pause)
        return fetch_gnews(location, opener, self.store.clock().date(),
                           pause=gnews.PAUSE if self._pause is None else self._pause)

    def _run(self, location: Location, as_of: datetime | None) -> WorkerReport:
        if as_of is not None and as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        key = f"coverage|{self._source}|{location.key}"
        if as_of is not None:
            rec = self.store.get_as_of("news", key, as_of)
            if rec is None:
                raise NoDataAsOf("news cannot be rebuilt for a past date; no snapshot was stored by then")
            ref = as_of.astimezone(timezone.utc).date()
        else:
            rec = self.store.fetch_or_cache("news", key, lambda: self._fetch(location),
                                            max_age=LIVE_MAX_AGE, data_as_of=_latest)
            ref = self.store.clock().date()
        payload = rec.json()
        move = GNEWS_TONE_MOVE if self._source == "gnews" else TONE_MOVE
        capped = bool(payload.get("capped"))
        metrics = compute_metrics(payload.get("tone", []), payload.get("volume", []), ref, move)
        if metrics is None:
            raise NoDataAsOf(f"too little housing news coverage for {location.city} "
                             "(needs about 10 days of tone readings in each window)")
        if metrics is not None and capped:  # feeds stop near 100 items, so counts would just show the cap
            metrics["volume_recent"] = metrics["volume_base"] = metrics["volume_change_pct"] = None
        direction, conf, summary, judged_by, extra, attempts, cached = judge_with_fallback(
            self._judge, system=SYSTEM_PROMPT, user=build_user_prompt(metrics),
            baseline=(metrics["baseline_direction"], f"Rule-based: {metrics['baseline_rule']}."),
            baseline_name="news-tone", fallback=self._fallback, logger=self.logger, worker=self.name)
        conf = min(conf, MAX_CONFIDENCE)
        caveats = [
            f"Matches articles that mention '{location.city}' and housing words; other places with the same name "
            "can be mixed in.",
            "Tone is an automated score of headline wording, not a measure of facts, and it often lags the market.",
            "Only about three months of history exist, so this signal has not been backtested.",
        ]
        if metrics["days_since_latest_obs"] > STALE_AFTER_DAYS:
            caveats.append(f"Latest reading is {metrics['days_since_latest_obs']} days old.")
        if capped:
            caveats.append("A search returned its maximum number of headlines, so coverage volume is not reported.")
        caveats += extra
        km = dict(metrics)
        km.update(headlines_json=json.dumps(payload.get("headlines", [])), judge_attempts=attempts, judge_cached=cached)
        return WorkerReport(
            worker=self.name, location=location, signal="news_tone", direction=direction, confidence=conf,
            key_metrics=km, data_as_of=date.fromisoformat(metrics["latest_obs_date"]), geography_match="approximate",
            caveats=caveats, sources=[Source(
                name="Google News RSS (headline wording scored in code)" if self._source == "gnews"
                else "GDELT DOC 2.0 news index",
                url="https://news.google.com/" if self._source == "gnews" else "https://www.gdeltproject.org/",
                retrieved=rec.fetched_at)],
            summary=summary, judged_by=judged_by)
