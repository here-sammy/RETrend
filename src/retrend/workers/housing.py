"""Housing price worker: Zillow ZHVI (values) and ZORI (rents)."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone

from retrend.llm import JsonJudge
from retrend.location import Location
from retrend.schema import Source, WorkerReport
from retrend.sources.files import SourceError
from retrend.sources.zillow import DESCRIPTIONS, NoRegionMatch, ZillowLoader
from retrend.store import RawStore
from retrend.workers.base import BaseWorker, NoDataAsOf
from retrend.workers.housing_metrics import PROMPT_METRIC_KEYS, compute_metrics
from retrend.workers.judging import judge_with_fallback, load_payload
from retrend.workers.timeseries import Point

LIVE_MAX_AGE = timedelta(days=1)
PUBLICATION_LAG_DAYS = 20  # a month-end value is published roughly mid-following-month
RESEARCH_URL = "https://www.zillow.com/research/data/"

SYSTEM_PROMPT = """\
You are one component of a real-estate price-trend system. Your only input is a summary of \
recent home value and rent growth for one housing market (Zillow indexes). Judge the price \
pressure these trends imply over roughly the next 6 to 12 months, using ONLY these numbers. \
Other components cover supply, mortgage rates and the local economy.

Rules:
- direction: "up" if growth is positive and not decelerating sharply, "down" if values are \
falling or clearly rolling over, "flat" if there is no clear trend. Momentum is the signal.
- confidence: 0 to 1, your own estimate. Lower it when year-over-year and recent growth \
disagree, or history is short.
- summary: 1 to 3 sentences citing only numbers in the input.
- caveats: up to 3 short strings.
- Do not use outside knowledge about places, dates or events.

Reply with ONLY a JSON object, no markdown fences:
{"direction": "up|flat|down", "confidence": 0.0, "summary": "...", "caveats": ["..."]}
"""


def build_user_prompt(metrics: dict) -> str:
    shown = {k: metrics[k] for k in PROMPT_METRIC_KEYS}
    return (
        "Home value (ZHVI) and rent (ZORI) growth summary for one market. Percent figures; "
        "*_annualized_pct converts the recent change to a yearly rate; acceleration_pp is "
        "3-month annualized minus year-over-year; drawdown is vs the 5-year peak "
        "(0 = at peak).\n\n" + json.dumps(shown, indent=2)
    )


def _points(payload: dict, ref: date, lag_days: int) -> list[Point]:
    out = []
    for o in payload.get("observations", []):
        d = date.fromisoformat(o["date"])
        if d + timedelta(days=lag_days) <= ref:
            out.append(Point(d, float(o["value"])))
    return out


def _latest(payload: dict) -> date | None:
    days = [o["date"] for o in payload.get("observations", [])]
    return date.fromisoformat(max(days)) if days else None


class HousingPriceWorker(BaseWorker):
    name = "housing_prices"

    def __init__(self, store: RawStore, logger: logging.Logger | None = None, *,
                 loader: ZillowLoader, judge: JsonJudge | None,
                 geos: tuple[str, ...] = ("city", "metro"),
                 fallback_to_baseline: bool = True) -> None:
        super().__init__(store, logger)
        self._loader, self._judge, self._geos, self._fallback = loader, judge, geos, fallback_to_baseline

    def _fetch(self, location, dataset, as_of):
        """Try each geography in order -> (record, geo, point_in_time)."""
        last: Exception | None = None
        for geo in self._geos:
            key = f"{dataset}|{geo}|{location.key}"
            try:
                rec, pit = load_payload(
                    self.store, "zillow", key,
                    lambda g=geo: self._loader.extract(location, dataset, g),
                    as_of, _latest, LIVE_MAX_AGE)
                return rec, geo, pit
            except NoRegionMatch as exc:
                last = exc
        raise last or SourceError("no geography configured")

    def zhvi_series(self, location: Location) -> list[Point]:
        """The full ZHVI series this worker uses for ``location`` (no lag, no truncation).
        The evaluation harness uses it for ground truth, so truth and signal share a source."""
        rec, _geo, _pit = self._fetch(location, "zhvi", None)
        return [Point(date.fromisoformat(o["date"]), float(o["value"]))
                for o in rec.json()["observations"]]

    def series_bundle(self, location: Location) -> dict:
        """Full ZHVI and (if available) ZORI history for charts: no lag, no truncation."""
        zrec, zgeo, _ = self._fetch(location, "zhvi", None)
        z = zrec.json()
        out = {"geo": zgeo, "region": z["region"]["name"], "quality": z.get("quality"),
               "zhvi": [(o["date"], float(o["value"])) for o in z["observations"]], "zori": None}
        try:
            rrec, _g, _ = self._fetch(location, "zori", None)
            out["zori"] = [(o["date"], float(o["value"])) for o in rrec.json()["observations"]] or None
        except Exception:  # rents are optional
            pass
        out["dtp"] = None
        try:  # days to pending is optional and metro-level only
            drec, _pit = load_payload(
                self.store, "zillow", f"dtp|metro|{location.key}",
                lambda: self._loader.extract(location, "dtp", "metro"), None, _latest, LIVE_MAX_AGE)
            d = drec.json()
            out["dtp"] = {"region": d["region"]["name"], "file": d.get("file_url", ""),
                          "points": [(o["date"], float(o["value"])) for o in d["observations"]]} or None
        except Exception:
            pass
        return out

    def _run(self, location: Location, as_of: datetime | None) -> WorkerReport:
        if as_of is not None and as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        ref = as_of.astimezone(timezone.utc).date() if as_of else self.store.clock().date()

        zrec, zgeo, pit = self._fetch(location, "zhvi", as_of)
        zpayload = zrec.json()
        lag = PUBLICATION_LAG_DAYS if as_of else 0  # live: everything in the file is published
        zhvi = _points(zpayload, ref, lag)
        if len(zhvi) < 13:
            raise NoDataAsOf(f"only {len(zhvi)} ZHVI months available as of {ref}")

        caveats: list[str] = []
        zori, rrec = None, None
        try:
            rrec, rgeo, rpit = self._fetch(location, "zori", as_of)
            zori = _points(rrec.json(), ref, lag) or None
            pit = pit and rpit
            if rgeo != zgeo:
                caveats.append(f"Rents are {rgeo}-level while values are {zgeo}-level.")
        except SourceError as exc:
            caveats.append(f"Rent data unavailable: {str(exc)[:150]}")

        metrics = compute_metrics(zhvi, zori)
        direction, conf, summary, judged_by, extra, attempts, cached = judge_with_fallback(
            self._judge, system=SYSTEM_PROMPT, user=build_user_prompt(metrics),
            baseline=(metrics["baseline_direction"], f"Rule-based: {metrics['baseline_rule']}."),
            baseline_name="zhvi-momentum", fallback=self._fallback, logger=self.logger,
            worker=self.name)

        region = zpayload["region"]
        if zgeo != "city" or zpayload["quality"] != "exact":
            caveats.append(
                f"No exact city series; used Zillow {zgeo} region '{region['name']}' "
                f"({zpayload['n_candidates']} candidate(s)).")
        if not pit:
            caveats.append(
                "Not point-in-time: Zillow revises history and has no vintage API, so this "
                "uses the current file truncated to the as-of date.")
        caveats.append("ZHVI is a smoothed, modeled index, not raw sale prices.")
        caveats.extend(extra)

        km = dict(metrics)
        km.update(region_name=region["name"], geo=zgeo, point_in_time=pit,
                  judge_attempts=attempts, judge_cached=cached)
        sources = [Source(name=f"Zillow Research: {DESCRIPTIONS['zhvi']}", url=RESEARCH_URL,
                          retrieved=zrec.fetched_at)]
        if zori and rrec:
            sources.append(Source(name=f"Zillow Research: {DESCRIPTIONS['zori']}", url=RESEARCH_URL,
                                  retrieved=rrec.fetched_at))
        return WorkerReport(
            worker=self.name, location=location, signal="home_values", direction=direction,
            confidence=conf, key_metrics=km, data_as_of=zhvi[-1].day,
            geography_match="exact" if zgeo == "city" and zpayload["quality"] == "exact" else "approximate",
            caveats=caveats, sources=sources, summary=summary, judged_by=judged_by)
