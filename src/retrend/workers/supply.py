"""Supply / inventory worker: Redfin Data Center market tracker."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone

from retrend.llm import JsonJudge
from retrend.location import Location
from retrend.schema import Source, WorkerReport
from retrend.sources.redfin import NoRegionMatch, RedfinLoader
from retrend.store import RawStore
from retrend.workers.base import BaseWorker, NoDataAsOf
from retrend.workers.judging import judge_with_fallback, load_payload
from retrend.workers.supply_metrics import PROMPT_METRIC_KEYS, compute_metrics

LIVE_MAX_AGE = timedelta(days=1)
PUBLICATION_LAG_DAYS = 21
DATA_CENTER_URL = "https://www.redfin.com/news/data-center/"

SYSTEM_PROMPT = """\
You are one component of a real-estate price-trend system. Your only input is a summary of \
housing supply and demand indicators for one market (Redfin). Judge the price pressure these \
imply over roughly the next 6 to 12 months, using ONLY these numbers. Other components cover \
price history, mortgage rates and the local economy.

Rules:
- direction: "up" if supply is tight and tightening (low or falling months of supply, short \
and shortening days on market, many sales above list), "down" if supply is building and \
demand is cooling (rising inventory and months of supply, more price drops, longer days on \
market), "flat" if mixed or balanced.
- confidence: 0 to 1, your own estimate. Lower it when indicators conflict.
- summary: 1 to 3 sentences citing only numbers in the input.
- caveats: up to 3 short strings.
- Values may be null when unavailable. Do not use outside knowledge about places or events.

Reply with ONLY a JSON object, no markdown fences:
{"direction": "up|flat|down", "confidence": 0.0, "summary": "...", "caveats": ["..."]}
"""


def build_user_prompt(metrics: dict) -> str:
    shown = {k: metrics[k] for k in PROMPT_METRIC_KEYS if metrics[k] is not None}
    return (
        "Housing supply/demand summary (3-month averages; *_yoy compares with the same "
        "3 months a year earlier; price_drops_pct is the share of listings with a price cut; "
        "sold_above_list_pct the share of sales above list). Indicators the data source does "
        "not provide are omitted.\n\n" + json.dumps(shown, indent=2)
    )


def _latest(payload: dict) -> date | None:
    days = [r["period_end"] for r in payload.get("rows", []) if r.get("period_end")]
    return date.fromisoformat(max(days)) if days else None


class SupplyWorker(BaseWorker):
    name = "supply"

    def __init__(self, store: RawStore, logger: logging.Logger | None = None, *,
                 loader: RedfinLoader, judge: JsonJudge | None,
                 geos: tuple[str, ...] = ("metro",), fallback_to_baseline: bool = True) -> None:
        super().__init__(store, logger)
        self._loader, self._judge, self._geos, self._fallback = loader, judge, geos, fallback_to_baseline

    def dom_series(self, location: Location) -> dict:
        """Median days on market by month for charts: {"geo", "region", "points": [(date, days)]}."""
        last: Exception | None = None
        for g in self._geos:
            try:
                rec, _pit = load_payload(
                    self.store, "redfin", f"tracker|{g}|{location.key}",
                    lambda g=g: self._loader.extract(location, g), None, _latest, LIVE_MAX_AGE)
            except NoRegionMatch as exc:
                last = exc
                continue
            payload = rec.json()
            pts = [(r["period_end"], float(r["median_dom"])) for r in payload["rows"]
                   if r.get("median_dom") is not None]
            return {"geo": g, "region": payload["region"]["name"], "points": pts}
        raise last or SourceError("no Redfin geography configured")

    def _run(self, location: Location, as_of: datetime | None) -> WorkerReport:
        if as_of is not None and as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        ref = as_of.astimezone(timezone.utc).date() if as_of else self.store.clock().date()

        rec = geo = None
        pit = True
        last: Exception | None = None
        for g in self._geos:
            try:
                rec, pit = load_payload(
                    self.store, "redfin", f"tracker|{g}|{location.key}",
                    lambda g=g: self._loader.extract(location, g), as_of, _latest, LIVE_MAX_AGE)
                geo = g
                break
            except NoRegionMatch as exc:
                last = exc
        if rec is None:
            raise last  # type: ignore[misc]

        payload = rec.json()
        # Live runs: whatever is in the file is already published. Backtests: apply the lag.
        lag = timedelta(days=PUBLICATION_LAG_DAYS if as_of else 0)
        rows = [r for r in payload["rows"]
                if date.fromisoformat(r["period_end"]) + lag <= ref]
        if len(rows) < 3:
            raise NoDataAsOf(f"only {len(rows)} Redfin periods available as of {ref}")

        metrics = compute_metrics(rows)
        direction, conf, summary, judged_by, extra, attempts, cached = judge_with_fallback(
            self._judge, system=SYSTEM_PROMPT, user=build_user_prompt(metrics),
            baseline=(metrics["baseline_direction"], f"Rule-based: {metrics['baseline_rule']}."),
            baseline_name="months-of-supply", fallback=self._fallback, logger=self.logger,
            worker=self.name)

        name = payload["region"]["name"]
        exact = geo == "city" and payload["quality"] == "exact"
        caveats = []
        if not exact:
            caveats.append(f"No exact city series; used Redfin {geo} region '{name}'.")
        if not pit:
            caveats.append(
                "Not point-in-time: Redfin revises history, so this uses the current file "
                "truncated to the as-of date.")
        if metrics["months_of_supply_source"] != "reported":
            caveats.append("Months of supply is estimated as active listings divided by homes "
                           "sold in the month; Redfin's own definition may differ.")
        if metrics["inventory_yoy_pct"] is None:
            caveats.append("No year-over-year data in this file; trends are not available.")
        if metrics["price_drops_pct"] is None:
            caveats.append("Price-drop and sale-to-list indicators are not in this data file.")
        caveats.extend(extra)

        km = dict(metrics)
        km.update(region_name=name, geo=geo, point_in_time=pit, judge_attempts=attempts,
                  judge_cached=cached)
        return WorkerReport(
            worker=self.name, location=location, signal="supply_inventory", direction=direction,
            confidence=conf, key_metrics=km, data_as_of=date.fromisoformat(rows[-1]["period_end"]),
            geography_match="exact" if exact else "approximate", caveats=caveats,
            sources=[Source(name="Redfin Data Center, market tracker (All Residential)",
                            url=DATA_CENTER_URL, retrieved=rec.fetched_at)],
            summary=summary, judged_by=judged_by)
