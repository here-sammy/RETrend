"""Building-permits worker: metro permit trend (Census Building Permits Survey via FRED)."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime

from retrend.fred import series_page_url
from retrend.llm import JsonJudge
from retrend.location import Location
from retrend.schema import Source, WorkerReport
from retrend.sources.fred_metro import MetroSeriesFinder, NoMetroSeries
from retrend.store import RawStore
from retrend.workers.base import BaseWorker, NoDataAsOf
from retrend.workers.judging import judge_with_fallback
from retrend.workers.monthly import pull_monthly, run_date
from retrend.workers.permits_metrics import PROMPT_METRIC_KEYS, RULES, compute_metrics

MAX_CONFIDENCE = 0.4
STALE_AFTER_DAYS = 150

READING = {
    "demand": "Read rising permits as builders responding to strong demand (supports prices) and falling permits as a pullback.",
    "supply": "Read rising permits as new supply coming to market (softens prices) and falling permits as less incoming supply.",
}

SYSTEM_PROMPT = """\
You are one component of a real-estate price-trend system. Your only input is a summary of residential \
building permits issued in one metro area. Judge what this implies for home prices over roughly the next \
6 to 12 months, using ONLY these numbers and the stated reading. Permits are a noisy signal, so keep \
confidence low.

{reading}

Rules:
- direction: "up", "down" or "flat" for the implied pressure on PRICES; "flat" for small or mixed changes.
- confidence: 0 to 1, your own estimate; rarely above 0.4.
- summary: 1 to 3 sentences citing only numbers in the input.
- caveats: up to 3 short strings.
- Values may be null. Do not use outside knowledge about places or events.

Reply with ONLY a JSON object, no markdown fences:
{{"direction": "up|flat|down", "confidence": 0.0, "summary": "...", "caveats": ["..."]}}
"""


def build_user_prompt(metrics: dict) -> str:
    shown = {k: metrics[k] for k in PROMPT_METRIC_KEYS if metrics[k] is not None}
    return ("Metro-area residential building permits (count of housing units or structures authorized). "
            "permits_12m_yoy_pct compares the latest 12 months with the 12 before; "
            "permits_3m_vs_year_ago_pct compares the latest 3 months with the same 3 months a year earlier.\n\n"
            + json.dumps(shown, indent=2))


class PermitsWorker(BaseWorker):
    name = "permits"

    def __init__(self, store: RawStore, logger: logging.Logger | None = None, *, fred_source,
                 finder: MetroSeriesFinder, judge: JsonJudge | None, fallback_to_baseline: bool = True,
                 rule: str | None = None) -> None:
        super().__init__(store, logger)
        self._fred, self._finder, self._judge, self._fallback = fred_source, finder, judge, fallback_to_baseline
        self._rule = (rule or os.environ.get("RETREND_PERMITS_RULE") or "context").strip().lower()
        if self._rule not in RULES:
            raise ValueError(f"unknown permits rule {self._rule!r}; use one of {list(RULES)}")

    def _run(self, location: Location, as_of: datetime | None) -> WorkerReport:
        ref, as_of_date = run_date(self.store, as_of)
        if self._fred is None:
            raise NoMetroSeries("the permits signal needs FRED_API_KEY")
        series = self._finder.find("permits", location)
        rec, obs = pull_monthly(self.store, self._fred, series["series_id"], ref, as_of_date)
        metrics = compute_metrics(obs, ref, self._rule)
        if metrics is None:
            raise NoDataAsOf(f"fewer than 24 unbroken months of permits for {location.key} as of {ref}")
        baseline = (metrics["baseline_direction"], f"Rule-based: {metrics['baseline_rule']}.")
        if self._rule == "context":
            # nothing to judge: the rule takes no side, so a model would only be inventing one
            direction, conf, summary, judged_by, extra, attempts, cached = (
                baseline[0], 0.2, baseline[1], "rule:permits-context", ["No side taken: permits are shown as context only."], 0, False)
        else:
            direction, conf, summary, judged_by, extra, attempts, cached = judge_with_fallback(
                self._judge, system=SYSTEM_PROMPT.format(reading=READING[self._rule]), user=build_user_prompt(metrics),
                baseline=baseline, baseline_name=f"permits-{self._rule}", fallback=self._fallback,
                logger=self.logger, worker=self.name)
        conf = min(conf, MAX_CONFIDENCE)
        caveats = [f"Uses the {series['area']} metro area as a stand-in for {location.city}.",
                   "Whether more permits push prices up or down is not settled; the reading used is "
                   f"'{self._rule}'."]
        if metrics["days_since_latest_obs"] > STALE_AFTER_DAYS:
            caveats.append(f"Latest reading is {metrics['days_since_latest_obs']} days old.")
        caveats += extra
        km = dict(metrics)
        km.update(permits_series=series["series_id"], metro_area=series["area"], judge_attempts=attempts,
                  judge_cached=cached)
        return WorkerReport(
            worker=self.name, location=location, signal="building_permits", direction=direction,
            confidence=conf, key_metrics=km, data_as_of=obs[-1].day, geography_match="approximate",
            caveats=caveats, sources=[Source(name=series["title"], url=series_page_url(series["series_id"]),
                                             retrieved=rec.fetched_at)],
            summary=summary, judged_by=judged_by)
