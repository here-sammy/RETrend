"""Local jobs worker: metro unemployment rate and payroll growth as a demand signal (FRED / BLS)."""

from __future__ import annotations

import json
import logging
from datetime import datetime

from retrend.fred import series_page_url
from retrend.llm import JsonJudge
from retrend.location import Location
from retrend.schema import Source, WorkerReport
from retrend.sources.fred_metro import MetroSeriesFinder, NoMetroSeries
from retrend.store import RawStore
from retrend.workers.base import BaseWorker
from retrend.workers.jobs_metrics import PROMPT_METRIC_KEYS, compute_metrics
from retrend.workers.judging import judge_with_fallback
from retrend.workers.monthly import pull_monthly, run_date

STALE_AFTER_DAYS = 150
MAX_CONFIDENCE = 0.6

SYSTEM_PROMPT = """\
You are one component of a real-estate price-trend system. Your only input is a summary of the local \
labor market for one metro area (unemployment rate and payroll job growth). Judge what this implies \
about household demand for housing there over roughly the next 6 to 12 months, using ONLY these numbers. \
Other components cover prices, supply, mortgage rates and schools.

Rules:
- direction: "up" if jobs are growing and unemployment is flat or falling, "down" if jobs are shrinking \
or unemployment is rising meaningfully, "flat" if small, mixed or reversing.
- confidence: 0 to 1, your own estimate; lower when the two measures disagree or payroll data is missing.
- summary: 1 to 3 sentences citing only numbers in the input.
- caveats: up to 3 short strings.
- Values may be null. Do not use outside knowledge about places or events.

Reply with ONLY a JSON object, no markdown fences:
{"direction": "up|flat|down", "confidence": 0.0, "summary": "...", "caveats": ["..."]}
"""


def build_user_prompt(metrics: dict) -> str:
    shown = {k: metrics[k] for k in PROMPT_METRIC_KEYS if metrics[k] is not None}
    return ("Metro-area labor market summary. unemployment_rate is a percent; *_pp are percentage-point "
            "changes; unemployment_5y_percentile is the share of the last five years at or below the "
            "latest rate; payroll_* are total nonfarm job growth percent.\n\n" + json.dumps(shown, indent=2))


class JobsWorker(BaseWorker):
    name = "jobs"

    def __init__(self, store: RawStore, logger: logging.Logger | None = None, *, fred_source,
                 finder: MetroSeriesFinder, judge: JsonJudge | None, fallback_to_baseline: bool = True) -> None:
        super().__init__(store, logger)
        self._fred, self._finder, self._judge, self._fallback = fred_source, finder, judge, fallback_to_baseline

    def _run(self, location: Location, as_of: datetime | None) -> WorkerReport:
        ref, as_of_date = run_date(self.store, as_of)
        if self._fred is None:
            raise NoMetroSeries("the jobs signal needs FRED_API_KEY")
        u_series = self._finder.find("unemployment", location)
        u_rec, unemp = pull_monthly(self.store, self._fred, u_series["series_id"], ref, as_of_date)
        payroll, p_series, p_rec, notes = None, None, None, []
        try:
            p_series = self._finder.find("payrolls", location)
            p_rec, payroll = pull_monthly(self.store, self._fred, p_series["series_id"], ref, as_of_date)
        except NoMetroSeries as exc:
            notes.append(f"No payroll series found, so this uses unemployment only ({str(exc)[:120]}).")
        metrics = compute_metrics(unemp, payroll, ref)
        direction, conf, summary, judged_by, extra, attempts, cached = judge_with_fallback(
            self._judge, system=SYSTEM_PROMPT, user=build_user_prompt(metrics),
            baseline=(metrics["baseline_direction"], f"Rule-based: {metrics['baseline_rule']}."),
            baseline_name="jobs-trend", fallback=self._fallback, logger=self.logger, worker=self.name)
        conf = min(conf, MAX_CONFIDENCE)
        caveats = [f"Uses the {u_series['area']} metro area as a stand-in for {location.city}; jobs data is not "
                   "available for the city alone.",
                   "Employment is a lagging, indirect signal of housing demand."]
        if metrics["days_since_latest_obs"] > STALE_AFTER_DAYS:
            caveats.append(f"Latest reading is {metrics['days_since_latest_obs']} days old.")
        caveats += notes + extra
        km = dict(metrics)
        km.update(unemployment_series=u_series["series_id"], payroll_series=p_series["series_id"] if p_series else None,
                  metro_area=u_series["area"], judge_attempts=attempts, judge_cached=cached)
        sources = [Source(name=u_series["title"], url=series_page_url(u_series["series_id"]), retrieved=u_rec.fetched_at)]
        if p_series and p_rec:
            sources.append(Source(name=p_series["title"], url=series_page_url(p_series["series_id"]),
                                  retrieved=p_rec.fetched_at))
        return WorkerReport(
            worker=self.name, location=location, signal="local_jobs", direction=direction, confidence=conf,
            key_metrics=km, data_as_of=unemp[-1].day, geography_match="approximate", caveats=caveats,
            sources=sources, summary=summary, judged_by=judged_by)
