"""School enrollment worker: district enrollment trend as a (weak) demand signal. Context only."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone

from retrend.llm import JsonJudge
from retrend.location import Location
from retrend.schema import Source, WorkerReport
from retrend.sources.schools import NoDistrictMatch, SchoolsLoader
from retrend.store import RawStore
from retrend.workers.base import BaseWorker, NoDataAsOf
from retrend.workers.judging import judge_with_fallback, load_payload
from retrend.workers.schools_metrics import PROMPT_METRIC_KEYS, compute_metrics

LIVE_MAX_AGE = timedelta(days=1)
# Enrollment is counted in the fall; assume a year's figures are public by June of the next year.
PUBLICATION_LAG = (1, 6, 30)
CCD_URL = "https://nces.ed.gov/ccd/"

SYSTEM_PROMPT = """\
You are one component of a real-estate price-trend system. Your only input is a summary of \
public-school enrollment trends for one school district. Judge what this implies about household \
demand for housing there over roughly the next 6 to 12 months, using ONLY these numbers. \
Enrollment is a slow and weak signal (births, school choice and policy also move it), so keep \
confidence low to moderate. Other components cover prices, supply and mortgage rates.

Rules:
- direction: "up" if enrollment is growing and the trend is steady or improving, "down" if it is \
shrinking and the trend is steady or worsening, "flat" if small, mixed or reversing.
- confidence: 0 to 1, your own estimate; rarely above 0.5 for this signal.
- summary: 1 to 3 sentences citing only numbers in the input.
- caveats: up to 3 short strings.
- Values may be null. Do not use outside knowledge about places or events.

Reply with ONLY a JSON object, no markdown fences:
{"direction": "up|flat|down", "confidence": 0.0, "summary": "...", "caveats": ["..."]}
"""


def build_user_prompt(metrics: dict) -> str:
    shown = {k: metrics[k] for k in PROMPT_METRIC_KEYS if metrics[k] is not None}
    return ("School district enrollment summary (percent changes are fall-to-fall; "
            "enrollment_3y_cagr_pct is the average yearly change over 3 years).\n\n"
            + json.dumps(shown, indent=2))


def _latest(payload: dict) -> date | None:
    ys = [r["year"] for r in payload.get("rows", [])]
    return date(max(ys), 10, 1) if ys else None


class SchoolsWorker(BaseWorker):
    name = "schools"

    def __init__(self, store: RawStore, logger: logging.Logger | None = None, *,
                 loader: SchoolsLoader, judge: JsonJudge | None, fallback_to_baseline: bool = True) -> None:
        super().__init__(store, logger)
        self._loader, self._judge, self._fallback = loader, judge, fallback_to_baseline

    def _run(self, location: Location, as_of: datetime | None) -> WorkerReport:
        if as_of is not None and as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        ref = as_of.astimezone(timezone.utc).date() if as_of else self.store.clock().date()
        rec, pit = load_payload(
            self.store, "schools", f"enrollment|{location.key}",
            lambda: self._loader.extract(location), as_of, _latest, LIVE_MAX_AGE)
        payload = rec.json()
        rows = [r for r in payload["rows"]
                if as_of is None or date(r["year"] + PUBLICATION_LAG[0], PUBLICATION_LAG[1], PUBLICATION_LAG[2]) <= ref]
        if len(rows) < 3:
            raise NoDataAsOf(f"only {len(rows)} years of enrollment available as of {ref}")
        metrics = compute_metrics(rows)
        direction, conf, summary, judged_by, extra, attempts, cached = judge_with_fallback(
            self._judge, system=SYSTEM_PROMPT, user=build_user_prompt(metrics),
            baseline=(metrics["baseline_direction"], f"Rule-based: {metrics['baseline_rule']}."),
            baseline_name="enrollment-trend", fallback=self._fallback, logger=self.logger,
            worker=self.name)
        conf = min(conf, 0.5)  # weak signal: never present it as confident
        caveats = [
            "Enrollment is a slow, indirect demand signal; births, school choice and boundary changes also move it.",
            f"District '{payload['district']}' stands in for the city; district boundaries rarely match city limits.",
        ]
        if payload["quality"] != "exact":
            caveats.append(f"{payload['n_candidates']} districts matched; used the largest. Others: {payload['other_candidates']}.")
        if not pit:
            caveats.append("Not point-in-time: uses the current file truncated to the as-of date.")
        caveats.extend(extra)
        km = dict(metrics)
        km.update(enrollment_history_json=json.dumps(rows), district=payload["district"], point_in_time=pit, judge_attempts=attempts, judge_cached=cached)
        return WorkerReport(
            worker=self.name, location=location, signal="school_enrollment", direction=direction,
            confidence=conf, key_metrics=km, data_as_of=date(metrics["latest_year"], 10, 1),
            geography_match="approximate", caveats=caveats,
            sources=[Source(name=f"School enrollment file ({payload['file']})", url=CCD_URL, retrieved=rec.fetched_at)],
            summary=summary, judged_by=judged_by)
