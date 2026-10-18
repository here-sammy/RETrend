"""Mortgage rate worker.

Pipeline: fetch FRED data (cached) -> compute metrics (code) -> model judges (LLM) ->
assemble a WorkerReport (code). The model supplies only direction, confidence, a short
summary and caveats; every number in the report comes from code.

``direction`` follows the project convention (see schema.py): the price pressure implied
by this signal, so rising mortgage rates map to "down".
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from retrend.fred import Observation, parse_observations, series_page_url
from retrend.llm import JsonJudge
from retrend.location import Location
from retrend.logging_utils import log_event
from retrend.schema import Source, WorkerReport
from retrend.store import RawStore
from retrend.workers.base import BaseWorker, NoDataAsOf
from retrend.workers.mortgage_metrics import PROMPT_METRIC_KEYS, compute_metrics

SERIES_ID = "MORTGAGE30US"
SOURCE = "fred"
HISTORY_YEARS = 5
LIVE_MAX_AGE = timedelta(hours=12)  # the survey is weekly
FOREVER = timedelta(days=36500)  # a past vintage never changes
PMMS_METHOD_CHANGE = date(2022, 11, 17)  # Freddie Mac changed the survey methodology
STALE_AFTER_DAYS = 14
BASELINE_CONFIDENCE = 0.3  # placeholder: the rule has no calibrated confidence

SYSTEM_PROMPT = """\
You are one component of a real-estate price-trend system. Your only input is a summary \
of U.S. 30-year fixed mortgage rates (Freddie Mac weekly survey). Judge what this rate \
environment implies for U.S. home prices over roughly the next 6 to 12 months, \
considering mortgage rates ONLY. Other components cover supply, the local economy and \
so on.

Rules:
- direction: "up" if rates imply upward pressure on prices (rates falling meaningfully, \
or low relative to their recent range), "down" if they imply downward pressure (rates \
rising meaningfully, or high relative to their recent range), "flat" if there is no \
clear pressure. This is the implied pressure on PRICES, not the direction of the rate.
- confidence: a number from 0 to 1. Use lower values when signals conflict (for example \
the recent move opposes the 52-week move) or the data is stale. It is your own estimate.
- summary: 1 to 3 sentences. Cite only numbers that appear in the input.
- caveats: up to 3 short strings about what could make your judgment wrong.
- Use only the numbers provided. Do not use outside knowledge about specific dates or events.

Reply with ONLY a JSON object, no markdown fences, no other text:
{"direction": "up|flat|down", "confidence": 0.0, "summary": "...", "caveats": ["..."]}
"""

FredSource = Callable[..., dict]


class RateJudgment(BaseModel):
    """What the model is asked to return. Extra keys are ignored, not fatal."""

    model_config = ConfigDict(extra="ignore")

    direction: Literal["up", "flat", "down"]
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = Field(min_length=1, max_length=1000)
    caveats: list[str] = Field(default_factory=list)

    @field_validator("direction", mode="before")
    @classmethod
    def _normalize_direction(cls, v):
        return v.strip().lower() if isinstance(v, str) else v

    @field_validator("caveats", mode="before")
    @classmethod
    def _coerce_caveats(cls, v):
        if v is None:
            return []
        return [v] if isinstance(v, str) else v

    @field_validator("caveats")
    @classmethod
    def _tidy_caveats(cls, v: list[str]) -> list[str]:
        return [c.strip()[:200] for c in v if isinstance(c, str) and c.strip()][:3]


def build_user_prompt(metrics: dict) -> str:
    """The prompt omits the location (the rate is national) and all calendar dates, which
    also means one model call serves every location for the same week."""
    shown = {k: metrics[k] for k in PROMPT_METRIC_KEYS}
    return (
        "U.S. 30-year fixed mortgage rate summary. Changes are in basis points (bp). "
        "percentile_* is the share of weekly readings in that window at or below the "
        "latest. payment_per_100k is the monthly principal+interest payment on a $100,000 "
        "30-year loan at the latest rate.\n\n" + json.dumps(shown, indent=2)
    )


def _latest_obs_date(payload: dict) -> date | None:
    days = [r["date"] for r in payload.get("observations", []) if r.get("value") is not None]
    return date.fromisoformat(max(days)) if days else None


class MortgageRateWorker(BaseWorker):
    name = "mortgage_rates"

    def __init__(
        self,
        store: RawStore,
        logger: logging.Logger | None = None,
        *,
        fred_source: FredSource,
        judge: JsonJudge | None,
        series_id: str = SERIES_ID,
        fallback_to_baseline: bool = True,
        rule: str | None = None,
    ) -> None:
        """``judge=None`` gives the rule-based baseline only. With a judge, a failed model
        call falls back to the baseline (clearly marked in the report) unless
        ``fallback_to_baseline`` is false, in which case the error propagates."""
        super().__init__(store, logger)
        self._fred_source = fred_source
        self._judge = judge
        self._series_id = series_id
        self._fallback = fallback_to_baseline
        # "13w" (default, the original rule) or "affordability" (level-based; see mortgage_metrics).
        # Can also be chosen with the RETREND_MORTGAGE_RULE environment variable.
        self._rule = (rule or os.environ.get("RETREND_MORTGAGE_RULE") or "13w").strip().lower()
        if self._rule not in ("13w", "affordability"):
            raise ValueError(f"unknown mortgage rule {self._rule!r}; use '13w' or 'affordability'")

    def _run(self, location: Location, as_of: datetime | None) -> WorkerReport:
        if as_of is not None and as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        ref = as_of.astimezone(timezone.utc).date() if as_of else self.store.clock().date()
        as_of_date = ref if as_of is not None else None
        start = ref - timedelta(days=365 * HISTORY_YEARS)

        # A past vintage never changes, so as-of pulls are cached forever; live pulls expire.
        key = f"{self._series_id}|asof={as_of_date}" if as_of_date else f"{self._series_id}|live"
        record = self.store.fetch_or_cache(
            SOURCE,
            key,
            lambda: self._fred_source(
                self._series_id,
                observation_start=start,
                observation_end=as_of_date,
                as_of=as_of_date,
            ),
            max_age=FOREVER if as_of_date else LIVE_MAX_AGE,
            data_as_of=_latest_obs_date,
        )

        # Belt and braces: never use a reading dated after the run date.
        observations = [o for o in parse_observations(record.json()) if o.day <= ref]
        if not observations:
            raise NoDataAsOf(f"no {self._series_id} readings available as of {ref}")

        metrics = compute_metrics(observations, ref)
        direction, confidence, summary, judged_by, extra_caveats, attempts, cached = self._judgment(
            metrics
        )

        caveats = [
            "National average rate applied to this location; local lending conditions differ.",
        ]
        if observations[0].day < PMMS_METHOD_CHANGE <= observations[-1].day:
            caveats.append(
                "Freddie Mac changed the survey methodology on 2022-11-17; rate levels "
                "before and after are not strictly comparable."
            )
        if metrics["days_since_latest_obs"] > STALE_AFTER_DAYS:
            caveats.append(
                f"Latest reading is {metrics['days_since_latest_obs']} days old; "
                "the survey is normally weekly."
            )
        caveats.extend(extra_caveats)

        key_metrics = {k: v for k, v in metrics.items()}
        key_metrics["series_id"] = self._series_id
        key_metrics["judge_attempts"] = attempts
        key_metrics["judge_cached"] = cached

        return WorkerReport(
            worker=self.name,
            location=location,
            signal="mortgage_rates",
            direction=direction,
            confidence=confidence,
            key_metrics=key_metrics,
            data_as_of=observations[-1].day,
            geography_match="approximate",
            caveats=caveats,
            sources=[
                Source(
                    name="Freddie Mac PMMS, 30-Year Fixed Rate Mortgage Average (via FRED)",
                    url=series_page_url(self._series_id),
                    retrieved=record.fetched_at,
                )
            ],
            summary=summary,
            judged_by=judged_by,
        )

    def _judgment(self, metrics: dict):
        """-> (direction, confidence, summary, judged_by, caveats, attempts, cached)"""
        if self._rule == "affordability":
            baseline = (metrics["affordability_direction"], BASELINE_CONFIDENCE,
                        f"Rule-based: {metrics['affordability_rule']}.")
            label = "rule:affordability-level"
        else:
            baseline = (metrics["baseline_direction"], BASELINE_CONFIDENCE,
                        f"Rule-based: {metrics['baseline_rule']}.")
            label = "rule:baseline-13w"
        if self._judge is None:
            return (*baseline, label, ["Rule-based baseline only; no model used."], 0, False)
        try:
            result = self._judge.judge(SYSTEM_PROMPT, build_user_prompt(metrics), RateJudgment)
        except Exception as exc:
            log_event(self.logger, "judge_failed", worker=self.name, model=self._judge.model_name,
                      error_type=type(exc).__name__, error=str(exc)[:300])
            if not self._fallback:
                raise
            return (
                *baseline,
                f"{label} (fallback from {self._judge.model_name})",
                [f"Model {self._judge.model_name} failed ({type(exc).__name__}); "
                 "rule-based fallback used instead."],
                0,
                False,
            )
        j: RateJudgment = result.value  # type: ignore[assignment]
        return (j.direction, j.confidence, j.summary, result.model, j.caveats,
                result.attempts, result.cached)
