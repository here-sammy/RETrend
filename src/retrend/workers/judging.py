"""Shared pieces for workers whose judgment comes from a model with a rule-based fallback."""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from retrend.llm import JsonJudge
from retrend.logging_utils import log_event

BASELINE_CONFIDENCE = 0.3  # placeholder: rules have no calibrated confidence


class Judgment(BaseModel):
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


def judge_with_fallback(
    judge: JsonJudge | None,
    *,
    system: str,
    user: str,
    baseline: tuple[str, str],
    baseline_name: str,
    fallback: bool,
    logger: logging.Logger | None,
    worker: str,
):
    """-> (direction, confidence, summary, judged_by, caveats, attempts, cached).

    ``baseline`` is (direction, summary) from the rule."""
    base = (baseline[0], BASELINE_CONFIDENCE, baseline[1])
    if judge is None:
        return (*base, f"rule:{baseline_name}", ["Rule-based baseline only; no model used."], 0, False)
    try:
        result = judge.judge(system, user, Judgment)
    except Exception as exc:
        log_event(logger, "judge_failed", worker=worker, model=judge.model_name,
                  error_type=type(exc).__name__, error=str(exc)[:300])
        if not fallback:
            raise
        return (
            *base,
            f"rule:{baseline_name} (fallback from {judge.model_name})",
            [f"Model {judge.model_name} failed ({type(exc).__name__}); "
             "rule-based fallback used instead."],
            0,
            False,
        )
    j: Judgment = result.value  # type: ignore[assignment]
    return j.direction, j.confidence, j.summary, result.model, j.caveats, result.attempts, result.cached


def load_payload(store, source, key, fetcher, as_of, data_as_of, max_age):
    """Fetch through the cache. For an as-of run, prefer a stored snapshot taken at or before
    ``as_of``; if none exists, use the live pull and report ``point_in_time=False`` because
    Zillow and Redfin revise history and offer no vintage API.

    -> (record, point_in_time)"""
    if as_of is None:
        return store.fetch_or_cache(source, key, fetcher, max_age=max_age, data_as_of=data_as_of), True
    snap = store.get_as_of(source, key, as_of)
    if snap is not None:
        return snap, True
    return store.fetch_or_cache(source, key, fetcher, max_age=max_age, data_as_of=data_as_of), False
