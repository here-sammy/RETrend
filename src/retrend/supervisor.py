"""Supervisor v1: run workers, check their reports, combine them, explain the result.

Principles (from the plan):
- The combination is plain code with transparent weights, not chosen by a model, so the
  evaluation harness can tune them later.
- A critic step inspects reports before synthesis (missing, stale, approximate, conflicting)
  and decides whether to retry a failed worker.
- A missing worker degrades the outlook (and says so) instead of crashing the run.
- A model may write the narrative, but only from the finished numbers.

The steps are plain functions over a state dict, so the same code runs in the simple
sequential runner below or as nodes of the LangGraph graph in supervisor_graph.py.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from retrend.llm import JsonJudge
from retrend.location import Location
from retrend.logging_utils import log_event
from retrend.outlook import Contribution, Finding, Outlook
from retrend.schema import WorkerReport
from retrend.store import RawStore
from retrend.sources.fred_metro import NoMetroSeries
from retrend.workers.base import BaseWorker, NoDataAsOf

# Transparent starting weights. The evaluation harness (Phase 4) is what should change them.
# Housing and supply are equal: the backtest gave no basis to prefer one (supply timed the
# market better, housing ranked cities better). Mortgage rates are weight 0: in the first real
# backtest the rule had the wrong sign (-3.1 pt spread out of sample), so the worker still
# runs and is shown as context, but it does not vote until it is redesigned.
DEFAULT_WEIGHTS = {"housing_prices": 0.5, "supply": 0.5, "mortgage_rates": 0.0,
                   "schools": 0.0, "jobs": 0.0, "permits": 0.0, "news": 0.0}  # extra signals: context only until a backtest says otherwise
# The weights used before the first backtest; the harness keeps a variant that uses them.
LEGACY_WEIGHTS = {"housing_prices": 0.40, "supply": 0.35, "mortgage_rates": 0.25}
# Days a report's data may be old before its weight decays.
FRESH_DAYS = {"housing_prices": 75, "supply": 75, "mortgage_rates": 14, "schools": 900,
              "jobs": 150, "permits": 150, "news": 45}
# Signals that are national by nature are not penalized for "approximate" geography.
NATIONAL = {"mortgage_rates"}
APPROX_GEOGRAPHY_FACTOR = 0.8
FLAT_BAND = 0.15  # |score| below this reads as flat
CONFLICT_MIN_CONFIDENCE = 0.4
MAX_ATTEMPTS = 2
NON_RETRYABLE = (NoDataAsOf, NoMetroSeries, ValueError, KeyError)  # retrying cannot change these
DIRECTION_SIGN = {"up": 1.0, "flat": 0.0, "down": -1.0}

NARRATIVE_SYSTEM = """\
You write the explanation for a housing-market outlook produced by other components. \
Input: the finished outlook as JSON. Write 2 to 4 plain sentences: the overall direction, \
the main drivers, the main dissent or data problem (if any), and how uncertain it is. \
Use only facts and numbers present in the input. Do not add predictions, causes, or \
outside knowledge. Do not name the place.

Reply with ONLY a JSON object, no markdown fences:
{"explanation": "..."}
"""


class NarrativeJudgment(BaseModel):
    model_config = ConfigDict(extra="ignore")
    explanation: str = Field(min_length=1, max_length=1200)


def freshness_factor(worker: str, age_days: int) -> float:
    fresh = FRESH_DAYS.get(worker, 60)
    if age_days <= fresh:
        return 1.0
    return round(max(0.25, fresh / age_days), 3)


# -- state -----------------------------------------------------------------------------


def new_state(location: Location, as_of: datetime | None, ref: date) -> dict[str, Any]:
    return {
        "location": location, "as_of": as_of, "ref": ref,
        "reports": {},      # worker -> WorkerReport
        "errors": {},       # worker -> "ErrorType: message"
        "retryable": {},    # worker -> bool
        "attempts": {},     # worker -> int
        "findings": [], "outlook": None,
    }


class Supervisor:
    def __init__(
        self,
        store: RawStore,
        workers: dict[str, BaseWorker],
        *,
        logger: logging.Logger | None = None,
        weights: dict[str, float] | None = None,
        narrator: JsonJudge | None = None,
        max_attempts: int = MAX_ATTEMPTS,
        expected: list[str] | None = None,
    ) -> None:
        self.store, self.workers, self.logger = store, workers, logger
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.narrator = narrator
        self.max_attempts = max_attempts
        # Workers the outlook is supposed to rest on; absent ones are reported as missing.
        self.expected = expected or list(workers)

    # -- nodes -------------------------------------------------------------------------

    def run_workers(self, state: dict) -> dict:
        """Run every worker that has no report yet and has attempts left."""
        for name in self.expected:
            if name in state["reports"] or state["attempts"].get(name, 0) >= self.max_attempts:
                continue
            if name in state["errors"] and not state["retryable"].get(name, False):
                continue
            worker = self.workers.get(name)
            if worker is None:
                state["errors"][name] = "no worker configured"
                state["retryable"][name] = False
                continue
            state["attempts"][name] = state["attempts"].get(name, 0) + 1
            try:
                state["reports"][name] = worker.run(state["location"], as_of=state["as_of"])
                state["errors"].pop(name, None)
            except Exception as exc:  # one broken worker must not stop the run
                state["errors"][name] = f"{type(exc).__name__}: {str(exc)[:200]}"
                state["retryable"][name] = not isinstance(exc, NON_RETRYABLE) and not (
                    "NoRegionMatch" in type(exc).__name__)
                log_event(self.logger, "supervisor_worker_failed", worker=name,
                          attempt=state["attempts"][name], error=state["errors"][name])
        return state

    def needs_retry(self, state: dict) -> bool:
        return any(
            n in state["errors"] and state["retryable"].get(n)
            and state["attempts"].get(n, 0) < self.max_attempts
            for n in self.expected if n not in state["reports"]
        )

    def critic(self, state: dict) -> dict:
        """Inspect reports and record findings. Pure code: no model."""
        findings: list[Finding] = []
        ref = state["ref"]
        for name in self.expected:
            if name not in state["reports"]:
                findings.append(Finding(kind="missing_worker", worker=name,
                                        detail=state["errors"].get(name, "did not run")))
                continue
            r: WorkerReport = state["reports"][name]
            if state["attempts"].get(name, 1) > 1:
                findings.append(Finding(kind="retried", worker=name,
                                        detail=f"succeeded on attempt {state['attempts'][name]}"))
            age = (ref - r.data_as_of).days
            if freshness_factor(name, age) < 1.0:
                findings.append(Finding(kind="stale_data", worker=name,
                                        detail=f"data is {age} days old; weight reduced"))
            if r.geography_match == "approximate" and name not in NATIONAL:
                findings.append(Finding(kind="approximate_geography", worker=name,
                                        detail="used a broader region than the city; weight "
                                               f"x{APPROX_GEOGRAPHY_FACTOR}"))
            if (r.judged_by or "").startswith("rule:"):
                findings.append(Finding(kind="rule_only", worker=name,
                                        detail="direction came from the rule baseline, not a model"))
            if r.key_metrics.get("point_in_time") is False:
                findings.append(Finding(kind="not_point_in_time", worker=name,
                                        detail="history may have been revised since the as-of date"))
        strong = [(n, r) for n, r in state["reports"].items()
                  if r.direction != "flat" and r.confidence >= CONFLICT_MIN_CONFIDENCE
                  and self.weights.get(n, 0.0) > 0]
        ups = [n for n, r in strong if r.direction == "up"]
        downs = [n for n, r in strong if r.direction == "down"]
        if ups and downs:
            findings.append(Finding(kind="conflict", detail=f"up: {ups}; down: {downs}"))
        state["findings"] = findings
        log_event(self.logger, "supervisor_critic", findings=[f.model_dump() for f in findings])
        return state

    def synthesize(self, state: dict) -> dict:
        """score = confidence-weighted average of the workers' directions (-1 down .. +1 up):

            score = sum(w * freshness * geography * confidence * sign) / sum(w * freshness * geography * confidence)

        so the score measures how strongly the signals agree on a direction, independent of how
        confident the workers are in absolute terms. Confidence is computed separately below.
        A flat report is a vote for zero, so it pulls the score toward flat."""
        contributions: list[Contribution] = []
        denom = 0.0        # sum of w * freshness * geography * confidence
        numer = 0.0
        confs = []
        for name, r in state["reports"].items():
            w = self.weights.get(name, 0.0)
            f = freshness_factor(name, (state["ref"] - r.data_as_of).days)
            g = 1.0 if (r.geography_match == "exact" or name in NATIONAL) else APPROX_GEOGRAPHY_FACTOR
            eff = round(r.confidence * f * g, 4)
            denom += w * f * g * r.confidence
            numer += w * f * g * r.confidence * DIRECTION_SIGN[r.direction]
            confs.append((w, eff))
            contributions.append(Contribution(
                worker=name, direction=r.direction, reported_confidence=r.confidence,
                effective_confidence=eff, base_weight=w, freshness_factor=f, geography_factor=g,
                contribution=0.0, data_as_of=r.data_as_of, summary=r.summary, sources=r.sources))

        score = round(numer / denom, 3) if denom > 0 else 0.0
        direction = "up" if score > FLAT_BAND else "down" if score < -FLAT_BAND else "flat"
        # Contribution = signed share of the score (they sum to the score).
        for c in contributions:
            c.contribution = round(
                c.base_weight * c.freshness_factor * c.geography_factor
                * c.reported_confidence * DIRECTION_SIGN[c.direction] / denom, 3) if denom else 0.0

        voting = [c for c in contributions if c.base_weight > 0]
        context = [c for c in contributions if c.base_weight <= 0]
        sign = DIRECTION_SIGN[direction]
        voting.sort(key=lambda c: abs(c.contribution), reverse=True)
        if sign == 0.0:
            drivers, dissent = voting, []
        else:
            drivers = [c for c in voting if c.contribution * sign > 0]
            dissent = [c for c in voting if c.contribution * sign < 0]

        # Confidence: weight-averaged effective confidence over ALL expected voting workers
        # (missing ones count as zero, so missing signals cost confidence), reduced by
        # disagreement. Zero-weight workers neither help nor hurt.
        expected_w = sum(self.weights.get(n, 0.0) for n in self.expected) or 1.0
        coverage_conf = sum(w * e for w, e in confs) / expected_w
        gross = sum(abs(c.contribution) for c in voting)
        agreement = abs(sum(c.contribution for c in voting)) / gross if gross else 0.0
        confidence = round(max(0.0, min(1.0, coverage_conf * (0.5 + 0.5 * agreement))), 3)

        missing = {n: state["errors"].get(n, "did not run")
                   for n in self.expected if n not in state["reports"]}
        n_voting_expected = len([n for n in self.expected if self.weights.get(n, 0.0) > 0])
        n_voting_used = len([n for n in state["reports"] if self.weights.get(n, 0.0) > 0])
        outlook = Outlook(
            location=state["location"], direction=direction, score=score, confidence=confidence,
            uncertainty=self._uncertainty(confidence, n_voting_used, n_voting_expected,
                                          bool(dissent), state["findings"]),
            drivers=drivers, dissent=dissent, context=context, missing_workers=missing,
            findings=state["findings"], narrative="", weights=self.weights,
            as_of=state["as_of"].date() if state["as_of"] else None,
            generated_at=self.store.clock())
        state["outlook"] = outlook
        return state

    @staticmethod
    def _uncertainty(confidence, n_used, n_expected, has_dissent, findings) -> str:
        level = "low" if confidence < 0.3 else "moderate" if confidence < 0.6 else "fairly high"
        parts = [f"Confidence is {level} ({confidence:.2f}, an uncalibrated estimate)",
                 f"based on {n_used} of {n_expected} signals"]
        if has_dissent:
            parts.append("with at least one signal pointing the other way")
        kinds = {f.kind for f in findings}
        if "stale_data" in kinds:
            parts.append("some data is stale")
        if "approximate_geography" in kinds:
            parts.append("some signals use a broader region than the city")
        return ", ".join(parts) + "."

    def narrate(self, state: dict) -> dict:
        o: Outlook = state["outlook"]
        fallback = self._template_narrative(o)
        if self.narrator is None:
            o.narrative, o.narrated_by = fallback, "template"
            return state
        payload = {
            "direction": o.direction, "score": o.score, "confidence": o.confidence,
            "uncertainty": o.uncertainty,
            "drivers": [{"signal": c.worker, "direction": c.direction, "summary": c.summary,
                         "contribution": c.contribution} for c in o.drivers],
            "dissent": [{"signal": c.worker, "direction": c.direction, "summary": c.summary,
                         "contribution": c.contribution} for c in o.dissent],
            "context_only_signals (not part of the vote)": [
                {"signal": c.worker, "direction": c.direction, "summary": c.summary} for c in o.context],
            "missing_signals": list(o.missing_workers),
            "data_problems": [f.detail for f in o.findings if f.kind != "retried"],
        }
        try:
            result = self.narrator.judge(NARRATIVE_SYSTEM, json.dumps(payload, indent=2),
                                         NarrativeJudgment)
            o.narrative, o.narrated_by = result.value.explanation, result.model  # type: ignore[union-attr]
        except Exception as exc:
            log_event(self.logger, "narrate_failed", error_type=type(exc).__name__,
                      error=str(exc)[:200])
            o.narrative, o.narrated_by = fallback, "template (model failed)"
        return state

    @staticmethod
    def _template_narrative(o: Outlook) -> str:
        names = lambda cs: ", ".join(c.worker for c in cs) or "none"  # noqa: E731
        if o.direction == "flat":
            text = (f"Outlook for the {o.horizon}: flat (score {o.score:+.2f}); no clear net "
                    f"pressure. Signals considered: {names(o.drivers)}.")
        else:
            text = (f"Outlook for the {o.horizon}: {o.direction} (score {o.score:+.2f}). "
                    f"Main drivers: {names(o.drivers)}. Dissenting signals: {names(o.dissent)}.")
        if o.missing_workers:
            text += f" Unavailable: {', '.join(o.missing_workers)}."
        if o.context:
            text += f" Shown for context only (not voting): {names(o.context)}."
        return text + " " + o.uncertainty

    def combine(self, location: Location, reports: dict[str, WorkerReport],
                errors: dict[str, str] | None, as_of: datetime | None, ref: date) -> Outlook:
        """Critic + synthesis over reports already in hand (no workers run, no narrative).
        Used by the evaluation harness to re-combine the same reports under different
        weights or with a worker left out."""
        state = new_state(location, as_of, ref)
        state["reports"] = {n: r for n, r in reports.items() if n in self.expected}
        state["errors"] = dict(errors or {})
        state["attempts"] = {n: 1 for n in state["reports"]}
        return self.synthesize(self.critic(state))["outlook"]

    # -- the sequential runner -----------------------------------------------------------

    def run(self, location: Location, *, as_of: datetime | None = None) -> Outlook:
        ref = as_of.astimezone(timezone.utc).date() if as_of else self.store.clock().date()
        state = new_state(location, as_of, ref)
        log_event(self.logger, "supervisor_start", location=location.key, as_of=as_of,
                  workers=self.expected)
        state = self.run_workers(state)
        while self.needs_retry(state):
            log_event(self.logger, "supervisor_retry",
                      workers=[n for n in state["errors"] if state["retryable"].get(n)])
            state = self.run_workers(state)
        state = self.synthesize(self.critic(state))
        state = self.narrate(state)
        o: Outlook = state["outlook"]
        log_event(self.logger, "supervisor_done", location=location.key, direction=o.direction,
                  score=o.score, confidence=o.confidence, missing=list(o.missing_workers),
                  drivers=[c.worker for c in o.drivers], dissent=[c.worker for c in o.dissent])
        return o
