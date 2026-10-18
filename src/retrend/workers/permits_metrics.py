"""Building-permit metrics (metro, monthly). Plain code.

The SIGN of permits as a price signal is not settled: more permits can mean builders see strong demand
(prices firm) or that new supply is coming (prices soften). So the baseline rule is chosen by setting,
and the default ("context") reports the numbers without taking a side.
"""

from __future__ import annotations

from datetime import date

from retrend.fred import Observation
from retrend.workers.monthly import at_months_back, has_gap

RULES = ("context", "demand", "supply")
THRESHOLD_PCT = 10.0  # placeholder, untuned

PROMPT_METRIC_KEYS = ("permits_12m_total", "permits_12m_yoy_pct", "permits_prev_12m_yoy_pct",
                      "permits_acceleration_pp", "permits_3m_vs_year_ago_pct")


def _pct(new, old):
    return None if new is None or not old else round((new / old - 1.0) * 100.0, 1)


def baseline_direction(yoy: float | None, rule: str) -> tuple[str, str]:
    if rule not in RULES:
        raise ValueError(f"unknown permits rule {rule!r}; use one of {RULES}")
    if yoy is None:
        return "flat", "not enough history"
    if rule == "context":
        return "flat", f"permits {yoy:+.1f}% over the last year; shown as context, no side taken"
    up_when_rising = rule == "demand"
    if yoy >= THRESHOLD_PCT:
        return ("up" if up_when_rising else "down"), \
            f"permits {yoy:+.1f}% over the last year, read as {'builder demand' if up_when_rising else 'incoming supply'}"
    if yoy <= -THRESHOLD_PCT:
        return ("down" if up_when_rising else "up"), \
            f"permits {yoy:+.1f}% over the last year, read as {'builder pullback' if up_when_rising else 'less incoming supply'}"
    return "flat", f"permits {yoy:+.1f}% over the last year: little change"


def compute_metrics(obs: list[Observation], ref: date, rule: str = "context") -> dict | None:
    """None when there is not enough unbroken monthly history (needs 24 months; 36 for the prior-year change)."""
    if has_gap(obs, 24):
        return None
    vals = [o.value for o in obs]
    ttm = sum(vals[-12:])
    prev = sum(vals[-24:-12])
    yoy = _pct(ttm, prev)
    prev_yoy = _pct(prev, sum(vals[-36:-24])) if not has_gap(obs, 36) else None
    acc = None if yoy is None or prev_yoy is None else round(yoy - prev_yoy, 1)
    r3 = _pct(sum(vals[-3:]), sum(vals[-15:-12]))
    direction, why = baseline_direction(yoy, rule)
    return {
        "permits_12m_total": round(ttm),
        "permits_12m_yoy_pct": yoy,
        "permits_prev_12m_yoy_pct": prev_yoy,
        "permits_acceleration_pp": acc,
        "permits_3m_vs_year_ago_pct": r3,
        "latest_obs_date": obs[-1].day.isoformat(),
        "days_since_latest_obs": (ref - obs[-1].day).days,
        "baseline_direction": direction,
        "baseline_rule": why,
        "permits_rule": rule,
    }
