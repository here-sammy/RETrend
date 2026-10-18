"""Local labor-market metrics (metro unemployment rate and payroll jobs). Plain code."""

from __future__ import annotations

from datetime import date

from retrend.fred import Observation
from retrend.workers.monthly import at_months_back, month_index

# placeholders, untuned
UNEMP_RISE_PP = 0.5
UNEMP_FALL_PP = -0.3
PAYROLL_STRONG_PCT = 0.5
PAYROLL_WEAK_PCT = -1.0

PROMPT_METRIC_KEYS = (
    "unemployment_rate", "unemployment_change_12m_pp", "unemployment_prev_change_12m_pp",
    "unemployment_5y_percentile", "payroll_yoy_pct", "payroll_prev_yoy_pct", "payroll_acceleration_pp",
)


def _pct(new, old):
    return None if new is None or not old else round((new / old - 1.0) * 100.0, 2)


def baseline_direction(u12: float | None, p: float | None) -> tuple[str, str]:
    """Falling unemployment with steady job growth suggests demand for housing; the reverse, the opposite."""
    if u12 is None and p is None:
        return "flat", "not enough history"
    strong = (u12 is not None and u12 <= UNEMP_FALL_PP and (p is None or p >= PAYROLL_STRONG_PCT)) or \
             (u12 is None and p is not None and p >= 2.0)
    weak = (u12 is not None and u12 >= UNEMP_RISE_PP and (p is None or p < 1.5)) or \
           (p is not None and p <= PAYROLL_WEAK_PCT)
    pu = "n/a" if u12 is None else f"{u12:+.1f} pp"
    pp = "n/a" if p is None else f"{p:+.1f}%"
    if strong and not weak:
        return "up", f"unemployment {pu} over 12 months and payrolls {pp}: a strengthening job market"
    if weak and not strong:
        return "down", f"unemployment {pu} over 12 months and payrolls {pp}: a weakening job market"
    return "flat", f"unemployment {pu} over 12 months and payrolls {pp}: no clear change"


def compute_metrics(unemp: list[Observation], payroll: list[Observation] | None, ref: date) -> dict:
    latest = unemp[-1]
    u_now = latest.value
    u_12 = at_months_back(unemp, 12)
    u_24 = at_months_back(unemp, 24)
    u12 = None if u_12 is None else round(u_now - u_12, 2)
    u_prev = None if u_12 is None or u_24 is None else round(u_12 - u_24, 2)
    window = [o.value for o in unemp if month_index(o.day) > month_index(latest.day) - 60]
    pct = round(sum(1 for v in window if v <= u_now) / len(window), 2) if len(window) >= 24 else None
    py = pp = acc = None
    pay_latest = None
    if payroll:
        pay_latest = payroll[-1]
        py = _pct(pay_latest.value, at_months_back(payroll, 12))
        p12 = at_months_back(payroll, 12)
        pp = _pct(p12, at_months_back(payroll, 24)) if p12 is not None else None
        acc = None if py is None or pp is None else round(py - pp, 2)
    direction, rule = baseline_direction(u12, py)
    return {
        "unemployment_rate": round(u_now, 2),
        "unemployment_change_12m_pp": u12,
        "unemployment_prev_change_12m_pp": u_prev,
        "unemployment_5y_percentile": pct,
        "payroll_yoy_pct": py,
        "payroll_prev_yoy_pct": pp,
        "payroll_acceleration_pp": acc,
        "latest_obs_date": latest.day.isoformat(),
        "payroll_latest_obs_date": pay_latest.day.isoformat() if pay_latest else None,
        "days_since_latest_obs": (ref - latest.day).days,
        "baseline_direction": direction,
        "baseline_rule": rule,
    }
