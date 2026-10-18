"""Enrollment trend metrics. Plain code; the model only judges these numbers."""

from __future__ import annotations

YOY_THRESHOLD_PCT = 1.0  # placeholder, untuned

PROMPT_METRIC_KEYS = (
    "enrollment_yoy_pct", "enrollment_prev_yoy_pct", "enrollment_acceleration_pp",
    "enrollment_3y_cagr_pct", "enrollment_5y_change_pct", "years_of_data",
)


def _pct(new: float | None, old: float | None) -> float | None:
    if new is None or not old:
        return None
    return round((new / old - 1.0) * 100.0, 1)


def baseline_direction(yoy: float | None, cagr3: float | None) -> tuple[str, str]:
    """Growing enrollment suggests household demand; shrinking suggests the reverse. Weak signal."""
    if yoy is None:
        return "flat", "insufficient enrollment history"
    t = YOY_THRESHOLD_PCT
    if yoy >= t and (cagr3 is None or cagr3 > 0):
        return "up", f"enrollment {yoy:+.1f}% last year" + (f", {cagr3:+.1f}%/yr over 3 years" if cagr3 is not None else "")
    if yoy <= -t and (cagr3 is None or cagr3 < 0):
        return "down", f"enrollment {yoy:+.1f}% last year" + (f", {cagr3:+.1f}%/yr over 3 years" if cagr3 is not None else "")
    return "flat", f"enrollment change {yoy:+.1f}% last year is small or mixed with the 3-year trend"


def compute_metrics(rows: list[dict]) -> dict:
    """``rows``: [{'year': 2022, 'students': 51000}, ...] oldest first."""
    by = {r["year"]: r["students"] for r in rows}
    last = max(by)
    yoy = _pct(by.get(last), by.get(last - 1))
    prev = _pct(by.get(last - 1), by.get(last - 2))
    cagr3 = None
    if by.get(last - 3):
        cagr3 = round(((by[last] / by[last - 3]) ** (1 / 3) - 1.0) * 100.0, 1)
    direction, rule = baseline_direction(yoy, cagr3)
    return {
        "latest_year": last, "latest_students": round(by[last]),
        "enrollment_yoy_pct": yoy, "enrollment_prev_yoy_pct": prev,
        "enrollment_acceleration_pp": None if yoy is None or prev is None else round(yoy - prev, 1),
        "enrollment_3y_cagr_pct": cagr3, "enrollment_5y_change_pct": _pct(by.get(last), by.get(last - 5)),
        "years_of_data": len(by),
        "baseline_direction": direction, "baseline_rule": rule,
    }
