"""News-coverage metrics (tone and volume of housing coverage for a city). Plain code."""

from __future__ import annotations

from datetime import date, timedelta
from statistics import mean

RECENT_DAYS = 30
BASE_DAYS = 60
TONE_MOVE = 0.5      # placeholder, untuned: tone points
MIN_DAYS = 10        # days of data needed in each window

PROMPT_METRIC_KEYS = ("tone_recent", "tone_base", "tone_change", "volume_recent", "volume_base",
                      "volume_change_pct", "days_with_coverage")


def _window(points: list[dict], end: date, days: int, skip: int = 0) -> list[float]:
    hi = end - timedelta(days=skip)
    lo = hi - timedelta(days=days)
    return [p["value"] for p in points if lo < date.fromisoformat(p["date"]) <= hi]


def baseline_direction(change: float | None, move: float = TONE_MOVE) -> tuple[str, str]:
    if change is None:
        return "flat", "not enough coverage to compare"
    if change >= move:
        return "up", f"coverage tone improved {change:+.2f} points versus the earlier two months"
    if change <= -move:
        return "down", f"coverage tone worsened {change:+.2f} points versus the earlier two months"
    return "flat", f"coverage tone changed {change:+.2f} points: little change"


def compute_metrics(tone: list[dict], volume: list[dict], ref: date, move: float = TONE_MOVE) -> dict | None:
    """None when either window has fewer than MIN_DAYS days of tone readings."""
    if not tone:
        return None
    end = min(ref, date.fromisoformat(tone[-1]["date"]))
    t_recent = _window(tone, end, RECENT_DAYS)
    t_base = _window(tone, end, BASE_DAYS, skip=RECENT_DAYS)
    if len(t_recent) < MIN_DAYS or len(t_base) < MIN_DAYS:
        return None
    v_recent = _window(volume, end, RECENT_DAYS)
    v_base = _window(volume, end, BASE_DAYS, skip=RECENT_DAYS)
    vr = mean(v_recent) if v_recent else None
    vb = mean(v_base) if v_base else None
    change = round(mean(t_recent) - mean(t_base), 2)
    direction, why = baseline_direction(change, move)
    return {
        "tone_recent": round(mean(t_recent), 2), "tone_base": round(mean(t_base), 2), "tone_change": change,
        "volume_recent": None if vr is None else round(vr, 4), "volume_base": None if vb is None else round(vb, 4),
        "volume_change_pct": None if not vr or not vb else round((vr / vb - 1) * 100, 1),
        "days_with_coverage": len(t_recent) + len(t_base),
        "latest_obs_date": end.isoformat(), "days_since_latest_obs": (ref - end).days,
        "baseline_direction": direction, "baseline_rule": why,
    }
