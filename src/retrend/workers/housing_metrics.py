"""Home value / rent metrics from Zillow series. Plain code; the model never computes."""

from __future__ import annotations

from datetime import date

from retrend.workers.timeseries import Point, by_month, diff, months_ago, pct_change, ym_key

PROMPT_METRIC_KEYS = (
    "zhvi_yoy_pct", "zhvi_3m_annualized_pct", "zhvi_6m_annualized_pct",
    "zhvi_acceleration_pp", "zhvi_drawdown_from_peak_pct", "zori_yoy_pct",
    "rent_minus_value_growth_pp", "months_of_history",
)


def _annualize(new: float | None, old: float | None, months: int) -> float | None:
    """Annualize from the raw values (not an already-rounded percentage)."""
    if new is None or old is None or old <= 0:
        return None
    return round(((new / old) ** (12.0 / months) - 1) * 100.0, 1)


def compute_metrics(zhvi: list[Point], zori: list[Point] | None) -> dict:
    if not zhvi:
        raise ValueError("no home value observations")
    zhvi = sorted(zhvi)
    latest = zhvi[-1]
    s = by_month(zhvi)
    now = latest.value
    m3, m6, m12 = (months_ago(s, latest.day, n) for n in (3, 6, 12))
    yoy = pct_change(now, m12)
    ann3 = _annualize(now, m3, 3)
    ann6 = _annualize(now, m6, 6)
    window = [p.value for p in zhvi if ym_key(p.day) > ym_key(latest.day) - 60]
    peak = max(window)
    metrics: dict = {
        "zhvi_latest": round(now),
        "zhvi_yoy_pct": yoy,
        "zhvi_3m_annualized_pct": ann3,
        "zhvi_6m_annualized_pct": ann6,
        "zhvi_acceleration_pp": diff(ann3, yoy, 1),
        "zhvi_drawdown_from_peak_pct": round((now / peak - 1) * 100.0, 1),
        "months_of_history": len(zhvi),
        "zori_latest": None,
        "zori_yoy_pct": None,
        "rent_minus_value_growth_pp": None,
    }
    if zori:
        zs = sorted(zori)
        zl = zs[-1]
        zyoy = pct_change(zl.value, months_ago(by_month(zs), zl.day, 12))
        metrics["zori_latest"] = round(zl.value)
        metrics["zori_yoy_pct"] = zyoy
        metrics["rent_minus_value_growth_pp"] = diff(zyoy, yoy, 1)
    metrics["baseline_direction"], metrics["baseline_rule"] = baseline_direction(metrics)
    return metrics


def baseline_direction(m: dict) -> tuple[str, str]:
    """Momentum rule: both year-over-year and 3-month annualized growth must agree."""
    yoy, ann3 = m.get("zhvi_yoy_pct"), m.get("zhvi_3m_annualized_pct")
    if yoy is None or ann3 is None:
        return "flat", "insufficient history for the momentum rule"
    if yoy >= 1.0 and ann3 > 0:
        return "up", f"YoY {yoy}% and 3-month annualized {ann3}% are both positive"
    if yoy <= -1.0 and ann3 < 0:
        return "down", f"YoY {yoy}% and 3-month annualized {ann3}% are both negative"
    return "flat", f"YoY {yoy}% and 3-month annualized {ann3}% do not agree on a clear trend"


def last_day(points: list[Point]) -> date:
    return max(p.day for p in points)
