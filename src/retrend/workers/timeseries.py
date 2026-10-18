"""Small helpers for monthly series. Plain code, no I/O."""

from __future__ import annotations

from datetime import date
from typing import NamedTuple


class Point(NamedTuple):
    day: date
    value: float


def ym_key(day: date) -> int:
    """Month index: consecutive months differ by 1, whatever their lengths."""
    return day.year * 12 + day.month - 1


def by_month(points: list[Point]) -> dict[int, float]:
    return {ym_key(p.day): p.value for p in points}


def months_ago(series: dict[int, float], latest_day: date, n: int) -> float | None:
    """Value ``n`` calendar months before ``latest_day``, or None if that month is missing."""
    return series.get(ym_key(latest_day) - n)


def pct_change(new: float | None, old: float | None) -> float | None:
    if new is None or old is None or old == 0:
        return None
    return round((new / old - 1.0) * 100.0, 1)


def diff(new: float | None, old: float | None, digits: int = 2) -> float | None:
    if new is None or old is None:
        return None
    return round(new - old, digits)
