"""Ground truth and the plain-trend baseline, both from the Zillow home value series."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Callable

from retrend.location import Location
from retrend.workers.timeseries import Point, by_month, pct_change, ym_key


class ZhviTruth:
    """Realized and trailing changes in ZHVI, using the same publication lag as the worker
    so the 'origin' month is the latest one the worker could have seen."""

    def __init__(self, series_fn: Callable[[Location], list[Point]], lag_days: int = 20) -> None:
        self._series_fn = series_fn
        self._lag = lag_days
        self._cache: dict[str, list[Point]] = {}

    def series(self, loc: Location) -> list[Point]:
        if loc.key not in self._cache:
            self._cache[loc.key] = sorted(self._series_fn(loc))
        return self._cache[loc.key]

    def origin(self, loc: Location, as_of: date) -> Point | None:
        usable = [p for p in self.series(loc) if p.day + timedelta(days=self._lag) <= as_of]
        return usable[-1] if usable else None

    def trailing_pct(self, loc: Location, origin: Point, months: int = 12) -> float | None:
        old = by_month(self.series(loc)).get(ym_key(origin.day) - months)
        return pct_change(origin.value, old)

    def forward_pct(self, loc: Location, origin: Point, horizon_months: int) -> float | None:
        """Realized change over the horizon, or None if that month is not in the data yet."""
        new = by_month(self.series(loc)).get(ym_key(origin.day) + horizon_months)
        return pct_change(new, origin.value)
