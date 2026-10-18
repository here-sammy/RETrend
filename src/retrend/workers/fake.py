"""A synthetic worker used to exercise the Phase 0 plumbing.

It is not a real signal. It shows the pattern real workers follow: fetch through the
cache for live runs, read the store only for as-of (backtest) runs, return a report.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Callable

from retrend.location import Location
from retrend.schema import Source, WorkerReport
from retrend.store import RawStore
from retrend.workers.base import BaseWorker, NoDataAsOf

SOURCE = "fake_source"


def _default_fetcher() -> dict:
    return {"index": 100.0, "change_12m_pct": 4.2}


class FakeWorker(BaseWorker):
    name = "fake"

    def __init__(
        self,
        store: RawStore,
        logger: logging.Logger | None = None,
        *,
        fetcher: Callable[[], dict] = _default_fetcher,
        max_age: timedelta = timedelta(days=1),
    ) -> None:
        super().__init__(store, logger)
        self._fetcher = fetcher
        self._max_age = max_age

    def _run(self, location: Location, as_of: datetime | None) -> WorkerReport:
        if as_of is None:
            record = self.store.fetch_or_cache(
                SOURCE,
                location.key,
                self._fetcher,
                max_age=self._max_age,
                data_as_of=self.store.clock().date(),
            )
        else:
            record = self.store.get_as_of(SOURCE, location.key, as_of)
            if record is None:
                raise NoDataAsOf(f"no {SOURCE} data for {location.key} at or before {as_of}")

        data = record.json()
        change = data["change_12m_pct"]
        direction = "up" if change > 1 else "down" if change < -1 else "flat"
        return WorkerReport(
            worker=self.name,
            location=location,
            signal="fake",
            direction=direction,
            confidence=0.5,
            key_metrics={"index": data["index"], "change_12m_pct": change},
            data_as_of=record.data_as_of or record.fetched_at.date(),
            geography_match="exact",
            caveats=["Synthetic data from the Phase 0 fake worker; not a real signal."],
            sources=[Source(name=SOURCE, url=None, retrieved=record.fetched_at)],
        )
