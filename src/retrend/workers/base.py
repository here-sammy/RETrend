"""Base class every worker extends.

A worker is a prompt/tool bundle that turns a ``Location`` into a schema-valid
``WorkerReport``. The base class handles validation and logging so individual workers
only implement ``_run``.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime

from retrend.location import Location
from retrend.logging_utils import log_event
from retrend.schema import WorkerReport
from retrend.store import RawStore


class NoDataAsOf(RuntimeError):
    """Raised when an as-of (backtest) run has no data fetched at or before that time."""


class BaseWorker(ABC):
    name: str

    def __init__(self, store: RawStore, logger: logging.Logger | None = None) -> None:
        self.store = store
        self.logger = logger

    def run(self, location: Location, *, as_of: datetime | None = None) -> WorkerReport:
        """Produce a validated report for ``location``.

        If ``as_of`` is given (backtesting), the worker must use only information that
        existed at that time. For sources that return a full history, fetch through the
        cache as usual and drop observations dated after ``as_of``. Raise ``NoDataAsOf``
        when nothing qualifying remains.
        """
        log_event(self.logger, "worker_start", worker=self.name, location=location.key,
                  as_of=as_of)
        started = time.perf_counter()
        try:
            report = self._run(location, as_of)
            if not isinstance(report, WorkerReport):
                report = WorkerReport.model_validate(report)
            if report.worker != self.name:
                raise ValueError(f"report.worker {report.worker!r} != worker name {self.name!r}")
            if report.location != location:
                raise ValueError("report.location does not match the requested location")
        except Exception as exc:
            log_event(self.logger, "worker_error", worker=self.name, location=location.key,
                      error_type=type(exc).__name__, error=str(exc))
            raise
        log_event(
            self.logger,
            "worker_done",
            worker=self.name,
            location=location.key,
            direction=report.direction,
            confidence=report.confidence,
            data_as_of=report.data_as_of,
            geography_match=report.geography_match,
            n_sources=len(report.sources),
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return report

    @abstractmethod
    def _run(self, location: Location, as_of: datetime | None) -> WorkerReport:
        """Fetch data (via ``self.store``), compute the signal, return a report."""
