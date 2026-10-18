"""Shared test helpers."""

from datetime import datetime, timedelta, timezone

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self, now: datetime = T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs: int) -> None:
        self.now += timedelta(**kwargs)
