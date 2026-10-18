"""Shared helpers for workers built on monthly FRED series (metro jobs and building permits)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from retrend.fred import Observation, parse_observations
from retrend.workers.base import NoDataAsOf

LIVE_MAX_AGE = timedelta(days=1)
FOREVER = timedelta(days=36500)
PUBLICATION_LAG_DAYS = 80  # assumed: a month's figure is public about 6 weeks after the month ends
HISTORY_YEARS = 8


def month_index(d: date) -> int:
    return d.year * 12 + d.month - 1


def at_months_back(obs: list[Observation], n: int) -> float | None:
    """Value n calendar months before the latest reading, or None if that month is missing."""
    if not obs:
        return None
    want = month_index(obs[-1].day) - n
    for o in reversed(obs):
        if month_index(o.day) == want:
            return o.value
        if month_index(o.day) < want:
            break
    return None


def has_gap(obs: list[Observation], months: int) -> bool:
    """True if the last ``months`` months are not all present (so sums over them would be wrong)."""
    if len(obs) < months:
        return True
    tail = obs[-months:]
    return month_index(tail[-1].day) - month_index(tail[0].day) != months - 1


def run_date(store, as_of: datetime | None) -> tuple[date, date | None]:
    if as_of is not None and as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    ref = as_of.astimezone(timezone.utc).date() if as_of else store.clock().date()
    return ref, (ref if as_of is not None else None)


def pull_monthly(store, fred_source, series_id: str, ref: date, as_of_date: date | None):
    """Fetch through the cache -> (record, observations usable on ``ref``). As-of pulls are cached
    forever (a past vintage never changes) and drop readings not yet public on that date."""
    start = date(ref.year - HISTORY_YEARS, 1, 1)
    key = f"{series_id}|asof={as_of_date}" if as_of_date else f"{series_id}|live"

    def latest(payload):
        days = [r["date"] for r in payload.get("observations", []) if r.get("value") is not None]
        return date.fromisoformat(max(days)) if days else None

    record = store.fetch_or_cache(
        "fred", key,
        lambda: fred_source(series_id, observation_start=start, observation_end=as_of_date, as_of=as_of_date),
        max_age=FOREVER if as_of_date else LIVE_MAX_AGE, data_as_of=latest)
    obs = parse_observations(record.json())
    if as_of_date is not None:
        obs = [o for o in obs if o.day + timedelta(days=PUBLICATION_LAG_DAYS) <= ref]
    else:
        obs = [o for o in obs if o.day <= ref]
    if not obs:
        raise NoDataAsOf(f"no {series_id} readings available as of {ref}")
    return record, obs
