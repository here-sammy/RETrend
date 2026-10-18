"""Timestamped raw-data cache.

Every pull from an external source is stored with the time it was fetched. That gives
reproducibility and makes backtests possible: ``get_as_of`` returns what we knew at a
past moment, so a worker run "as of 2022" sees only data available then.

Backed by SQLite (standard library) for now. Callers use only this class, so swapping
in DuckDB later changes this one file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from retrend.logging_utils import log_event

_TS_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_pulls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    query_key TEXT NOT NULL,
    fetched_at TEXT NOT NULL,       -- first time this exact content was seen
    last_checked_at TEXT NOT NULL,  -- most recent time the source returned it
    data_as_of TEXT,                -- period the data describes, if known (ISO date)
    content_hash TEXT NOT NULL,
    content_type TEXT NOT NULL,
    payload BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_raw_pulls_lookup
    ON raw_pulls (source, query_key, fetched_at);

CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,   -- only replies that passed validation are stored
    UNIQUE (model, prompt_hash)
);
"""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError("datetimes must be timezone-aware")
    return dt.astimezone(timezone.utc).strftime(_TS_FORMAT)


def _from_iso(s: str) -> datetime:
    return datetime.strptime(s, _TS_FORMAT).replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class PullRecord:
    id: int
    source: str
    query_key: str
    fetched_at: datetime
    last_checked_at: datetime
    data_as_of: date | None
    content_hash: str
    content_type: str
    payload: bytes

    def text(self) -> str:
        return self.payload.decode("utf-8")

    def json(self) -> Any:
        return json.loads(self.payload)


def _normalize(payload: bytes | str | dict | list, content_type: str | None) -> tuple[bytes, str]:
    if isinstance(payload, bytes):
        return payload, content_type or "application/octet-stream"
    if isinstance(payload, str):
        return payload.encode("utf-8"), content_type or "text/plain"
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        content_type or "application/json",
    )


class RawStore:
    """Cache of raw source pulls, keyed by (source, query_key) and fetch time."""

    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        clock: Callable[[], datetime] = utcnow,
        logger: logging.Logger | None = None,
    ) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.executescript(_SCHEMA)
        self.clock = clock
        self._logger = logger

    # -- writing -----------------------------------------------------------------

    def put(
        self,
        source: str,
        query_key: str,
        payload: bytes | str | dict | list,
        *,
        data_as_of: date | None = None,
        content_type: str | None = None,
        fetched_at: datetime | None = None,
    ) -> PullRecord:
        """Store a pull. If the content is identical to what we already held at that
        time, no new row is added; the existing row's ``last_checked_at`` is refreshed."""
        data, ctype = _normalize(payload, content_type)
        digest = hashlib.sha256(data).hexdigest()
        when = fetched_at or self.clock()
        when_iso = _to_iso(when)

        prev = self.get_as_of(source, query_key, when)
        if prev is not None and prev.content_hash == digest:
            self._conn.execute(
                "UPDATE raw_pulls SET last_checked_at = ? WHERE id = ? AND last_checked_at < ?",
                (when_iso, prev.id, when_iso),
            )
            self._conn.commit()
            return self._get_by_id(prev.id)

        cur = self._conn.execute(
            "INSERT INTO raw_pulls (source, query_key, fetched_at, last_checked_at, data_as_of,"
            " content_hash, content_type, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                source,
                query_key,
                when_iso,
                when_iso,
                data_as_of.isoformat() if data_as_of else None,
                digest,
                ctype,
                data,
            ),
        )
        self._conn.commit()
        return self._get_by_id(cur.lastrowid)

    def fetch_or_cache(
        self,
        source: str,
        query_key: str,
        fetcher: Callable[[], bytes | str | dict | list],
        *,
        max_age: timedelta,
        data_as_of: date | None | Callable[[Any], date | None] = None,
        content_type: str | None = None,
    ) -> PullRecord:
        """Return the cached pull if it was checked within ``max_age``, else call
        ``fetcher`` and store the result. ``data_as_of`` may be a function of the fetched
        payload, for sources whose period is only known after fetching."""
        cached = self.get_latest(source, query_key, max_age=max_age)
        if cached is not None:
            log_event(self._logger, "cache_hit", source=source, query_key=query_key,
                      fetched_at=cached.fetched_at)
            return cached
        log_event(self._logger, "cache_miss", source=source, query_key=query_key)
        fetched = fetcher()
        period = data_as_of(fetched) if callable(data_as_of) else data_as_of
        record = self.put(
            source, query_key, fetched, data_as_of=period, content_type=content_type
        )
        log_event(self._logger, "pull_stored", source=source, query_key=query_key,
                  record_id=record.id, content_hash=record.content_hash[:12])
        return record

    # -- reading -----------------------------------------------------------------

    def get_latest(
        self, source: str, query_key: str, *, max_age: timedelta | None = None
    ) -> PullRecord | None:
        row = self._conn.execute(
            "SELECT id FROM raw_pulls WHERE source = ? AND query_key = ?"
            " ORDER BY fetched_at DESC, id DESC LIMIT 1",
            (source, query_key),
        ).fetchone()
        if row is None:
            return None
        record = self._get_by_id(row[0])
        if max_age is not None and self.clock() - record.last_checked_at > max_age:
            return None
        return record

    def get_as_of(self, source: str, query_key: str, as_of: datetime) -> PullRecord | None:
        """What we knew at ``as_of``: the newest pull fetched at or before that time."""
        row = self._conn.execute(
            "SELECT id FROM raw_pulls WHERE source = ? AND query_key = ? AND fetched_at <= ?"
            " ORDER BY fetched_at DESC, id DESC LIMIT 1",
            (source, query_key, _to_iso(as_of)),
        ).fetchone()
        return self._get_by_id(row[0]) if row else None

    def keys(self, source: str) -> list[str]:
        """Every query key stored for ``source``, oldest first."""
        rows = self._conn.execute(
            "SELECT query_key FROM raw_pulls WHERE source = ? GROUP BY query_key"
            " ORDER BY MIN(id)", (source,)).fetchall()
        return [r[0] for r in rows]

    def history(self, source: str, query_key: str) -> list[PullRecord]:
        rows = self._conn.execute(
            "SELECT id FROM raw_pulls WHERE source = ? AND query_key = ?"
            " ORDER BY fetched_at ASC, id ASC",
            (source, query_key),
        ).fetchall()
        return [self._get_by_id(r[0]) for r in rows]

    # -- LLM call cache ----------------------------------------------------------

    def get_llm_response(self, model: str, prompt: str) -> str | None:
        """A previously stored, validated reply for this exact model and prompt."""
        row = self._conn.execute(
            "SELECT response FROM llm_calls WHERE model = ? AND prompt_hash = ?",
            (model, hashlib.sha256(prompt.encode("utf-8")).hexdigest()),
        ).fetchone()
        return row[0] if row else None

    def put_llm_response(self, model: str, prompt: str, response: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO llm_calls (model, prompt_hash, created_at, prompt, response)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                model,
                hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                _to_iso(self.clock()),
                prompt,
                response,
            ),
        )
        self._conn.commit()

    # -- plumbing ----------------------------------------------------------------

    def _get_by_id(self, record_id: int) -> PullRecord:
        r = self._conn.execute(
            "SELECT id, source, query_key, fetched_at, last_checked_at, data_as_of,"
            " content_hash, content_type, payload FROM raw_pulls WHERE id = ?",
            (record_id,),
        ).fetchone()
        return PullRecord(
            id=r[0],
            source=r[1],
            query_key=r[2],
            fetched_at=_from_iso(r[3]),
            last_checked_at=_from_iso(r[4]),
            data_as_of=date.fromisoformat(r[5]) if r[5] else None,
            content_hash=r[6],
            content_type=r[7],
            payload=bytes(r[8]),
        )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "RawStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
