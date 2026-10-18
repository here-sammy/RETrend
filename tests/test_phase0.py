"""Phase 0 checks: schema validity, location key, cache + as-of replay, logging.

Run with:  PYTHONPATH=src python3 -m unittest discover -s tests -v
(or `pytest` once it is installed; pyproject sets the path).
"""

import json
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from pydantic import ValidationError

from retrend.location import Location, load_registry
from retrend.logging_utils import get_logger, log_event
from retrend.schema import Source, WorkerReport
from retrend.store import RawStore
from retrend.workers.base import BaseWorker, NoDataAsOf
from retrend.workers.fake import SOURCE, FakeWorker

ROOT = Path(__file__).resolve().parents[1]
T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
SEATTLE = Location(city="Seattle", state="WA", county_fips="53033", cbsa="42660")


class Clock:
    def __init__(self, now: datetime = T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs: int) -> None:
        self.now += timedelta(**kwargs)


def make_report(**overrides) -> WorkerReport:
    fields = dict(
        worker="fake",
        location=SEATTLE,
        signal="test",
        direction="up",
        confidence=0.7,
        key_metrics={"a": 1, "b": 2.5, "c": "x", "d": None},
        data_as_of=date(2026, 9, 30),
        geography_match="exact",
        caveats=[],
        sources=[Source(name="s", url="https://example.com", retrieved=T0)],
    )
    fields.update(overrides)
    return WorkerReport(**fields)


class LocationTests(unittest.TestCase):
    def test_registry_loads_test_cities(self):
        reg = load_registry(ROOT / "config" / "test_cities.yaml")
        self.assertEqual(set(reg), {"seattle-wa", "austin-tx", "phoenix-az"})
        self.assertEqual(reg["phoenix-az"].county_fips, "04013")  # leading zero preserved

    def test_key_slug(self):
        self.assertEqual(Location(city="St. Louis", state="mo").key, "st-louis-mo")

    def test_rejects_bad_values(self):
        with self.assertRaises(ValidationError):
            Location(city="Seattle", state="Washington")
        with self.assertRaises(ValidationError):
            Location(city="Seattle", state="WA", county_fips="123")
        with self.assertRaises(ValidationError):
            Location(city="Seattle", state="WA", zips=("9810",))
        with self.assertRaises(ValidationError):
            Location(city="  ", state="WA")


class SchemaTests(unittest.TestCase):
    def test_json_roundtrip(self):
        report = make_report()
        again = WorkerReport.model_validate_json(report.model_dump_json())
        self.assertEqual(report, again)

    def test_rejects_invalid_reports(self):
        with self.assertRaises(ValidationError):
            make_report(confidence=1.5)
        with self.assertRaises(ValidationError):
            make_report(direction="sideways")
        with self.assertRaises(ValidationError):
            make_report(geography_match="roughly")
        with self.assertRaises(ValidationError):
            make_report(sources=[])  # every report must cite evidence
        with self.assertRaises(ValidationError):
            make_report(extra_field="nope")
        with self.assertRaises(ValidationError):
            make_report(sources=[Source(name="s", retrieved=datetime(2026, 1, 1))])  # naive dt


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.store = RawStore(clock=self.clock)

    def tearDown(self):
        self.store.close()

    def test_put_and_get_latest(self):
        self.store.put("src", "k", {"v": 1}, data_as_of=date(2026, 9, 30))
        rec = self.store.get_latest("src", "k")
        self.assertEqual(rec.json(), {"v": 1})
        self.assertEqual(rec.data_as_of, date(2026, 9, 30))
        self.assertIsNone(self.store.get_latest("src", "other"))

    def test_identical_content_is_deduplicated_but_refreshes_last_checked(self):
        first = self.store.put("src", "k", {"v": 1})
        self.clock.advance(days=3)
        second = self.store.put("src", "k", {"v": 1})
        self.assertEqual(first.id, second.id)
        self.assertEqual(len(self.store.history("src", "k")), 1)
        self.assertEqual(second.fetched_at, T0)
        self.assertEqual(second.last_checked_at, T0 + timedelta(days=3))

    def test_changed_content_adds_a_row(self):
        self.store.put("src", "k", {"v": 1})
        self.clock.advance(days=1)
        self.store.put("src", "k", {"v": 2})
        self.assertEqual(len(self.store.history("src", "k")), 2)
        self.assertEqual(self.store.get_latest("src", "k").json(), {"v": 2})

    def test_as_of_replays_past_state(self):
        self.store.put("src", "k", {"v": 1})  # T0
        self.clock.advance(days=30)
        self.store.put("src", "k", {"v": 2})  # T0 + 30d
        self.assertIsNone(self.store.get_as_of("src", "k", T0 - timedelta(seconds=1)))
        self.assertEqual(self.store.get_as_of("src", "k", T0 + timedelta(days=10)).json(), {"v": 1})
        self.assertEqual(self.store.get_as_of("src", "k", T0 + timedelta(days=30)).json(), {"v": 2})

    def test_fetch_or_cache_respects_max_age(self):
        calls = []

        def fetcher():
            calls.append(1)
            return {"v": 1}

        age = timedelta(days=1)
        self.store.fetch_or_cache("src", "k", fetcher, max_age=age)
        self.store.fetch_or_cache("src", "k", fetcher, max_age=age)
        self.assertEqual(len(calls), 1)  # second call served from cache

        self.clock.advance(days=2)
        self.store.fetch_or_cache("src", "k", fetcher, max_age=age)
        self.assertEqual(len(calls), 2)  # stale, refetched
        self.assertEqual(len(self.store.history("src", "k")), 1)  # same content: no new row

        self.store.fetch_or_cache("src", "k", fetcher, max_age=age)
        self.assertEqual(len(calls), 2)  # refresh of last_checked made it fresh again

    def test_rejects_naive_datetimes(self):
        with self.assertRaises(ValueError):
            self.store.put("src", "k", {"v": 1}, fetched_at=datetime(2026, 1, 1))

    def test_persists_to_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "cache.db"
            with RawStore(path, clock=self.clock) as s1:
                s1.put("src", "k", "hello")
            with RawStore(path, clock=self.clock) as s2:
                self.assertEqual(s2.get_latest("src", "k").text(), "hello")


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.store = RawStore(clock=self.clock)

    def tearDown(self):
        self.store.close()

    def test_fake_worker_writes_a_schema_valid_report(self):
        report = FakeWorker(self.store).run(SEATTLE)
        # Re-validating from JSON proves the output satisfies the contract.
        WorkerReport.model_validate_json(report.model_dump_json())
        self.assertEqual(report.direction, "up")
        self.assertEqual(report.worker, "fake")
        self.assertEqual(report.data_as_of, T0.date())

    def test_cached_pull_is_reused(self):
        calls = []

        def fetcher():
            calls.append(1)
            return {"index": 100.0, "change_12m_pct": 4.2}

        worker = FakeWorker(self.store, fetcher=fetcher)
        worker.run(SEATTLE)
        worker.run(SEATTLE)
        self.assertEqual(len(calls), 1)

    def test_cached_pull_can_be_replayed_as_of(self):
        data = {"index": 100.0, "change_12m_pct": 4.2}
        worker = FakeWorker(self.store, fetcher=lambda: dict(data))
        self.assertEqual(worker.run(SEATTLE).direction, "up")

        self.clock.advance(days=40)
        data["change_12m_pct"] = -3.0  # market turned
        self.assertEqual(worker.run(SEATTLE).direction, "down")

        # Replaying a past date returns what we knew then, without any live fetch.
        self.assertEqual(worker.run(SEATTLE, as_of=T0 + timedelta(days=1)).direction, "up")
        with self.assertRaises(NoDataAsOf):
            worker.run(SEATTLE, as_of=T0 - timedelta(days=1))

    def test_base_worker_rejects_mismatched_reports(self):
        class WrongLocation(BaseWorker):
            name = "wrong"

            def _run(self, location, as_of):
                return make_report(worker="wrong", location=Location(city="Austin", state="TX"))

        class WrongName(BaseWorker):
            name = "right"

            def _run(self, location, as_of):
                return make_report(worker="other")

        with self.assertRaises(ValueError):
            WrongLocation(self.store).run(SEATTLE)
        with self.assertRaises(ValueError):
            WrongName(self.store).run(SEATTLE)


class LoggingTests(unittest.TestCase):
    def test_log_event_writes_json_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            logger = get_logger("retrend.test_events", path)
            log_event(logger, "hello", a=1, when=T0)
            logger.handlers[0].flush()
            row = json.loads(path.read_text().splitlines()[0])
            self.assertEqual(row["event"], "hello")
            self.assertEqual(row["a"], 1)
            self.assertIn("ts", row)
            get_logger("retrend.test_events")  # release the file handle

    def test_worker_run_logs_start_and_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            logger = get_logger("retrend.test_worker", path)
            store = RawStore(clock=Clock(), logger=logger)
            FakeWorker(store, logger).run(SEATTLE)
            for h in logger.handlers:
                h.flush()
            events = [json.loads(line)["event"] for line in path.read_text().splitlines()]
            self.assertEqual(
                events, ["worker_start", "cache_miss", "pull_stored", "worker_done"]
            )
            store.close()
            get_logger("retrend.test_worker")


if __name__ == "__main__":
    unittest.main()
