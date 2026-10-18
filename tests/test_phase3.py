"""Phase 3: supervisor critic, weighted synthesis, retry, degradation, narrative, CLI."""

import io
import json
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone

from helpers import T0, Clock
from retrend import cli
from retrend.llm import JsonJudge, StubChat
from retrend.location import Location
from retrend.outlook import Outlook
from retrend.schema import Source, WorkerReport
from retrend.store import RawStore
from retrend.supervisor import (APPROX_GEOGRAPHY_FACTOR, DEFAULT_WEIGHTS, LEGACY_WEIGHTS,
                                Supervisor, freshness_factor)
from retrend.workers.base import BaseWorker, NoDataAsOf
from test_phase1 import payload_from, series
from test_phase2 import all_files, opener

SEATTLE = Location(city="Seattle", state="WA")
TODAY = T0.date()


def report(worker, direction, confidence=0.8, *, age=10, geo="exact", judged_by="m",
           loc=SEATTLE, pit=True):
    return WorkerReport(
        worker=worker, location=loc, signal=worker, direction=direction, confidence=confidence,
        key_metrics={"point_in_time": pit}, data_as_of=TODAY - timedelta(days=age),
        geography_match=geo, caveats=[], summary=f"{worker} says {direction}.",
        sources=[Source(name="s", retrieved=T0)], judged_by=judged_by)


class Scripted(BaseWorker):
    """Returns the next scripted item: a report, or an exception to raise."""

    def __init__(self, store, name, *items):
        super().__init__(store)
        self.name, self.items, self.calls = name, list(items), 0

    def _run(self, location, as_of):
        item = self.items[min(self.calls, len(self.items) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.store = RawStore(clock=Clock())

    def tearDown(self):
        self.store.close()

    def sup(self, spec, **kw):
        workers = {n: Scripted(self.store, n, *items) for n, items in spec.items()}
        return Supervisor(self.store, workers, **kw), workers

    def test_all_agree_up(self):
        s, _ = self.sup({"housing_prices": [report("housing_prices", "up")],
                         "supply": [report("supply", "up")],
                         "mortgage_rates": [report("mortgage_rates", "up", geo="approximate")]})
        o = s.run(SEATTLE)
        Outlook.model_validate(o.model_dump())
        self.assertEqual(o.direction, "up")
        self.assertAlmostEqual(o.score, 1.0)  # unanimous: vote strength is 1 whatever the confidence
        self.assertEqual({c.worker for c in o.drivers}, {"housing_prices", "supply"})
        self.assertEqual([c.worker for c in o.context], ["mortgage_rates"])  # weight 0: shown only
        self.assertEqual(o.dissent, [])
        self.assertEqual(o.missing_workers, {})
        self.assertAlmostEqual(sum(c.contribution for c in o.drivers), o.score, places=2)
        self.assertTrue(all(c.sources for c in o.drivers + o.context))  # evidence on everything shown
        self.assertEqual(o.narrated_by, "template")

    def test_hand_computed_conflict(self):
        # housing up .8, supply down .6, mortgage down .5 with default weights
        s, _ = self.sup({"housing_prices": [report("housing_prices", "up", 0.8)],
                         "supply": [report("supply", "down", 0.6)],
                         "mortgage_rates": [report("mortgage_rates", "down", 0.5)]},
                       weights=LEGACY_WEIGHTS)
        o = s.run(SEATTLE)
        w = LEGACY_WEIGHTS
        num = w["housing_prices"] * 0.8 - w["supply"] * 0.6 - w["mortgage_rates"] * 0.5
        den = w["housing_prices"] * 0.8 + w["supply"] * 0.6 + w["mortgage_rates"] * 0.5
        self.assertAlmostEqual(o.score, round(num / den, 3), places=3)
        self.assertEqual(o.direction, "flat")  # about -0.02, inside the flat band
        self.assertTrue(any(f.kind == "conflict" for f in o.findings))
        self.assertLess(o.confidence, 0.5)

    def test_dissent_listed_when_direction_is_up(self):
        s, _ = self.sup({"housing_prices": [report("housing_prices", "up", 0.9)],
                         "supply": [report("supply", "up", 0.9)],
                         "mortgage_rates": [report("mortgage_rates", "down", 0.5)]},
                       weights=LEGACY_WEIGHTS)
        o = s.run(SEATTLE)
        self.assertEqual(o.direction, "up")
        self.assertEqual([c.worker for c in o.dissent], ["mortgage_rates"])
        self.assertLess(o.dissent[0].contribution, 0)

    def test_missing_worker_degrades_and_is_reported(self):
        full, _ = self.sup({"housing_prices": [report("housing_prices", "up")],
                            "supply": [report("supply", "up")],
                            "mortgage_rates": [report("mortgage_rates", "up")]})
        part, w = self.sup({"housing_prices": [report("housing_prices", "up")],
                            "supply": [NoDataAsOf("nothing")],
                            "mortgage_rates": [report("mortgage_rates", "up")]})
        a, b = full.run(SEATTLE), part.run(SEATTLE)
        self.assertIn("supply", b.missing_workers)
        self.assertLess(b.confidence, a.confidence)
        self.assertEqual(w["supply"].calls, 1)  # NoDataAsOf is not retried
        self.assertTrue(any(f.kind == "missing_worker" and f.worker == "supply" for f in b.findings))
        self.assertIn("1 of 2", b.uncertainty)  # voting signals only

    def test_transient_failure_is_retried_once(self):
        s, w = self.sup({"housing_prices": [OSError("boom"), report("housing_prices", "up")],
                         "supply": [report("supply", "up")],
                         "mortgage_rates": [report("mortgage_rates", "up")]})
        o = s.run(SEATTLE)
        self.assertEqual(w["housing_prices"].calls, 2)
        self.assertEqual(w["supply"].calls, 1)  # healthy workers are not rerun
        self.assertEqual(o.missing_workers, {})
        self.assertTrue(any(f.kind == "retried" for f in o.findings))

    def test_persistent_failure_stops_at_max_attempts(self):
        s, w = self.sup({"housing_prices": [OSError("down")],
                         "supply": [report("supply", "up")]}, max_attempts=3)
        o = s.run(SEATTLE)
        self.assertEqual(w["housing_prices"].calls, 3)
        self.assertIn("housing_prices", o.missing_workers)
        self.assertIn("OSError", o.missing_workers["housing_prices"])

    def test_everything_missing_is_flat_zero_confidence(self):
        s, _ = self.sup({"a": [NoDataAsOf("x")], "b": [ValueError("y")]},
                        weights={"a": 0.5, "b": 0.5})
        o = s.run(SEATTLE)
        self.assertEqual((o.direction, o.score, o.confidence), ("flat", 0.0, 0.0))
        self.assertEqual(set(o.missing_workers), {"a", "b"})

    def test_stale_and_approximate_reduce_effective_confidence(self):
        self.assertEqual(freshness_factor("mortgage_rates", 14), 1.0)
        self.assertEqual(freshness_factor("mortgage_rates", 28), 0.5)
        self.assertEqual(freshness_factor("supply", 10000), 0.25)
        s, _ = self.sup({"housing_prices": [report("housing_prices", "up", 0.8, age=150, geo="approximate")],
                         "mortgage_rates": [report("mortgage_rates", "up", 0.8, geo="approximate")]},
                       weights=LEGACY_WEIGHTS)
        o = s.run(SEATTLE)
        by = {c.worker: c for c in o.drivers}
        self.assertLess(by["housing_prices"].freshness_factor, 1.0)
        self.assertEqual(by["housing_prices"].geography_factor, APPROX_GEOGRAPHY_FACTOR)
        self.assertEqual(by["mortgage_rates"].geography_factor, 1.0)  # national signal: no penalty
        kinds = {(f.kind, f.worker) for f in o.findings}
        self.assertIn(("stale_data", "housing_prices"), kinds)
        self.assertIn(("approximate_geography", "housing_prices"), kinds)
        self.assertNotIn(("approximate_geography", "mortgage_rates"), kinds)

    def test_score_does_not_depend_on_absolute_confidence(self):
        def run(conf):
            s, _ = self.sup({"housing_prices": [report("housing_prices", "up", conf)],
                             "supply": [report("supply", "up", conf)]})
            return s.run(SEATTLE)
        low, high = run(0.3), run(0.9)
        self.assertEqual((low.score, low.direction), (high.score, high.direction))
        self.assertEqual(low.direction, "up")
        self.assertLess(low.confidence, high.confidence)  # confidence still reflects it

    def test_flat_report_pulls_score_toward_flat(self):
        s, _ = self.sup({"housing_prices": [report("housing_prices", "up", 0.6)],
                         "supply": [report("supply", "flat", 0.6)]})
        self.assertEqual(s.run(SEATTLE).score, 0.5)

    def test_zero_weight_worker_is_context_and_never_a_conflict(self):
        s, _ = self.sup({"housing_prices": [report("housing_prices", "up", 0.8)],
                         "supply": [report("supply", "up", 0.8)],
                         "mortgage_rates": [report("mortgage_rates", "down", 0.9)]})
        o = s.run(SEATTLE)
        self.assertEqual(o.direction, "up")
        self.assertEqual(o.dissent, [])
        self.assertEqual([c.worker for c in o.context], ["mortgage_rates"])
        self.assertFalse(any(f.kind == "conflict" for f in o.findings))
        self.assertIn("context only", o.narrative)
        # a missing zero-weight worker costs nothing
        s2, _ = self.sup({"housing_prices": [report("housing_prices", "up", 0.8)],
                          "supply": [report("supply", "up", 0.8)],
                          "mortgage_rates": [OSError("x")]}, max_attempts=1)
        o2 = s2.run(SEATTLE)
        self.assertIn("mortgage_rates", o2.missing_workers)
        self.assertEqual(o2.confidence, o.confidence)

    def test_rule_only_and_not_point_in_time_flagged(self):
        s, _ = self.sup({"housing_prices": [report("housing_prices", "up", judged_by="rule:x", pit=False)]})
        kinds = {f.kind for f in s.run(SEATTLE).findings}
        self.assertTrue({"rule_only", "not_point_in_time"} <= kinds)

    def test_narrator_model_used_or_template_on_failure(self):
        ok = JsonJudge(StubChat([json.dumps({"explanation": "Prices look set to rise."})]),
                       store=self.store)
        s, _ = self.sup({"supply": [report("supply", "up")]}, narrator=ok)
        o = s.run(SEATTLE)
        self.assertEqual(o.narrative, "Prices look set to rise.")
        self.assertEqual(o.narrated_by, "stub")
        bad = JsonJudge(StubChat(["not json"] * 5, name="bad"), store=self.store)
        s, _ = self.sup({"supply": [report("supply", "up")]}, narrator=bad)
        o = s.run(SEATTLE)
        self.assertEqual(o.narrated_by, "template (model failed)")
        self.assertIn("Outlook", o.narrative)

    def test_narrator_prompt_hides_place_and_dates(self):
        chat = StubChat([json.dumps({"explanation": "ok"})])
        s, _ = self.sup({"supply": [report("supply", "up")]},
                        narrator=JsonJudge(chat, store=self.store))
        s.run(SEATTLE)
        sent = json.dumps(chat.calls[0])
        self.assertNotIn("Seattle", sent)
        self.assertNotIn("2026", sent)

    def test_as_of_is_passed_to_workers(self):
        seen = []

        class Spy(Scripted):
            def _run(self, location, as_of):
                seen.append(as_of)
                return super()._run(location, as_of)

        w = {"supply": Spy(self.store, "supply", report("supply", "up", age=100))}
        as_of = datetime(2026, 9, 30, 23, 59, tzinfo=timezone.utc)
        o = Supervisor(self.store, w).run(SEATTLE, as_of=as_of)
        self.assertEqual(seen, [as_of])
        self.assertEqual(o.as_of, date(2026, 9, 30))

    def test_langgraph_engine_matches_plain_when_installed(self):
        try:
            import langgraph  # noqa: F401
        except ImportError:
            self.skipTest("langgraph not installed")
        from retrend.supervisor_graph import run_with_graph
        spec = {"housing_prices": [OSError("x"), report("housing_prices", "up")],
                "supply": [report("supply", "up")]}
        a = self.sup(spec)[0].run(SEATTLE)
        b = run_with_graph(self.sup(spec)[0], SEATTLE)
        self.assertEqual((a.direction, a.score, a.confidence), (b.direction, b.score, b.confidence))


class CliTests(unittest.TestCase):
    def test_outlook_end_to_end_with_fixture_data(self):
        with tempfile.TemporaryDirectory() as d:
            out = io.StringIO()
            rc = cli.main(
                ["--env-file", d + "/.env", "--config-dir", "config", "--db", d + "/c.db",
                 "--log", d + "/l.jsonl", "outlook", "seattle-wa", "--models", "baseline,stub"],
                fred_source=lambda *a, **k: payload_from(series()),
                open_url=opener(all_files()), out=out)
            text = out.getvalue()
            self.assertEqual(rc, 0, text)
            self.assertIn("Outlook (next 6 to 12 months)", text)
            self.assertEqual(text.count("==="), 4)  # header lines for baseline and stub
            saved = list((__import__("pathlib").Path(d) / "reports" / "seattle-wa").glob("outlook_*"))
            self.assertEqual(len(saved), 2)
            Outlook.model_validate_json(saved[0].read_text())


if __name__ == "__main__":
    unittest.main()
