"""School enrollment worker: loader, metrics, worker, planner routing, UI."""

import io
import json
import shutil
import tempfile
import threading
import unittest
import http.client
from datetime import datetime, timezone
from pathlib import Path

from retrend.location import Location, load_registry
from retrend.sources.files import SourceError
from retrend.sources.schools import (NoDistrictMatch, SchoolsLoader, district_matches, extract_district,
                                     parse_count, parse_year)
from retrend.store import RawStore
from retrend.supervisor import Supervisor
from retrend.ui import planner as P
from retrend.ui.planner import rule_plan
from retrend.ui.runner import QueryRunner
from retrend.ui.server import JobThread, serve
from retrend.workers.base import NoDataAsOf
from retrend.workers.schools import SchoolsWorker
from retrend.workers.schools_metrics import compute_metrics
from test_phase4 import fake_fred, opener_for, zillow_csv

SEA = Location(city="Seattle", state="WA")
REG = load_registry("config/backtest_cities.yaml")


def csv_text(rows, header="District Name,State,School Year,Enrollment"):
    return header + "\n" + "\n".join(",".join(str(c) for c in r) for r in rows) + "\n"


GROWING = [("Seattle Public Schools", "WA", f"{y}-{str(y + 1)[2:]}", int(50000 * 1.02 ** (y - 2018)))
           for y in range(2018, 2025)]
SHRINKING = [("Seattle Public Schools", "WA", y, int(50000 * 0.97 ** (y - 2018))) for y in range(2018, 2025)]


class LoaderTests(unittest.TestCase):
    def load(self, text, loc=SEA):
        return extract_district(io.StringIO(text), loc)

    def test_parsers(self):
        self.assertEqual(parse_year("2023-24"), 2023)
        self.assertEqual(parse_year("2023"), 2023)
        self.assertIsNone(parse_year("n/a"))
        self.assertEqual(parse_count("51,234"), 51234.0)
        self.assertIsNone(parse_count("-1"))  # NCES missing code
        self.assertIsNone(parse_count(""))

    def test_district_name_matching(self):
        self.assertTrue(district_matches("Seattle School District No. 1", "Seattle"))
        self.assertTrue(district_matches("Austin ISD", "Austin"))
        self.assertFalse(district_matches("North Seattle Charter", "Seattle"))
        self.assertTrue(district_matches("Salt Lake City School District", "Salt Lake City"))

    def test_extracts_rows_and_state_names(self):
        p = self.load(csv_text([("Seattle Public Schools", "Washington", "2022-23", "50,000"),
                                ("Seattle Public Schools", "WA", "2023-24", "51,000"),
                                ("Seattle Public Schools", "OR", "2023-24", "9"),  # wrong state
                                ("Tacoma Public Schools", "WA", "2023-24", "30,000")]))
        self.assertEqual([r["year"] for r in p["rows"]], [2022, 2023])
        self.assertEqual(p["quality"], "exact")

    def test_header_aliases(self):
        p = self.load(csv_text([("Seattle SD", "WA", 2022, 100), ("Seattle SD", "WA", 2023, 101)],
                               header="LEA_NAME,STABR,YEAR,MEMBERSHIP"))
        self.assertEqual(len(p["rows"]), 2)

    def test_duplicate_district_year_rows_fail_loudly(self):
        with self.assertRaises(SourceError) as cm:
            self.load(csv_text([("Seattle SD", "WA", 2022, 100), ("Seattle SD", "WA", 2022, 50)]))
        self.assertIn("per-grade", str(cm.exception))

    def test_missing_column_and_no_match(self):
        with self.assertRaises(SourceError) as cm:
            self.load("a,b\n1,2\n")
        self.assertIn("columns found", str(cm.exception))
        with self.assertRaises(NoDistrictMatch):
            self.load(csv_text([("Tacoma Public Schools", "WA", 2022, 1)]))

    def test_multiple_matches_pick_largest_and_flag(self):
        p = self.load(csv_text([("Seattle Public Schools", "WA", 2023, 50000),
                                ("Seattle Prep Charter", "WA", 2023, 300)]))
        self.assertEqual(p["district"], "Seattle Public Schools")
        self.assertEqual(p["quality"], "approximate")


class MetricTests(unittest.TestCase):
    def rows(self, growth, n=7):
        return [{"year": 2018 + i, "students": 50000 * growth ** i} for i in range(n)]

    def test_growth_and_decline(self):
        up = compute_metrics(self.rows(1.02))
        self.assertEqual(up["baseline_direction"], "up")
        self.assertAlmostEqual(up["enrollment_yoy_pct"], 2.0, places=1)
        self.assertAlmostEqual(up["enrollment_3y_cagr_pct"], 2.0, places=1)
        self.assertEqual(compute_metrics(self.rows(0.97))["baseline_direction"], "down")
        self.assertEqual(compute_metrics(self.rows(1.001))["baseline_direction"], "flat")

    def test_short_history_has_none_metrics(self):
        m = compute_metrics(self.rows(1.02, n=3))
        self.assertIsNone(m["enrollment_3y_cagr_pct"])
        self.assertIsNone(m["enrollment_5y_change_pct"])


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def worker(self, store, rows):
        (self.dir / "e.csv").write_text(csv_text(rows))
        return SchoolsWorker(store, judge=None, loader=SchoolsLoader(self.dir))

    def test_report_is_weak_and_labeled(self):
        with RawStore() as st:
            r = self.worker(st, GROWING).run(SEA)
        self.assertEqual((r.worker, r.direction, r.geography_match), ("schools", "up", "approximate"))
        self.assertLessEqual(r.confidence, 0.5)
        self.assertEqual(r.judged_by, "rule:enrollment-trend")
        self.assertTrue(any("district" in c.lower() for c in r.caveats))

    def test_as_of_applies_publication_lag(self):
        with RawStore() as st:
            w = self.worker(st, GROWING)  # school years 2018..2024
            before = w.run(SEA, as_of=datetime(2023, 5, 1, tzinfo=timezone.utc))
            after = w.run(SEA, as_of=datetime(2023, 9, 1, tzinfo=timezone.utc))
        self.assertEqual(before.key_metrics["latest_year"], 2021)  # fall 2022 data public by 2023-06-30
        self.assertEqual(after.key_metrics["latest_year"], 2022)
        self.assertFalse(after.key_metrics["point_in_time"])

    def test_too_little_history_before_as_of(self):
        with RawStore() as st:
            w = self.worker(st, GROWING)
            with self.assertRaises(NoDataAsOf):
                w.run(SEA, as_of=datetime(2020, 1, 1, tzinfo=timezone.utc))

    def test_missing_file_is_a_clear_error(self):
        with RawStore() as st:
            w = SchoolsWorker(st, judge=None, loader=SchoolsLoader(self.dir / "nope"))
            with self.assertRaises(SourceError) as cm:
                w.run(SEA)
        self.assertIn("no school enrollment file", str(cm.exception))

    def test_supervisor_shows_schools_as_context_not_vote(self):
        with RawStore() as st:
            w = self.worker(st, SHRINKING)
            from test_phase4 import Env
            env = Env(self.tmp.name)
            workers = dict(env.workers)
            workers["schools"] = SchoolsWorker(env.store, judge=None, loader=SchoolsLoader(self.dir))
            o = Supervisor(env.store, workers, expected=["housing_prices", "schools"]).run(SEA)
            env.close()
        self.assertIn("schools", [c.worker for c in o.context])
        self.assertNotIn("schools", [c.worker for c in o.drivers + o.dissent])
        self.assertEqual(o.weights["schools"], 0.0)


class PlannerTests(unittest.TestCase):
    def test_school_only_question(self):
        p = rule_plan("Is school enrollment growing in Seattle, WA?", REG)
        self.assertEqual((p.workers, p.intent), (P.KNOWN_WORKERS, "outlook"))
        self.assertEqual(p.unsupported, [])

    def test_schools_added_to_price_question(self):
        p = rule_plan("Will prices rise in Seattle, WA and what about school enrollment?", REG)
        self.assertIn("schools", p.workers)
        self.assertIn("housing_prices", p.workers)

    def test_general_question_runs_schools_too(self):
        self.assertIn("schools", rule_plan("Will home prices rise in Seattle, WA?", REG).workers)

    def test_school_quality_is_unsupported(self):
        p = rule_plan("Does Seattle, WA have good schools?", REG)
        self.assertIn("school quality or ratings", p.unsupported)


class UiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        cfg = d / "cfg"
        cfg.mkdir()
        for f in ("models.yaml", "test_cities.yaml", "backtest_cities.yaml"):
            shutil.copy(f"config/{f}", cfg / f)
        self.db = str(d / "data" / "cache.db")
        self.sdir = d / "data" / "files" / "schools"
        holder = {}

        def make():
            store = RawStore(self.db)
            holder["s"] = store
            return QueryRunner(config_dir=cfg, db_path=self.db, store=store, logger=None,
                               fred_source=fake_fred, open_url=opener_for(zillow_csv()))

        self.jobs = JobThread(make, close=lambda: holder["s"].close())
        self.httpd = serve(self.jobs, port=0, warnings=[], monthly_dir=d / "m", config_dir=cfg)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.jobs.stop()
        self.jobs.join(5)
        self.tmp.cleanup()

    def events(self, q):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        c.request("POST", "/api/ask", json.dumps({"query": q, "model": "baseline"}),
                  {"Host": f"127.0.0.1:{self.port}", "Content-Type": "application/json"})
        data = c.getresponse().read()
        return [json.loads(l) for l in data.decode().splitlines() if l.strip()]

    def test_school_question_end_to_end(self):
        self.sdir.mkdir(parents=True)
        (self.sdir / "e.csv").write_text(csv_text(GROWING))
        evs = self.events("Is school enrollment growing in Seattle, WA?")
        done = [e for e in evs if e["event"] == "worker_done"]
        self.assertIn(("schools", True), [(e["worker"], e["ok"]) for e in done])
        ans = next(e for e in evs if e["event"] == "answer")
        self.assertIn("School enrollment", ans["answer"])

    def test_missing_school_file_reports_error_without_crashing(self):
        evs = self.events("Is school enrollment growing in Seattle, WA?")
        done = [e for e in evs if e["event"] == "worker_done"]
        bad = next(e for e in done if e["worker"] == "schools")
        self.assertFalse(bad["ok"])
        self.assertIn("no school enrollment file", bad["error"])
        self.assertEqual(evs[-1]["event"], "done")


if __name__ == "__main__":
    unittest.main()
