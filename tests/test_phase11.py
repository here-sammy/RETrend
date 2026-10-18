"""School-enrollment test, backtest schools variants, only_<signal> fix, UI schools lookup."""

import io
import json
import random
import shutil
import tempfile
import unittest
from datetime import date
from pathlib import Path

from retrend import cli
from retrend.eval import backtest as bt
from retrend.eval import metrics as M
from retrend.eval.schools_test import format_schools_test, run_schools_test
from retrend.eval.truth import ZhviTruth
from retrend.sources.schools import SchoolsLoader
from retrend.workers.schools import SchoolsWorker
from test_phase2 import month_ends
from test_phase4 import LAST, LOCS, N_MONTHS, NAMES, Env, fake_fred, opener_for, values_for, zillow_csv

DAYS = month_ends(N_MONTHS, LAST)


def price(i, y, m):
    """ZHVI of fixture city i at the end of month m of year y."""
    return values_for(i)[DAYS.index(next(d for d in DAYS if d.year == y and d.month == m))]


def schools_csv(planted: bool, seed=1):
    """Enrollment by school year 2013..2023. planted=True: enrollment growth in year y equals half the
    price growth over the NEXT year (a lead-lag effect built in on purpose). Otherwise it is random."""
    rng = random.Random(seed)
    lines = ["District Name,State,School Year,Enrollment"]
    for i, (c, s) in enumerate(NAMES):
        n = 50000.0
        for y in range(2013, 2024):
            g = price(i, y + 2, 6) / price(i, y + 1, 6) - 1 if planted else rng.gauss(0.01, 0.03)
            n *= 1 + (0.5 * g if planted else g)
            lines.append(f"{c} Public Schools,{s},{y}-{str(y + 1)[2:]},{int(n)}")
    return "\n".join(lines) + "\n"


class MetricTests(unittest.TestCase):
    def test_partial_spearman_removes_a_shared_driver(self):
        z = list(range(20))
        rng = random.Random(3)
        x = [v + rng.gauss(0, 2) for v in z]   # x and y both just follow z, plus independent noise
        y = [v + rng.gauss(0, 2) for v in z]
        self.assertGreater(M.spearman(x, y), 0.8)
        self.assertLess(abs(M.partial_spearman(x, y, z)), 0.5)


class SchoolsTestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = Env(self.tmp.name)
        self.dir = Path(self.tmp.name) / "schools"
        self.dir.mkdir()
        self.dates = bt.month_end_dates(date(2017, 6, 30), date(2025, 6, 30), 12)

    def tearDown(self):
        self.env.close()
        self.tmp.cleanup()

    def run_test(self, planted):
        (self.dir / "e.csv").write_text(schools_csv(planted))
        worker = SchoolsWorker(self.env.store, judge=None, loader=SchoolsLoader(self.dir))
        truth = ZhviTruth(self.env.housing.zhvi_series)
        return run_schools_test(worker, truth, LOCS, self.dates, horizon=12)

    def test_planted_relationship_is_detected(self):
        r = self.run_test(True)
        self.assertEqual(r["n_cities_matched"], 12)
        self.assertGreaterEqual(r["n_dates"], 6)
        self.assertGreater(r["raw"]["mean"], 0.3)
        self.assertGreater(r["raw"]["ci90"][0], 0)
        self.assertIn("Average rank correlation", r["verdict"])
        text = format_schools_test(r)
        self.assertIn("Limits:", text)

    def test_random_enrollment_shows_nothing(self):
        r = self.run_test(False)
        self.assertLess(abs(r["raw"]["mean"]), 0.35)
        self.assertNotIn("adds information", r["verdict"])

    def test_unmatched_cities_are_listed_not_hidden(self):
        (self.dir / "e.csv").write_text(schools_csv(False).replace("Seattle Public Schools", "Elsewhere SD"))
        worker = SchoolsWorker(self.env.store, judge=None, loader=SchoolsLoader(self.dir))
        r = run_schools_test(worker, ZhviTruth(self.env.housing.zhvi_series), LOCS, self.dates)
        self.assertEqual(r["n_cities_matched"], 11)
        self.assertIn("seattle-wa", r["unmatched"])

    def test_too_few_cities_gives_no_verdict_claim(self):
        (self.dir / "e.csv").write_text(schools_csv(True))
        worker = SchoolsWorker(self.env.store, judge=None, loader=SchoolsLoader(self.dir))
        r = run_schools_test(worker, ZhviTruth(self.env.housing.zhvi_series), LOCS[:4], self.dates)
        self.assertIn("Not enough data", r["verdict"])


class BacktestVariantTests(unittest.TestCase):
    def test_only_variants_are_not_all_flat_and_schools_variants_exist(self):
        with tempfile.TemporaryDirectory() as d:
            env = Env(d)
            sd = Path(d) / "schools"
            sd.mkdir()
            (sd / "e.csv").write_text(schools_csv(True))
            workers = dict(env.workers)
            workers["schools"] = SchoolsWorker(env.store, judge=None, loader=SchoolsLoader(sd))
            dates = bt.month_end_dates(date(2018, 6, 30), date(2024, 6, 30), 6)
            result = bt.run_backtest(env.store, workers, LOCS, env.truth, dates)
            report = bt.evaluate(env.store, result, list(workers))
            env.close()
        v = report["variants"]["all"]
        for name in ("full_plus_schools_0.1", "full_plus_schools_0.2", "only_schools", "only_mortgage_rates"):
            self.assertIn(name, v)
        self.assertGreater(v["only_mortgage_rates"]["n_up"] + v["only_mortgage_rates"]["n_down"], 0)
        self.assertGreater(v["only_schools"]["n_up"] + v["only_schools"]["n_down"], 0)


class CliTests(unittest.TestCase):
    def test_schools_test_command_and_backtest_flag(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "cfg"
            cfg.mkdir()
            shutil.copy("config/models.yaml", cfg / "models.yaml")
            (cfg / "cities.yaml").write_text("cities:\n" + "".join(
                f"  - city: {c}\n    state: {s}\n" for c, s in NAMES))
            sd = Path(d) / "data" / "files" / "schools"
            sd.mkdir(parents=True)
            (sd / "e.csv").write_text(schools_csv(True))
            base = ["--env-file", d + "/.env", "--config-dir", str(cfg), "--db", d + "/data/c.db",
                    "--log", d + "/l.jsonl"]
            op = opener_for(zillow_csv())
            out = io.StringIO()
            rc = cli.main(base + ["schools-test", "--registry", "cities.yaml", "--start", "2017-06-30",
                                  "--end", "2025-06-30", "--step-months", "12"], open_url=op, out=out)
            self.assertEqual(rc, 0, out.getvalue())
            self.assertIn("School enrollment test", out.getvalue())
            self.assertTrue(list((Path(d) / "data" / "eval").glob("schools_test_*.json")))
            out = io.StringIO()
            rc = cli.main(base + ["backtest", "--registry", "cities.yaml", "--start", "2018-06-30",
                                  "--end", "2024-06-30", "--step-months", "6", "--with-schools"],
                          fred_source=fake_fred, open_url=op, out=out)
            self.assertEqual(rc, 0, out.getvalue())
            self.assertIn("full_plus_schools_0.1", out.getvalue())
            self.assertIn("schools ", out.getvalue())


if __name__ == "__main__":
    unittest.main()


class UiLookupTests(unittest.TestCase):
    def test_school_lookup_returns_history_cards_without_a_model(self):
        import http.client
        import threading
        from retrend.store import RawStore
        from retrend.ui.runner import QueryRunner
        from retrend.ui.server import JobThread, serve
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cfg = d / "cfg"
            cfg.mkdir()
            for f in ("models.yaml", "test_cities.yaml", "backtest_cities.yaml"):
                shutil.copy(f"config/{f}", cfg / f)
            sd = d / "data" / "files" / "schools"
            sd.mkdir(parents=True)
            (sd / "e.csv").write_text(schools_csv(False))
            db = str(d / "data" / "c.db")
            holder = {}

            def make():
                holder["s"] = RawStore(db)
                return QueryRunner(config_dir=cfg, db_path=db, store=holder["s"], logger=None,
                                   fred_source=fake_fred, open_url=opener_for(zillow_csv()))
            jobs = JobThread(make, close=lambda: holder["s"].close())
            httpd = serve(jobs, port=0, warnings=[], monthly_dir=d / "m", config_dir=cfg)
            port = httpd.server_address[1]
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            try:
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
                c.request("GET", "/api/schools?city=Seattle&state=WA", headers={"Host": f"127.0.0.1:{port}"})
                card = json.loads(c.getresponse().read().decode())
            finally:
                httpd.shutdown()
                httpd.server_close()
                jobs.stop()
                jobs.join(5)
        self.assertEqual(card["district"], "Seattle Public Schools")
        self.assertEqual(card["history"][-1]["year"], card["latest_year"])
        self.assertIsNone(card["history"][0]["change_pct"])
        self.assertIsNotNone(card["history"][-1]["change_pct"])
        self.assertGreater(card["latest_students"], 0)


class TabApiTests(unittest.TestCase):
    def setUp(self):
        import threading
        from retrend.store import RawStore
        from retrend.ui.runner import QueryRunner
        from retrend.ui.server import JobThread, serve
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        cfg = d / "cfg"
        cfg.mkdir()
        for f in ("models.yaml", "test_cities.yaml", "backtest_cities.yaml"):
            shutil.copy(f"config/{f}", cfg / f)
        sd = d / "data" / "files" / "schools"
        sd.mkdir(parents=True)
        (sd / "e.csv").write_text(schools_csv(True))
        db = str(d / "data" / "c.db")
        holder = {}

        def make():
            holder["s"] = RawStore(db)
            return QueryRunner(config_dir=cfg, db_path=db, store=holder["s"], logger=None,
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

    def get(self, path, host=None):
        import http.client
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        c.request("GET", path, headers={"Host": host or f"127.0.0.1:{self.port}"})
        r = c.getresponse()
        return r.status, json.loads(r.read())

    def test_schools_endpoint_returns_full_history(self):
        st, j = self.get("/api/schools?city=seattle&state=wa")
        self.assertEqual(st, 200)
        self.assertEqual((j["city"], j["state"], j["district"]), ("Seattle", "WA", "Seattle Public Schools"))
        self.assertEqual([r["year"] for r in j["history"]], list(range(2013, 2024)))

    def test_prices_endpoint_has_derived_series(self):
        st, j = self.get("/api/prices?city=Austin&state=TX")
        self.assertEqual(st, 200)
        self.assertEqual(j["region"], "Austin")
        for k in ("zhvi", "zhvi_yoy", "drawdown"):
            self.assertIn(k, j["series"])
        self.assertNotIn("zori", j["series"])  # the fixture has no rent file; rents are optional
        self.assertLessEqual(j["stats"]["from_peak_pct"], 0)
        first = j["series"]["zhvi"]["points"][0][0]
        self.assertEqual(j["series"]["zhvi_yoy"]["points"][0][0][:4], str(int(first[:4]) + 1))

    def test_bad_input_and_missing_places(self):
        self.assertEqual(self.get("/api/prices?city=Aus%3Cscript%3E&state=TX")[0], 400)
        self.assertEqual(self.get("/api/prices?city=Austin&state=Texas")[0], 400)
        self.assertEqual(self.get("/api/prices?city=&state=TX")[0], 400)
        st, j = self.get("/api/schools?city=Nowhere&state=WA")
        self.assertEqual(st, 404)
        self.assertIn("NoDistrictMatch", j["error"])
        self.assertEqual(self.get("/api/schools?city=Seattle&state=WA", host="evil.example")[0], 403)


class PriceViewTests(unittest.TestCase):
    def test_derived_measures(self):
        from retrend.location import Location
        from retrend.ui.prices import price_view
        z = [(f"{2020 + i // 12}-{i % 12 + 1:02d}-28", 100.0 + i) for i in range(30)]
        z[20] = (z[20][0], 80.0)  # a dip below the earlier peak
        r = [(d, 10.0) for d, _ in z]
        v = price_view(Location(city="X", state="WA"), {"geo": "city", "region": "X", "quality": "exact", "zhvi": z, "zori": r})
        self.assertEqual(v["series"]["zhvi_yoy"]["points"][0][1], 12.0)  # 112 vs 100 a year earlier
        self.assertLess(min(p[1] for p in v["series"]["drawdown"]["points"]), -10)
        self.assertEqual(v["series"]["price_to_rent"]["points"][0][1], round(100 / 120, 1))
        self.assertEqual(v["stats"]["rent"], 10.0)
