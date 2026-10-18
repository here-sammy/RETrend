"""ZIP resolver, FRED metro-series finder, jobs and permits workers, and their wiring."""

import io
import json
import shutil
import tempfile
import unittest
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from helpers import Clock
from retrend import cli
from retrend.fred import FredClient, FredError
from retrend.llm import JsonJudge, StubChat
from retrend.location import Location
from retrend.sources.fred_metro import MetroSeriesFinder, NoMetroSeries, load_overrides, pick_series
from retrend.sources.files import SourceError
from retrend.sources.zipcodes import ZipError, ZipNotFound, ZipResolver, fetch_zip_csv, parse_geonames
from retrend.store import RawStore
from retrend.supervisor import DEFAULT_WEIGHTS, Supervisor
from retrend.ui.planner import rule_plan, select_workers
from retrend.workers.base import NoDataAsOf
from retrend.workers.jobs import JobsWorker
from retrend.workers.permits import PermitsWorker
from retrend.workers.permits_metrics import compute_metrics as permit_metrics
from test_phase4 import LAST, NAMES, fake_fred, opener_for, zillow_csv

SEA = Location(city="Seattle", state="WA")
AS_OF = datetime(2026, 10, 1, tzinfo=timezone.utc)

SEARCH_ROWS = {
    "unemployment": [
        {"id": "WRONGNSA", "title": "Unemployment Rate in Seattle-Tacoma-Bellevue, WA (MSA)", "frequency_short": "M", "seasonal_adjustment_short": "NSA"},
        {"id": "SEAT653UR", "title": "Unemployment Rate in Seattle-Tacoma-Bellevue, WA (MSA)", "frequency_short": "M", "seasonal_adjustment_short": "SA"},
        {"id": "PORT", "title": "Unemployment Rate in Portland-Vancouver-Hillsboro, OR-WA (MSA)", "frequency_short": "M", "seasonal_adjustment_short": "SA"},
        {"id": "ANN", "title": "Unemployment Rate in Seattle-Tacoma-Bellevue, WA (MSA)", "frequency_short": "A", "seasonal_adjustment_short": "NSA"},
    ],
    "payrolls": [
        {"id": "SEAT653NA", "title": "All Employees: Total Nonfarm in Seattle-Tacoma-Bellevue, WA (MSA)", "frequency_short": "M", "seasonal_adjustment_short": "SA"},
    ],
    "permits": [
        {"id": "SEAT1UNIT", "title": "1-Unit Housing Structures Authorized by Building Permits for Seattle-Tacoma-Bellevue, WA (MSA)", "frequency_short": "M", "seasonal_adjustment_short": "NSA"},
        {"id": "SEAT653BPPRIV", "title": "New Private Housing Structures Authorized by Building Permits for Seattle-Tacoma-Bellevue, WA (MSA)", "frequency_short": "M", "seasonal_adjustment_short": "NSA"},
    ],
}


def months(n, end=date(2026, 8, 1)):
    out, y, m = [], end.year, end.month
    for _ in range(n):
        out.append(date(y, m, 1))
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return sorted(out)


def make_fred(unemp_step=-0.02, payroll_growth=0.02, permits_growth=0.0, permit_gap=False, calls=None):
    """FRED stand-in keyed by the series IDs in SEARCH_ROWS."""
    def fred(series_id, observation_start=None, observation_end=None, as_of=None):
        if calls is not None:
            calls.append((series_id, as_of))
        ds = months(96)
        rows = []
        for i, d in enumerate(ds):
            if series_id == "SEAT653UR":
                v = 5.0 + unemp_step * (i - 95)  # latest 5.0, earlier higher if step < 0
            elif series_id == "SEAT653NA":
                v = 1000.0 * (1 + payroll_growth) ** ((i - 95) / 12)
            elif series_id == "SEAT653BPPRIV":
                v = 500.0 * (1 + permits_growth) ** ((i - 95) / 12)
            else:
                raise AssertionError(series_id)
            if permit_gap and series_id == "SEAT653BPPRIV" and i == 90:
                continue
            rows.append({"date": d.isoformat(), "value": round(v, 3)})
        if observation_end:
            rows = [r for r in rows if r["date"] <= observation_end.isoformat()]
        return {"series_id": series_id, "as_of": None, "observations": rows}
    return fred


def fake_search(calls=None):
    def search(text):
        if calls is not None:
            calls.append(text)
        for kind, key in (("unemployment", "Unemployment"), ("payrolls", "All Employees"), ("permits", "Building Permits")):
            if key in text:
                return SEARCH_ROWS[kind]
        return []
    return search


def zip_csv(d, rows=None):
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    rows = rows or ["98101,Seattle,WA", "02108,Boston,MA", "78701,Austin,TX"]
    (p / "us_zip.csv").write_text("zip,city,state\n" + "\n".join(rows) + "\n")


class ZipTests(unittest.TestCase):
    def test_resolve_keeps_leading_zeros_and_accepts_aliases(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "x.csv").write_text("Postal Code,Primary City,State ID\n2108,Boston,ma\n98101,Seattle,WA\n")
            r = ZipResolver(d)
            self.assertEqual((r.resolve("02108").city, r.resolve("02108").state), ("Boston", "MA"))
            self.assertEqual(r.resolve("98101-1234").city, "Seattle")

    def test_unknown_and_malformed(self):
        with tempfile.TemporaryDirectory() as d:
            zip_csv(d)
            r = ZipResolver(d)
            with self.assertRaises(ZipNotFound):
                r.resolve("99999")
            with self.assertRaises(ZipError):
                r.resolve("9810")

    def test_missing_file_is_explained_and_missing_column_named(self):
        with tempfile.TemporaryDirectory() as d:
            r = ZipResolver(Path(d) / "none")
            self.assertFalse(r.available())
            with self.assertRaises(ZipError) as cm:
                r.resolve("98101")
            self.assertIn("zip-fetch", str(cm.exception))
            (Path(d) / "bad.csv").write_text("code,town\n1,2\n")
            with self.assertRaises(ZipError) as cm:
                ZipResolver(d).resolve("98101")
            self.assertIn("no column", str(cm.exception))

    def test_rewrite_replaces_known_zips_only(self):
        with tempfile.TemporaryDirectory() as d:
            zip_csv(d)
            text, places, bad = ZipResolver(d).rewrite("Compare 98101 with 78701 and 12345")
            self.assertEqual(text, "Compare Seattle, WA with Austin, TX and 12345")
            self.assertEqual([p.city for p in places], ["Seattle", "Austin"])
            self.assertEqual(bad, ["12345"])

    def test_rewrite_without_a_file_leaves_text_alone(self):
        with tempfile.TemporaryDirectory() as d:
            text, places, bad = ZipResolver(Path(d) / "none").rewrite("Prices in 98101?")
            self.assertEqual((text, places, bad), ("Prices in 98101?", [], ["98101"]))

    def geonames_zip(self, n=1500):
        lines = [f"US\t{10000 + i}\tTown{i}\tWashington\tWA\tKing\t033\t\t\t47.6\t-122.3\t4" for i in range(n)]
        lines.append("US\t02108\tBoston\tMassachusetts\tMA\tSuffolk\t025\t\t\t42.3\t-71.0\t4")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("US.txt", "\n".join(lines))
            z.writestr("readme.txt", "x")
        return buf.getvalue()

    def test_parse_geonames_and_fetch_command(self):
        rows = parse_geonames(self.geonames_zip())
        self.assertIn(("02108", "Boston", "MA"), rows)
        with tempfile.TemporaryDirectory() as d:
            out = io.StringIO()
            rc = cli.main(["--env-file", d + "/.env", "--db", d + "/data/c.db", "--log", d + "/l.jsonl", "zip-fetch"],
                          open_url=lambda url: io.BytesIO(self.geonames_zip()), out=out)
            self.assertEqual(rc, 0, out.getvalue())
            self.assertEqual(ZipResolver(Path(d) / "data" / "files" / "zip").resolve("02108").city, "Boston")

    def test_fetch_rejects_a_page_that_is_not_a_zip(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ZipError) as cm:
                fetch_zip_csv(Path(d) / "z.csv", open_url=lambda u: io.BytesIO(b"<html>blocked</html>"), say=lambda s: None)
            self.assertIn("not a zip", str(cm.exception))


class FredSearchTests(unittest.TestCase):
    def test_search_parses_and_redacts_key(self):
        seen = []

        def http_get(url, timeout):
            seen.append(url)
            return 200, json.dumps({"seriess": [{"id": "A", "title": "T", "frequency_short": "M",
                                                 "seasonal_adjustment_short": "SA", "observation_end": "2026-08-01", "extra": 1}]})
        rows = FredClient("SECRETKEY", http_get=http_get).search("Unemployment Rate in Seattle")
        self.assertEqual(rows[0]["id"], "A")
        self.assertNotIn("extra", rows[0])
        self.assertIn("search_text=Unemployment+Rate+in+Seattle", seen[0])

        def boom(url, timeout):
            raise OSError("cannot reach " + url)
        with self.assertRaises(FredError) as cm:
            FredClient("SECRETKEY", http_get=boom).search("x")
        self.assertNotIn("SECRETKEY", str(cm.exception))


class FinderTests(unittest.TestCase):
    def test_prefers_seasonally_adjusted_monthly_and_skips_other_metros(self):
        self.assertEqual(pick_series("unemployment", SEARCH_ROWS["unemployment"], SEA)["series_id"], "SEAT653UR")
        self.assertEqual(pick_series("unemployment", SEARCH_ROWS["unemployment"], Location(city="Portland", state="OR"))["series_id"], "PORT")
        self.assertIsNone(pick_series("unemployment", SEARCH_ROWS["unemployment"], Location(city="Tacoma", state="OR")))

    def test_subtype_series_are_skipped(self):
        self.assertEqual(pick_series("permits", SEARCH_ROWS["permits"], SEA)["series_id"], "SEAT653BPPRIV")

    def test_search_result_is_cached_and_overrides_win(self):
        calls = []
        store = RawStore(clock=Clock())
        f = MetroSeriesFinder(store, fake_search(calls))
        f.find("unemployment", SEA)
        f.find("unemployment", SEA)
        self.assertEqual(len(calls), 1)
        pinned = MetroSeriesFinder(store, None, {"seattle-wa": {"permits": "MYID"}})
        self.assertEqual(pinned.find("permits", SEA)["series_id"], "MYID")

    def test_no_match_says_how_to_fix_it(self):
        f = MetroSeriesFinder(RawStore(clock=Clock()), fake_search())
        with self.assertRaises(NoMetroSeries) as cm:
            f.find("unemployment", Location(city="Nowhere", state="WA"))
        self.assertIn("fred_series.yaml", str(cm.exception))
        with self.assertRaises(NoMetroSeries):
            MetroSeriesFinder(RawStore(clock=Clock()), None).find("unemployment", SEA)

    def test_override_file_is_validated(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.yaml"
            p.write_text("seattle-wa:\n  unemployment: ABC\n")
            self.assertEqual(load_overrides(p)["seattle-wa"]["unemployment"], "ABC")
            p.write_text("seattle-wa:\n  jobs: ABC\n")
            with self.assertRaises(SourceError):
                load_overrides(p)
            self.assertEqual(load_overrides(Path(d) / "absent.yaml"), {})


class JobsTests(unittest.TestCase):
    def worker(self, fred, judge=None, search=None):
        store = RawStore(clock=Clock())
        finder = MetroSeriesFinder(store, search or fake_search())
        return JobsWorker(store, fred_source=fred, finder=finder, judge=judge), store

    def test_strong_labor_market_reads_up(self):
        w, _ = self.worker(make_fred(unemp_step=-0.05, payroll_growth=0.02))
        r = w.run(SEA)
        self.assertEqual(r.direction, "up")
        self.assertEqual(r.geography_match, "approximate")
        self.assertAlmostEqual(r.key_metrics["unemployment_change_12m_pp"], -0.6, places=2)
        self.assertAlmostEqual(r.key_metrics["payroll_yoy_pct"], 2.0, places=1)
        self.assertEqual(r.key_metrics["unemployment_series"], "SEAT653UR")
        self.assertLessEqual(r.confidence, 0.6)
        self.assertEqual(len(r.sources), 2)

    def test_weak_labor_market_reads_down(self):
        w, _ = self.worker(make_fred(unemp_step=0.08, payroll_growth=-0.02))
        self.assertEqual(w.run(SEA).direction, "down")

    def test_missing_payroll_series_falls_back_to_unemployment_and_says_so(self):
        def search(text):
            return [] if "All Employees" in text else fake_search()(text)
        w, _ = self.worker(make_fred(unemp_step=-0.05), search=search)
        r = w.run(SEA)
        self.assertIsNone(r.key_metrics["payroll_yoy_pct"])
        self.assertTrue(any("No payroll series" in c for c in r.caveats))

    def test_as_of_run_uses_vintage_and_drops_unpublished_months(self):
        calls = []
        w, _ = self.worker(make_fred(calls=calls))
        r = w.run(SEA, as_of=datetime(2025, 3, 15, tzinfo=timezone.utc))
        self.assertTrue(all(a == date(2025, 3, 15) for _, a in calls))
        # a month is public ~80 days after it starts: Dec 2024 is the newest usable reading on 2025-03-15
        self.assertEqual(r.data_as_of, date(2024, 12, 1))

    def test_no_fred_key_is_a_clear_error(self):
        w, _ = self.worker(None)
        with self.assertRaises(NoMetroSeries):
            w.run(SEA)

    def test_model_judge_is_used_and_capped(self):
        reply = json.dumps({"direction": "up", "confidence": 0.95, "summary": "Jobs are growing.", "caveats": []})
        w, store = self.worker(make_fred(), judge=JsonJudge(StubChat([reply]), store=RawStore(clock=Clock())))
        r = w.run(SEA)
        self.assertEqual(r.confidence, 0.6)
        self.assertEqual(r.direction, "up")


class PermitsTests(unittest.TestCase):
    def worker(self, fred, rule="context", judge=None):
        store = RawStore(clock=Clock())
        return PermitsWorker(store, fred_source=fred, finder=MetroSeriesFinder(store, fake_search()),
                             judge=judge, rule=rule)

    def test_default_takes_no_side_and_never_calls_a_model(self):
        class Boom:
            model_name = "boom"

            def judge(self, *a, **k):
                raise AssertionError("model must not be called in context mode")
        r = self.worker(make_fred(permits_growth=0.5), judge=Boom()).run(SEA)
        self.assertEqual(r.direction, "flat")
        self.assertEqual(r.judged_by, "rule:permits-context")
        self.assertAlmostEqual(r.key_metrics["permits_12m_yoy_pct"], 50.0, delta=0.5)

    def test_demand_and_supply_rules_have_opposite_signs(self):
        self.assertEqual(self.worker(make_fred(permits_growth=0.3), "demand").run(SEA).direction, "up")
        self.assertEqual(self.worker(make_fred(permits_growth=0.3), "supply").run(SEA).direction, "down")
        self.assertEqual(self.worker(make_fred(permits_growth=-0.3), "demand").run(SEA).direction, "down")
        self.assertEqual(self.worker(make_fred(permits_growth=0.0), "demand").run(SEA).direction, "flat")

    def test_gap_in_recent_months_is_not_summed_over(self):
        with self.assertRaises(NoDataAsOf):
            self.worker(make_fred(permit_gap=True)).run(SEA)

    def test_unknown_rule_rejected(self):
        with self.assertRaises(ValueError):
            self.worker(make_fred(), rule="bullish")

    def test_metrics_sum_the_right_windows(self):
        from retrend.fred import Observation
        obs = [Observation(d, 10.0 if i < 12 else 20.0) for i, d in enumerate(months(24))]
        m = permit_metrics(obs, date(2026, 10, 1), "context")
        self.assertEqual((m["permits_12m_total"], m["permits_12m_yoy_pct"]), (240, 100.0))


class WiringTests(unittest.TestCase):
    def test_planner_picks_jobs_and_permits(self):
        self.assertEqual(select_workers("How is the job market in Seattle, WA?")[0], ["jobs"])
        self.assertEqual(select_workers("Are building permits up in Austin, TX?")[0], ["permits"])
        w, _, single = select_workers("Will prices rise in Seattle, WA given unemployment?")
        self.assertIn("jobs", w)
        self.assertIn("housing_prices", w)
        self.assertFalse(single)

    def test_jobs_no_longer_flagged_as_unsupported_news(self):
        p = rule_plan("What is unemployment doing in Seattle, WA?", {})
        self.assertIn("jobs", p.workers)
        self.assertEqual(p.unsupported, [])

    def test_extra_signals_do_not_vote(self):
        self.assertEqual(DEFAULT_WEIGHTS["jobs"], 0.0)
        self.assertEqual(DEFAULT_WEIGHTS["permits"], 0.0)

    def test_cli_builds_opt_in_workers(self):
        store = RawStore(clock=Clock())
        with tempfile.TemporaryDirectory() as d:
            w = cli._build_workers(None, store, None, make_fred(), None, d + "/data/c.db",
                                   with_jobs=True, with_permits=True, fred_search=fake_search(), config_dir=d)
            self.assertIn("jobs", w)
            self.assertIn("permits", w)
            self.assertNotIn("jobs", cli._build_workers(None, store, None, make_fred(), None, d + "/data/c.db"))

    def test_supervisor_shows_jobs_as_context(self):
        store = RawStore(clock=Clock())
        finder = MetroSeriesFinder(store, fake_search())
        jobs = JobsWorker(store, fred_source=make_fred(unemp_step=-0.05), finder=finder, judge=None)
        sup = Supervisor(store, {"jobs": jobs}, expected=["jobs"], max_attempts=1)
        o = sup.run(SEA)
        self.assertEqual(o.direction, "flat")
        self.assertEqual([c.worker for c in o.context], ["jobs"])


class UiZipTests(unittest.TestCase):
    def setUp(self):
        from retrend.ui.runner import QueryRunner
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        cfg = d / "cfg"
        cfg.mkdir()
        for f in ("models.yaml", "test_cities.yaml", "backtest_cities.yaml"):
            shutil.copy(f"config/{f}", cfg / f)
        (d / "data").mkdir()
        self.d = d
        self.store = RawStore(str(d / "data" / "c.db"))
        self.runner = QueryRunner(config_dir=cfg, db_path=str(d / "data" / "c.db"), store=self.store, logger=None,
                                  fred_source=make_fred(unemp_step=-0.05), open_url=opener_for(zillow_csv()),
                                  fred_search=fake_search())

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def ask(self, q):
        events = []
        res = self.runner.handle(q, "baseline", False, events.append)
        return res, events

    def test_zip_in_a_question_becomes_its_city(self):
        zip_csv(self.d / "data" / "files" / "zip")
        res, _ = self.ask("How is the job market in 98101?")
        self.assertEqual([l["city"] for l in res["plan"]["locations"]], ["Seattle"])
        self.assertIn("jobs", res["plan"]["workers"])
        self.assertTrue(any("ZIP 98101 is in Seattle, WA" in n for n in res["notes"]), res["notes"])
        self.assertEqual(res["plan"]["unsupported"], [])

    def test_zip_without_a_file_explains_how_to_get_one(self):
        res, _ = self.ask("Will prices rise in 98101?")
        self.assertTrue(any("zip-fetch" in n for n in res["notes"]), res["notes"])
        self.assertTrue(res["plan"]["unsupported"])

    def test_unknown_zip_is_reported(self):
        zip_csv(self.d / "data" / "files" / "zip")
        res, _ = self.ask("Will prices rise in 12345?")
        self.assertTrue(any("12345 is not in the ZIP file" in n for n in res["notes"]), res["notes"])

    def test_tabs_accept_a_zip(self):
        zip_csv(self.d / "data" / "files" / "zip", ["78701,Austin,TX"])
        v = self.runner.prices_view("", "", "78701")
        self.assertEqual((v["city"], v["state"], v["zip"]), ("Austin", "TX", "78701"))
        with self.assertRaises(ValueError):
            self.runner.prices_view("", "", "99999")
        with self.assertRaises(ValueError):
            self.runner.schools_view("", "", "abc")


if __name__ == "__main__":
    unittest.main()
