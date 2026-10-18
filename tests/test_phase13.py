"""News worker (GDELT): fetch parsing, metrics, point-in-time refusal, wiring."""

import io
import json
import shutil
import tempfile
import unittest
import urllib.error
import urllib.parse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from helpers import Clock
from retrend import cli
from retrend.location import Location
from retrend.sources.gdelt import NewsError, build_query, fetch_news
from retrend.store import RawStore
from retrend.ui.planner import select_workers
from retrend.workers.base import NoDataAsOf
from retrend.workers.news import NewsWorker
from retrend.workers.news_metrics import compute_metrics
from test_phase4 import opener_for, zillow_csv

SEA = Location(city="Seattle", state="WA")
TODAY = date(2026, 10, 1)


def day_points(fn, n=90, end=TODAY):
    return [{"date": (end - timedelta(days=n - 1 - i)).isoformat(), "value": fn(i)} for i in range(n)]


def gdelt_open(tone_fn=lambda i: -1.0, vol_fn=lambda i: 0.5, n=90, calls=None, articles=None, bad=None):
    def fmt(points):
        return [{"date": p["date"].replace("-", "") + "T000000Z", "value": p["value"]} for p in points]

    def open_url(url):
        if calls is not None:
            calls.append(url)
        mode = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["mode"][0]
        if bad:
            return io.BytesIO(bad)
        if mode == "timelinetone":
            body = {"timeline": [{"series": "Average Tone", "data": fmt(day_points(tone_fn, n))}]}
        elif mode == "timelinevol":
            body = {"timeline": [{"series": "Volume", "data": fmt(day_points(vol_fn, n))}]}
        else:
            body = {"articles": articles if articles is not None else [
                {"url": "https://example.com/a", "title": "Seattle  rents   fall", "seendate": "20260930T120000Z", "domain": "example.com"},
                {"url": "javascript:alert(1)", "title": "bad link", "seendate": "20260930T120000Z", "domain": "x"}]}
        return io.BytesIO(json.dumps(body).encode())
    return open_url


def worker(open_url, judge=None):
    return NewsWorker(RawStore(clock=Clock()), open_url=open_url, judge=judge, pause=0, source="gdelt")


class FetchTests(unittest.TestCase):
    def test_parses_timelines_and_headlines_and_drops_unsafe_links(self):
        calls = []
        out = fetch_news(SEA, gdelt_open(calls=calls), pause=0)
        self.assertEqual(len(out["tone"]), 90)
        self.assertEqual(out["tone"][0]["date"], (TODAY - timedelta(days=89)).isoformat())
        self.assertEqual([h["title"] for h in out["headlines"]], ["Seattle rents fall"])
        self.assertEqual(len(calls), 3)
        self.assertIn("sourcecountry:US", urllib.parse.unquote_plus(calls[0]))

    def test_query_names_the_city(self):
        self.assertIn('"Seattle"', build_query(SEA))

    def test_non_json_reply_is_explained(self):
        with self.assertRaises(NewsError) as cm:
            fetch_news(SEA, gdelt_open(bad=b"The specified phrase is too short."), pause=0)
        self.assertIn("phrase is too short", str(cm.exception))


class RateLimitTests(unittest.TestCase):
    def flaky(self, fails, kind):
        inner = gdelt_open()
        state = {"n": 0}

        def open_url(url):
            state["n"] += 1
            if state["n"] <= fails:
                if kind == "http":
                    raise urllib.error.HTTPError(url, 429, "Too Many Requests", {}, None)
                return io.BytesIO(b"Please limit requests to one every 5 seconds or contact kleinman@gdeltproject.org")
            return inner(url)
        return open_url, state

    def test_429_is_retried_with_a_growing_wait(self):
        for kind in ("http", "text"):
            open_url, state = self.flaky(2, kind)
            waits = []
            out = fetch_news(SEA, open_url, pause=0, sleep=waits.append)
            self.assertEqual(len(out["tone"]), 90, kind)
            self.assertEqual(waits, [15.0, 30.0], kind)

    def test_gives_up_with_advice_after_the_retries(self):
        open_url, _ = self.flaky(99, "http")
        with self.assertRaises(NewsError) as cm:
            fetch_news(SEA, open_url, pause=0, sleep=lambda s: None, retries=2)
        self.assertIn("too many requests", str(cm.exception))
        self.assertIn("wait a few minutes", str(cm.exception))

    def test_requests_are_spaced_across_calls(self):
        import retrend.sources.gdelt as g
        g._last_request = 0.0
        waits = []
        fetch_news(SEA, gdelt_open(), pause=6.0, sleep=waits.append)
        self.assertEqual(len(waits), 2)  # the first call needs no wait; the next two each wait ~6s
        self.assertTrue(all(5.0 < w <= 6.0 for w in waits), waits)

    def test_other_http_errors_are_not_retried(self):
        def open_url(url):
            raise urllib.error.HTTPError(url, 500, "boom", {}, None)
        with self.assertRaises(NewsError) as cm:
            fetch_news(SEA, open_url, pause=0, sleep=lambda s: None)
        self.assertIn("HTTP 500", str(cm.exception))


class MetricsTests(unittest.TestCase):
    def test_tone_change_compares_last_30_days_with_the_60_before(self):
        tone = day_points(lambda i: -2.0 if i < 60 else -0.5)
        m = compute_metrics(tone, day_points(lambda i: 1.0 if i < 60 else 2.0), TODAY)
        self.assertEqual(m["tone_change"], 1.5)
        self.assertEqual(m["baseline_direction"], "up")
        self.assertEqual(m["volume_change_pct"], 100.0)

    def test_too_little_data_gives_none(self):
        self.assertIsNone(compute_metrics(day_points(lambda i: 0.0, n=20), [], TODAY))
        self.assertIsNone(compute_metrics([], [], TODAY))


class WorkerTests(unittest.TestCase):
    def test_worsening_tone_reads_down_with_low_confidence_and_headlines(self):
        r = worker(gdelt_open(tone_fn=lambda i: -1.0 if i < 60 else -2.5)).run(SEA)
        self.assertEqual(r.direction, "down")
        self.assertLessEqual(r.confidence, 0.3)
        self.assertEqual(r.geography_match, "approximate")
        self.assertEqual(json.loads(r.key_metrics["headlines_json"])[0]["title"], "Seattle rents fall")

    def test_result_is_cached_for_a_day(self):
        calls = []
        w = worker(gdelt_open(calls=calls))
        w.run(SEA)
        w.run(SEA)
        self.assertEqual(len(calls), 3)

    def test_sparse_coverage_is_no_data(self):
        with self.assertRaises(NoDataAsOf):
            worker(gdelt_open(n=15)).run(SEA)

    def test_past_dates_are_refused_unless_a_snapshot_exists(self):
        w = worker(gdelt_open())
        with self.assertRaises(NoDataAsOf):
            w.run(SEA, as_of=datetime(2025, 5, 1, tzinfo=timezone.utc))
        w.run(SEA)  # stores a snapshot dated clock-now (2026-10-01)
        r = w.run(SEA, as_of=datetime(2026, 10, 2, tzinfo=timezone.utc))
        self.assertEqual(r.data_as_of, TODAY)
        with self.assertRaises(NoDataAsOf):  # a snapshot taken later is not visible earlier
            w.run(SEA, as_of=datetime(2026, 9, 1, tzinfo=timezone.utc))


class WiringTests(unittest.TestCase):
    def test_planner_picks_news(self):
        self.assertEqual(select_workers("What is the news about housing in Austin, TX?")[0], ["news"])
        w, _, single = select_workers("Will prices rise in Austin, TX? Any recent headlines?")
        self.assertIn("news", w)
        self.assertFalse(single)

    def test_cli_builds_opt_in_worker(self):
        store = RawStore(clock=Clock())
        with tempfile.TemporaryDirectory() as d:
            self.assertIn("news", cli._build_workers(None, store, None, None, None, d + "/data/c.db", with_news=True))
            self.assertNotIn("news", cli._build_workers(None, store, None, None, None, d + "/data/c.db"))

    def test_question_shows_headlines_in_the_screen_result(self):
        from retrend.ui.runner import QueryRunner
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            cfg = d / "cfg"
            cfg.mkdir()
            for f in ("models.yaml", "test_cities.yaml", "backtest_cities.yaml"):
                shutil.copy(f"config/{f}", cfg / f)
            (d / "data").mkdir()
            store = RawStore(str(d / "data" / "c.db"))
            try:
                runner = QueryRunner(config_dir=cfg, db_path=str(d / "data" / "c.db"), store=store, logger=None,
                                     fred_source=None, open_url=gdelt_open())
                import os
                import retrend.sources.gdelt as g
                g.PAUSE = 0
                os.environ["RETREND_NEWS_SOURCE"] = "gdelt"
                res = runner.handle("What is the housing news in Seattle, WA?", "baseline", False, lambda e: None)
            finally:
                store.close()
            os.environ.pop("RETREND_NEWS_SOURCE", None)
        self.assertIn("news", res["plan"]["workers"])
        self.assertEqual(res["headlines"]["Seattle, WA"][0]["title"], "Seattle rents fall")


if __name__ == "__main__":
    unittest.main()
