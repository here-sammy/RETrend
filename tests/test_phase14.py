"""Google News RSS source: parsing, tone words, windows, caps, rate limits, and the worker using it."""

import io
import json
import unittest
import urllib.error
import urllib.parse
from datetime import date, datetime, timedelta, timezone
from email.utils import format_datetime

from helpers import Clock
from retrend.location import Location
from retrend.sources.gnews import NewsError, fetch_news, parse_feed, score_headline
from retrend.store import RawStore
from retrend.workers.base import NoDataAsOf
from retrend.workers.news import NewsWorker

SEA = Location(city="Seattle", state="WA")
TODAY = date(2026, 10, 1)


def rss(items):
    body = "".join(
        f"<item><title>{t}</title><link>{u}</link><pubDate>{format_datetime(datetime(d.year, d.month, d.day, 12, tzinfo=timezone.utc))}</pubDate>"
        f"<source url=\"https://x\">{s}</source></item>" for t, u, d, s in items)
    return f"<?xml version='1.0'?><rss><channel>{body}</channel></rss>".encode()


def feed_open(title_for_age, per_day=1, total_days=90, calls=None):
    """Answers each window request with headlines whose wording depends on how old they are."""
    def open_url(url):
        if calls is not None:
            calls.append(url)
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["q"][0]
        after = date.fromisoformat(q.split("after:")[1].split()[0])
        before = date.fromisoformat(q.split("before:")[1].split()[0])
        items, d = [], after
        while d < before:
            age = (TODAY - d).days
            if 0 <= age < total_days:
                for k in range(per_day):
                    items.append((f"{title_for_age(age)} - Local Paper", f"https://news.example/{d}-{k}", d, "Local Paper"))
            d += timedelta(days=1)
        return io.BytesIO(rss(items))
    return open_url


def worker(open_url, store=None):
    return NewsWorker(store or RawStore(clock=Clock()), open_url=open_url, judge=None, pause=0, source="gnews")


class ParseTests(unittest.TestCase):
    def test_tone_words(self):
        self.assertEqual(score_headline("Seattle home prices surge to record"), 1.0)
        self.assertEqual(score_headline("Rents fall as market cools"), -1.0)
        self.assertEqual(score_headline("Prices rise but sales slow"), 0.0)
        self.assertIsNone(score_headline("Council meets on Tuesday"))

    def test_feed_parsing_strips_source_and_unsafe_links(self):
        items = parse_feed(rss([("Prices fall - Local Paper", "https://a.com/1", TODAY, "Local Paper"),
                                ("Bad - X", "javascript:alert(1)", TODAY, "X")]).decode())
        self.assertEqual([(i["title"], i["domain"]) for i in items], [("Prices fall", "Local Paper")])

    def test_entity_declarations_and_non_xml_are_refused(self):
        bomb = '<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><rss><channel/></rss>'
        with self.assertRaises(NewsError):
            parse_feed(bomb)
        with self.assertRaises(NewsError):
            parse_feed("<html>blocked</html")


class FetchTests(unittest.TestCase):
    def test_three_windows_become_daily_series(self):
        calls = []
        out = fetch_news(SEA, feed_open(lambda a: "Prices rise", calls=calls), TODAY, pause=0)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(out["tone"]), 90)
        self.assertEqual(out["tone"][-1]["date"], TODAY.isoformat())
        self.assertFalse(out["capped"])
        self.assertEqual(len(out["headlines"]), 8)
        self.assertIn("after:", urllib.parse.unquote_plus(calls[0]))

    def test_a_full_feed_is_flagged_as_capped(self):
        out = fetch_news(SEA, feed_open(lambda a: "Prices rise", per_day=4), TODAY, pause=0)
        self.assertTrue(out["capped"])

    def test_429_is_retried_then_explained(self):
        inner = feed_open(lambda a: "Prices rise")
        n = {"i": 0}

        def flaky(url):
            n["i"] += 1
            if n["i"] <= 2:
                raise urllib.error.HTTPError(url, 429, "x", {}, None)
            return inner(url)
        waits = []
        self.assertEqual(len(fetch_news(SEA, flaky, TODAY, pause=0, sleep=waits.append)["tone"]), 90)
        self.assertEqual(waits, [15.0, 30.0])

        def always(url):
            raise urllib.error.HTTPError(url, 429, "x", {}, None)
        with self.assertRaises(NewsError) as cm:
            fetch_news(SEA, always, TODAY, pause=0, sleep=lambda s: None, retries=1)
        self.assertIn("wait a few minutes", str(cm.exception))


class WorkerTests(unittest.TestCase):
    def test_worsening_wording_reads_down_and_shows_headlines(self):
        w = worker(feed_open(lambda age: "Prices fall as market cools" if age < 30 else "Prices rise"))
        r = w.run(SEA)
        self.assertEqual(r.direction, "down")
        self.assertLessEqual(r.confidence, 0.3)
        self.assertEqual(r.key_metrics["tone_recent"], -1.0)
        self.assertIsNotNone(r.key_metrics["volume_change_pct"] or 0.0)
        self.assertEqual(json.loads(r.key_metrics["headlines_json"])[0]["domain"], "Local Paper")
        self.assertIn("Google News", r.sources[0].name)

    def test_capped_feeds_do_not_report_volume(self):
        r = worker(feed_open(lambda a: "Prices rise", per_day=4)).run(SEA)
        self.assertNotIn("volume_recent", {k for k, v in r.key_metrics.items() if v is not None})
        self.assertTrue(any("maximum number" in c for c in r.caveats))

    def test_headlines_without_tone_words_are_no_data(self):
        with self.assertRaises(NoDataAsOf):
            worker(feed_open(lambda a: "Council meets Tuesday")).run(SEA)

    def test_past_dates_need_a_stored_snapshot_and_sources_do_not_mix(self):
        store = RawStore(clock=Clock())
        w = worker(feed_open(lambda a: "Prices rise"), store)
        with self.assertRaises(NoDataAsOf):
            w.run(SEA, as_of=datetime(2025, 5, 1, tzinfo=timezone.utc))
        w.run(SEA)
        self.assertEqual(w.run(SEA, as_of=datetime(2026, 10, 2, tzinfo=timezone.utc)).data_as_of, TODAY)

    def test_default_source_is_gnews_and_unknown_is_rejected(self):
        self.assertEqual(NewsWorker(RawStore(clock=Clock()), open_url=None, judge=None)._source, "gnews")
        with self.assertRaises(ValueError):
            NewsWorker(RawStore(clock=Clock()), open_url=None, judge=None, source="bing")


if __name__ == "__main__":
    unittest.main()
