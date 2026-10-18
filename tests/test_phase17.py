"""Days on market in the housing view, the sample-question table, descriptive answers."""
import unittest
from datetime import date, datetime, timezone

from retrend.location import Location
from retrend.outlook import Outlook
from retrend.ui import answer as A
from retrend.ui.planner import KNOWN_WORKERS, Plan
from retrend.ui.prices import price_view
from retrend.ui.server import EXAMPLES


def zhvi():
    return [(f"{2023 + (i // 12)}-{i % 12 + 1:02d}-28", 400000.0 + i * 1000) for i in range(30)]


class DaysOnMarketTests(unittest.TestCase):
    loc = Location(city="Seattle", state="WA")
    bundle = {"zhvi": zhvi(), "zori": None, "region": "Seattle", "geo": "city", "quality": "exact"}

    def test_series_and_tiles_when_redfin_has_data(self):
        pts = [(d, 30.0 + i) for i, (d, _) in enumerate(zhvi())]
        v = price_view(self.loc, self.bundle, {"geo": "metro", "region": "Seattle metro", "points": pts})
        self.assertIn("median_dom", v["series"])
        self.assertEqual(v["series"]["median_dom"]["unit"], "days")
        self.assertEqual(v["stats"]["median_dom"], 59)
        self.assertEqual(v["stats"]["median_dom_change_1y_days"], 12)
        self.assertEqual(v["notes"], [])

    def test_missing_redfin_data_is_explained_not_fatal(self):
        v = price_view(self.loc, self.bundle, None)
        self.assertNotIn("median_dom", v["series"])
        self.assertTrue(any("days on market" in n for n in v["notes"]))


class DaysToPendingTests(unittest.TestCase):
    def test_zillow_series_and_tile_data(self):
        loc = Location(city="Seattle", state="WA")
        b = dict(DaysOnMarketTests.bundle)
        b["dtp"] = {"region": "Seattle, WA", "points": [(d, 20.0 + i) for i, (d, _) in enumerate(zhvi())]}
        v = price_view(loc, b, None)
        self.assertEqual(v["series"]["dtp"]["unit"], "days")
        self.assertEqual(v["stats"]["dtp"], 49)
        self.assertEqual(v["stats"]["dtp_change_1y_days"], 12)


class HistoryTests(unittest.TestCase):
    def test_five_unique_most_recent_questions(self):
        import tempfile
        from pathlib import Path
        from retrend.store import RawStore
        from retrend.ui.runner import HISTORY_SOURCE, QueryRunner
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            (d / "data").mkdir()
            store = RawStore(str(d / "data" / "c.db"))
            try:
                r = QueryRunner(config_dir="config", db_path=str(d / "data" / "c.db"), store=store, logger=None,
                                fred_source=None)
                qs = ["a?", "b?", "A?  ", "c?", "d?", "e?", "f?", "g?", "b?"]
                for i, q in enumerate(qs):
                    store.put(HISTORY_SOURCE, f"20260101T0000{i:02d}", {"at": f"2026-01-01T00:00:{i:02d}", "query": q})
                got = [h["query"] for h in r.history()]
            finally:
                store.close()
        self.assertEqual(got, ["b?", "g?", "f?", "e?", "d?"])


class ExampleTableTests(unittest.TestCase):
    def test_rows_have_signal_and_question_and_cover_every_worker(self):
        self.assertTrue(all(set(e) == {"signal", "question"} for e in EXAMPLES))
        text = " ".join(e["signal"].lower() for e in EXAMPLES)
        for word in ("prices", "supply", "mortgage", "school", "jobs", "permits", "news", "zip"):
            self.assertIn(word, text)


class DescriptiveAnswerTests(unittest.TestCase):
    def outlook(self):
        def c(w, d, s):
            return {"worker": w, "direction": d, "reported_confidence": 0.6, "effective_confidence": 0.6,
                    "base_weight": 0.5, "freshness_factor": 1, "geography_factor": 1, "contribution": 0.1,
                    "data_as_of": date(2026, 9, 1), "summary": s, "sources": []}
        return Outlook.model_validate({
            "location": {"city": "Seattle", "state": "WA"}, "direction": "up", "score": 0.4, "confidence": 0.5,
            "uncertainty": "moderate", "drivers": [c("housing_prices", "up", "Home values rose 4.2% in a year")],
            "dissent": [c("supply", "down", "Inventory is rising")],
            "context": [c("jobs", "up", "Payrolls grew 1.1%")],
            "missing_workers": {"permits": "no data"}, "findings": [], "narrative": "Seattle leans up.",
            "weights": {}, "generated_at": datetime.now(timezone.utc)})

    def test_template_discusses_each_signal_then_the_trend(self):
        plan = Plan(intent="outlook", workers=list(KNOWN_WORKERS))
        text = A.template_answer(plan, [self.outlook()])
        for part in ("Home prices and rents (up)", "Supply and demand (down)", "Local jobs (up)",
                     "context only", "Building permits: not available", "Overall trend for Seattle, WA: up"):
            self.assertIn(part, text)
        self.assertLess(text.index("Home prices"), text.index("Overall trend"))


if __name__ == "__main__":
    unittest.main()
