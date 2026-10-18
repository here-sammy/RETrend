"""Phase 6: change detection, ranking, rendering (incl. escaping), monthly/compare commands."""

import csv
import io
import json
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from helpers import T0, Clock
from retrend import cli
from retrend import report as rp
from retrend.store import RawStore
from retrend.supervisor import Supervisor
from test_phase3 import SEATTLE, Scripted, report
from test_phase4 import NAMES, fake_fred, opener_for, zillow_csv

AUSTIN_LOC = __import__("retrend.location", fromlist=["Location"]).Location(city="Austin", state="TX")


def outlook_for(store, loc, directions, conf=0.8, narrative=None):
    """Build a real Outlook through the supervisor with scripted workers."""
    workers = {n: Scripted(store, n, report(n, d, conf, loc=loc)) for n, d in directions.items()}
    sup = Supervisor(store, workers, max_attempts=1)
    o = sup.run(loc)
    if narrative:
        o.narrative = narrative
    return o


class DiffTests(unittest.TestCase):
    def setUp(self):
        self.store = RawStore(clock=Clock())

    def tearDown(self):
        self.store.close()

    def snap(self, o, run_id):
        rp.save_snapshot(self.store, o, "m", run_id)

    def test_first_run_is_new(self):
        o = outlook_for(self.store, SEATTLE, {"housing_prices": "up", "supply": "up"})
        self.assertEqual([c.kind for c in rp.diff(o, None)], ["new"])

    def test_direction_change_is_high_severity(self):
        a = outlook_for(self.store, SEATTLE, {"housing_prices": "up", "supply": "up"})
        self.snap(a, "2026-09")
        b = outlook_for(self.store, SEATTLE, {"housing_prices": "down", "supply": "down"})
        prev = rp.previous_snapshot(self.store, SEATTLE.key, "m", "2026-10")
        ch = rp.diff(b, prev)
        self.assertEqual(ch[0].kind, "direction_change")
        self.assertEqual(ch[0].severity, "high")
        self.assertIn("up -> down", ch[0].detail)

    def test_small_moves_not_flagged_but_big_ones_are(self):
        a = outlook_for(self.store, SEATTLE, {"housing_prices": "up", "supply": "flat"})   # 0.5
        self.snap(a, "2026-09")
        prev = rp.previous_snapshot(self.store, SEATTLE.key, "m", "2026-10")
        same = outlook_for(self.store, SEATTLE, {"housing_prices": "up", "supply": "flat"})
        self.assertEqual(rp.diff(same, prev), [])
        bigger = outlook_for(self.store, SEATTLE, {"housing_prices": "up", "supply": "up"})  # 1.0
        kinds = [c.kind for c in rp.diff(bigger, prev)]
        self.assertIn("score_move", kinds)

    def test_worker_lost_and_back(self):
        full = outlook_for(self.store, SEATTLE, {"housing_prices": "up", "supply": "up"})
        self.snap(full, "2026-09")
        w = {"housing_prices": Scripted(self.store, "housing_prices", report("housing_prices", "up")),
             "supply": Scripted(self.store, "supply", OSError("down"))}
        lost = Supervisor(self.store, w, max_attempts=1).run(SEATTLE)
        kinds = [c.kind for c in rp.diff(lost, rp.previous_snapshot(self.store, SEATTLE.key, "m", "2026-10"))]
        self.assertIn("worker_lost", kinds)
        self.snap(lost, "2026-10")
        back = rp.diff(full, rp.previous_snapshot(self.store, SEATTLE.key, "m", "2026-11"))
        self.assertIn("worker_back", [c.kind for c in back])

    def test_same_month_rerun_compares_with_previous_month_not_itself(self):
        a = outlook_for(self.store, SEATTLE, {"housing_prices": "up", "supply": "up"})
        self.snap(a, "2026-09")
        b = outlook_for(self.store, SEATTLE, {"housing_prices": "down", "supply": "down"})
        self.snap(b, "2026-10")
        prev = rp.previous_snapshot(self.store, SEATTLE.key, "m", "2026-10")  # a rerun in October
        self.assertEqual(prev["run_id"], "2026-09")
        self.assertIsNone(rp.previous_snapshot(self.store, SEATTLE.key, "other-model", "2026-10"))


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.store = RawStore(clock=Clock())
        up = outlook_for(self.store, SEATTLE, {"housing_prices": "up", "supply": "up"})
        down = outlook_for(self.store, AUSTIN_LOC, {"housing_prices": "down", "supply": "down"},
                           narrative="<script>alert('x')</script> & more")
        flat = outlook_for(self.store, AUSTIN_LOC, {"housing_prices": "flat", "supply": "flat"})
        self.rows = [
            rp.Row("austin-tx", "Austin", "TX", down, changes=[rp.Change("direction_change", "high", "up -> down")]),
            rp.Row("seattle-wa", "Seattle", "WA", up),
            rp.Row("flat-tx", "Flat", "TX", flat),
            rp.Row("boom-xx", "Boom", "XX", None, error="OSError: boom <b>"),
        ]

    def tearDown(self):
        self.store.close()

    def test_rank_orders_by_score_failures_last(self):
        self.assertEqual([r.key for r in rp.rank(self.rows)], ["seattle-wa", "flat-tx", "austin-tx", "boom-xx"])

    def test_summary(self):
        s = rp.summarize(self.rows)
        self.assertEqual(s["counts"], {"up": 1, "flat": 1, "down": 1})
        self.assertEqual((s["n_failed"], s["direction_changes"]), (1, ["austin-tx"]))

    def test_html_escapes_everything_untrusted(self):
        page = rp.to_html(self.rows, "2026-10", "m<i>", T0)
        self.assertNotIn("<script>", page)
        self.assertIn("&lt;script&gt;", page)
        self.assertNotIn("<b>", page)
        self.assertIn("m&lt;i&gt;", page)
        self.assertIn("not financial advice", page)
        self.assertIn("prefers-color-scheme", page)
        self.assertIn("viewport", page)

    def test_csv_json_markdown_parse_and_agree(self):
        rows = list(csv.reader(io.StringIO(rp.to_csv(self.rows))))
        self.assertEqual(rows[0][:3], ["rank", "location", "direction"])
        self.assertEqual([r[1] for r in rows[1:]], ["seattle-wa", "flat-tx", "austin-tx", "boom-xx"])
        self.assertEqual(rows[-1][2], "ERROR")
        data = json.loads(rp.to_json(self.rows, "2026-10", "m", T0))
        self.assertEqual([x["key"] for x in data["locations"]], [r[1] for r in rows[1:]])
        self.assertIn("evidence_note", data)
        md = rp.to_markdown(self.rows, "2026-10", "m")
        self.assertIn("Direction changed since last run:** austin-tx", md)
        self.assertIn("| 1 | Seattle, WA |", md)


class CliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        d = Path(self._tmp.name)
        cfg = d / "cfg"
        cfg.mkdir()
        shutil.copy("config/models.yaml", cfg / "models.yaml")
        (cfg / "cities.yaml").write_text("cities:\n" + "".join(
            f"  - city: {c}\n    state: {s}\n" for c, s in NAMES[:4]) +
            "  - city: Nowhereville\n    state: ZZ\n")
        self.base = ["--env-file", str(d / ".env"), "--config-dir", str(cfg), "--db", str(d / "c.db"),
                     "--log", str(d / "l.jsonl")]
        self.d = d

    def tearDown(self):
        self._tmp.cleanup()

    def run_cmd(self, *args):
        out = io.StringIO()
        rc = cli.main(self.base + list(args), fred_source=fake_fred,
                      open_url=opener_for(zillow_csv()), out=out)
        return rc, out.getvalue()

    def test_monthly_writes_reports_survives_a_bad_location_and_flags_changes(self):
        rc, text = self.run_cmd("monthly", "--registry", "cities.yaml", "--run-id", "2026-09")
        self.assertEqual(rc, 0, text)
        self.assertIn("Report written", text)
        out = self.d / "monthly" / "2026-09"
        for f in ("report.html", "report.md", "report.json", "report.csv"):
            self.assertTrue((out / f).exists(), f)
        data = json.loads((out / "report.json").read_text())
        self.assertEqual(len(data["locations"]), 5)
        # Nowhereville has no Zillow row: the supervisor degrades to flat/zero, it does not crash
        nowhere = next(x for x in data["locations"] if x["key"] == "nowhereville-zz")
        self.assertEqual(nowhere["outlook"]["direction"], "flat")
        self.assertTrue(nowhere["outlook"]["missing_workers"])
        self.assertTrue(all(c["kind"] == "new" for x in data["locations"] for c in x["changes"]))

        rc, text = self.run_cmd("monthly", "--registry", "cities.yaml", "--run-id", "2026-10")
        self.assertEqual(rc, 0, text)
        data2 = json.loads((self.d / "monthly" / "2026-10" / "report.json").read_text())
        self.assertFalse(any(c["kind"] == "new" for x in data2["locations"] for c in x["changes"]))

    def test_compare_saves_no_snapshots(self):
        rc, text = self.run_cmd("compare", "--registry", "cities.yaml", "--locations", "seattle-wa,austin-tx",
                                "--run-id", "2026-09")
        self.assertEqual(rc, 0, text)
        rc, text = self.run_cmd("monthly", "--registry", "cities.yaml", "--locations", "seattle-wa",
                                "--run-id", "2026-09")
        data = json.loads((self.d / "monthly" / "2026-09" / "report.json").read_text())
        self.assertEqual(data["locations"][0]["changes"][0]["kind"], "new")  # compare left nothing behind

    def test_unknown_location_and_model_are_reported(self):
        rc, text = self.run_cmd("monthly", "--registry", "cities.yaml", "--locations", "nope-xx")
        self.assertEqual(rc, 2)
        self.assertIn("Unknown location", text)
        rc, text = self.run_cmd("monthly", "--registry", "cities.yaml", "--models", "nonesuch")
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
