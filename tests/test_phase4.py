"""Phase 4: metrics, truth, backtest (incl. no-look-ahead), ablation, ledger, CLI."""

import gzip
import io
import json
import random
import shutil
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from helpers import T0, Clock
from retrend import cli
from retrend.eval import backtest as bt
from retrend.eval import metrics as M
from retrend.eval.ledger import Ledger
from retrend.eval.truth import ZhviTruth
from retrend.location import Location
from retrend.sources.zillow import ZillowLoader
from retrend.store import RawStore
from retrend.supervisor import Supervisor
from retrend.workers.housing import HousingPriceWorker
from retrend.workers.mortgage import MortgageRateWorker
from retrend.workers.supply import SupplyWorker
from retrend.sources.redfin import RedfinLoader
from retrend.workers.timeseries import Point
from test_phase2 import month_ends

NAMES = [("Seattle", "WA"), ("Austin", "TX"), ("Phoenix", "AZ"), ("Denver", "CO"),
         ("Boston", "MA"), ("Miami", "FL"), ("Dallas", "TX"), ("Atlanta", "GA"),
         ("Portland", "OR"), ("Chicago", "IL"), ("Houston", "TX"), ("Tampa", "FL")]
LOCS = [Location(city=c, state=s) for c, s in NAMES]
N_MONTHS = 150
LAST = date(2026, 8, 31)


def values_for(seed, n=N_MONTHS):
    """Trending series with persistent growth regimes (so momentum has some signal)."""
    rng = random.Random(seed)
    g, v, out = 0.003, 300000.0, []
    for _ in range(n):
        g = 0.85 * g + rng.gauss(0.0005, 0.0025)
        v *= 1 + g
        out.append(v)
    return out


def zillow_csv(changed_after=None, n=N_MONTHS):
    days = month_ends(n, LAST)
    head = ["RegionID", "SizeRank", "RegionName", "RegionType", "StateName"] + [d.isoformat() for d in days]
    rows = [head]
    for i, (c, s) in enumerate(NAMES):
        vals = values_for(i, n)
        if changed_after:
            vals = [v * (3 if d > changed_after else 1) for v, d in zip(vals, days)]
        rows.append([str(i), str(i + 1), f'"{c}"', "city", s] + [f"{v:.0f}" for v in vals])
    return "\n".join(",".join(r) for r in rows) + "\n"


def opener_for(zhvi_csv):
    class Resp(io.BytesIO):
        pass

    def open_url(url):
        if "zhvi" in url.lower():
            return Resp(zhvi_csv.encode())
        raise OSError("404 (no such file in this fixture)")
    return open_url


def fake_fred(series_id, observation_start=None, observation_end=None, as_of=None):
    end = observation_end or date(2026, 10, 1)
    start = observation_start or end - timedelta(days=365 * 5)
    obs, d, k = [], start, 0
    while d <= end:
        obs.append({"date": d.isoformat(), "value": round(4.0 + 0.5 * ((k // 20) % 7), 2)})
        d += timedelta(weeks=1)
        k += 1
    return {"series_id": series_id, "as_of": None, "observations": obs}


class Env:
    """A store plus the three real workers over fixture data (supply file absent on purpose)."""

    def __init__(self, tmp, zhvi=None, clock=None):
        self.store = RawStore(clock=clock or Clock())
        self.tmp = Path(tmp)
        o = opener_for(zhvi if zhvi is not None else zillow_csv())
        zl = ZillowLoader(self.tmp / "z", open_url=o, clock=self.store.clock)
        self.housing = HousingPriceWorker(self.store, loader=zl, judge=None, geos=("city",))
        self.workers = {
            "mortgage_rates": MortgageRateWorker(self.store, fred_source=fake_fred, judge=None),
            "housing_prices": self.housing,
            "supply": SupplyWorker(self.store, loader=RedfinLoader(self.tmp / "r", open_url=o), judge=None),
        }
        self.truth = ZhviTruth(self.housing.zhvi_series)

    def close(self):
        self.store.close()


class MetricTests(unittest.TestCase):
    def test_label_and_threshold(self):
        self.assertEqual(M.threshold_for(12), 2.0)
        self.assertEqual(M.threshold_for(6), 1.0)
        self.assertEqual([M.label(x, 2) for x in (5, 2, 0, -2, -3)], ["up", "flat", "flat", "flat", "down"])

    def test_spearman_known_values(self):
        self.assertEqual(M.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertEqual(M.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertIsNone(M.spearman([1, 1, 1], [1, 2, 3]))
        self.assertIsNone(M.spearman([1, 2], [1, 2]))
        self.assertEqual(M.spearman([1, 2, 2, 3], [1, 2, 2, 3]), 1.0)  # ties

    def test_balanced_accuracy_defeats_always_up(self):
        truth = ["up"] * 8 + ["down"] * 2
        self.assertEqual(M.accuracy(["up"] * 10, truth), 0.8)
        self.assertEqual(M.balanced_accuracy(["up"] * 10, truth), 0.5)

    def test_spread_calibration_confusion(self):
        s = M.direction_spread(["up", "up", "down"], [4.0, 6.0, -2.0])
        self.assertEqual((s["mean_fwd_when_up"], s["mean_fwd_when_down"], s["spread"]), (5.0, -2.0, 7.0))
        cal = M.calibration([0.1, 0.2, 0.9], [True, False, True])
        self.assertEqual(cal[0]["n"], 2)
        self.assertEqual(cal[3]["accuracy"], 1.0)
        self.assertEqual(M.confusion(["up"], ["down"])["down"]["up"], 1)

    def test_cross_section_needs_enough_per_date(self):
        g = [1] * 8 + [2] * 3
        x = list(range(8)) + [1, 2, 3]
        self.assertEqual(M.mean_spearman_by_group(g, x, x), (1.0, 1))

    def test_block_bootstrap_deterministic_and_degenerate(self):
        self.assertEqual(M.block_bootstrap_ci([0.1] * 20, 4), (0.1, 0.1, 0.1))
        v = [0.1, -0.1, 0.2, 0.0, 0.3, -0.2, 0.1, 0.0, 0.1, 0.2]
        self.assertEqual(M.block_bootstrap_ci(v, 3), M.block_bootstrap_ci(v, 3))
        m, lo, hi = M.block_bootstrap_ci(v, 3)
        self.assertLessEqual(lo, m)
        self.assertLessEqual(m, hi)


class TruthTests(unittest.TestCase):
    def test_origin_lag_forward_and_trailing(self):
        pts = [Point(d, 100.0 * 1.01 ** k) for k, d in enumerate(month_ends(40, LAST))]
        t = ZhviTruth(lambda loc: pts)
        loc = LOCS[0]
        o = t.origin(loc, date(2026, 8, 31))
        self.assertEqual(o.day, date(2026, 7, 31))       # Aug 31 not yet published (20-day lag)
        self.assertEqual(t.origin(loc, date(2026, 9, 20)).day, date(2026, 8, 31))
        self.assertIsNone(t.origin(loc, date(2000, 1, 1)))
        self.assertEqual(t.forward_pct(loc, o, 1), 1.0)
        self.assertIsNone(t.forward_pct(loc, o, 2))      # beyond the data
        self.assertEqual(t.trailing_pct(loc, o), 12.7)


class DateTests(unittest.TestCase):
    def test_month_end_dates(self):
        d = bt.month_end_dates(date(2024, 1, 15), date(2024, 4, 30))
        self.assertEqual(d, [date(2024, 1, 31), date(2024, 2, 29), date(2024, 3, 31), date(2024, 4, 30)])
        self.assertEqual(len(bt.month_end_dates(date(2020, 1, 1), date(2020, 12, 31), 3)), 4)


class BacktestTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.env = Env(self._tmp.name)

    def tearDown(self):
        self.env.close()
        self._tmp.cleanup()

    def run_bt(self, locs=LOCS, start=date(2018, 1, 1), end=date(2025, 6, 30), step=1):
        dates = bt.month_end_dates(start, end, step)
        return bt.run_backtest(self.env.store, self.env.workers, locs, self.env.truth, dates, horizon=12)

    def test_no_look_ahead(self):
        """Changing the data after a date must not change that date's reports."""
        cutoff = date(2022, 12, 31)
        a = self.run_bt([LOCS[0]], date(2021, 1, 1), date(2022, 6, 30))
        with tempfile.TemporaryDirectory() as d2:
            env2 = Env(d2, zhvi=zillow_csv(changed_after=cutoff))
            try:
                dates = bt.month_end_dates(date(2021, 1, 1), date(2022, 6, 30))
                b = bt.run_backtest(env2.store, env2.workers, [LOCS[0]], env2.truth, dates, horizon=12)
            finally:
                env2.close()
        self.assertEqual(len(a.cases), len(b.cases))
        self.assertGreater(len(a.cases), 10)
        for ca, cb in zip(a.cases, b.cases):
            self.assertEqual(ca.reports["housing_prices"].key_metrics, cb.reports["housing_prices"].key_metrics)
            self.assertEqual(ca.reports["housing_prices"].direction, cb.reports["housing_prices"].direction)
            self.assertEqual(ca.reports["housing_prices"].data_as_of, cb.reports["housing_prices"].data_as_of)
            self.assertLess(ca.reports["housing_prices"].data_as_of, ca.as_of)
        self.assertNotEqual([c.fwd_pct for c in a.cases[-6:]], [c.fwd_pct for c in b.cases[-6:]])  # truth does see it

    def test_cases_cover_and_unmatured_are_skipped(self):
        r = self.run_bt(end=date(2026, 8, 31))
        self.assertGreater(r.skipped["not_matured"], 0)
        self.assertTrue(all(c.fwd_pct is not None for c in r.cases))
        self.assertEqual(r.coverage["supply"][0], 0)                      # supply file absent
        self.assertEqual(r.coverage["housing_prices"][0], r.coverage["housing_prices"][1])
        self.assertTrue(all("supply" in c.errors for c in r.cases))

    def test_unknown_location_is_skipped_not_fatal(self):
        r = self.run_bt([Location(city="Nowhereville", state="ZZ"), LOCS[0]])
        self.assertGreater(r.skipped["load_error"], 0)
        self.assertGreater(len(r.cases), 0)

    def test_evaluate_variants_ablation_and_baseline_consistency(self):
        r = self.run_bt()
        names = list(self.env.workers)
        rep = bt.evaluate(self.env.store, r, names)
        v = rep["variants"]["all"]
        for key in ("full", "no_housing_prices", "no_mortgage_rates", "only_housing_prices",
                    "only_mortgage_rates", "base_trend", "base_flat", "base_up"):
            self.assertIn(key, v)
        self.assertIn("full_legacy_weights", v)
        self.assertEqual(v["full"]["n"], len(r.cases))
        # baseline accuracy recomputed by hand
        hand = sum(M.label(c.trailing_pct, 2.0) == c.truth for c in r.cases) / len(r.cases)
        self.assertAlmostEqual(v["base_trend"]["accuracy"], round(hand, 4), places=4)
        # a constant predictor's balanced accuracy is at most 1/3 + tiny for 3 classes present
        self.assertLessEqual(v["base_up"]["balanced_accuracy"], 0.5)
        # supply is always missing, so removing it changes nothing
        self.assertEqual(v["no_supply"]["accuracy"], v["full"]["accuracy"])
        self.assertIn("verdict", rep)
        self.assertEqual(rep["coverage"]["supply"]["ok"], 0)
        self.assertTrue(rep["paired_vs_trend"]["n_dates"] > 20)
        self.assertEqual(len(rep["calibration"]), 4)
        text = bt.format_report(rep)
        self.assertIn("Caveats", text)
        self.assertIn("supply", text)

    def test_split_reports_in_and_out_of_sample_with_embargo(self):
        r = self.run_bt()
        rep = bt.evaluate(self.env.store, r, list(self.env.workers), split=date(2022, 1, 1))
        n_in = rep["variants"]["in_sample"]["full"]["n"]
        n_out = rep["variants"]["out_of_sample"]["full"]["n"]
        self.assertGreater(n_in, 0)
        self.assertGreater(n_out, 0)
        self.assertLess(n_in + n_out, rep["n_cases"])        # embargo drops the in-between months
        self.assertEqual(rep["focus_period"], "out_of_sample")

    def test_verdict_wording_never_overclaims(self):
        r = self.run_bt(end=date(2022, 6, 30), start=date(2020, 1, 1))  # only ~30 dates
        rep = bt.evaluate(self.env.store, r, list(self.env.workers))
        self.assertTrue(any(k in rep["verdict"] for k in ("beats", "WORSE", "within noise", "Too few")))

    def test_combine_matches_full_supervisor_run(self):
        loc, d = LOCS[0], date(2022, 6, 30)
        sup = Supervisor(self.env.store, self.env.workers, max_attempts=1)
        o_run = sup.run(loc, as_of=bt.end_of_day(d))
        state = bt.new_state(loc, bt.end_of_day(d), d)
        sup.run_workers(state)
        o_combine = sup.combine(loc, state["reports"], state["errors"], bt.end_of_day(d), d)
        self.assertEqual((o_run.direction, o_run.score, o_run.confidence),
                         (o_combine.direction, o_combine.score, o_combine.confidence))

    def test_tune_weights_train_test_and_too_few(self):
        small = self.run_bt([LOCS[0]], date(2021, 1, 1), date(2021, 12, 31))
        self.assertIn("too few", bt.tune_weights(self.env.store, small, list(self.env.workers), date(2022, 1, 1))["note"])
        big = self.run_bt(LOCS[:6])
        res = bt.tune_weights(self.env.store, big, list(self.env.workers), date(2022, 6, 30), step=0.25)
        self.assertAlmostEqual(sum(res["chosen_weights"].values()), 1.0)
        self.assertGreater(res["n_train"], 30)
        self.assertGreater(res["n_test"], 30)
        self.assertIn("Chosen on train only", res["note"])

    def test_weight_grid(self):
        g = list(bt.weight_grid(["a", "b", "c"], 0.5))
        self.assertEqual(len(g), 6)
        self.assertTrue(all(abs(sum(w.values()) - 1) < 1e-9 for w in g))


class FallbackVisibilityTests(unittest.TestCase):
    def test_failed_model_calls_are_counted_and_warned(self):
        from retrend.llm import JsonJudge, StubChat
        with tempfile.TemporaryDirectory() as d:
            env = Env(d)
            try:
                bad = JsonJudge(StubChat(["not json"] * 1000, name="bad"), store=env.store)
                zl = ZillowLoader(Path(d) / "z", open_url=opener_for(zillow_csv()), clock=env.store.clock)
                env.workers["housing_prices"] = HousingPriceWorker(
                    env.store, loader=zl, judge=bad, geos=("city",))
                r = bt.run_backtest(env.store, env.workers, LOCS[:2], env.truth,
                                    bt.month_end_dates(date(2021, 1, 1), date(2021, 6, 30)), horizon=12)
                self.assertEqual(r.fallbacks["housing_prices"], len(r.cases))
                self.assertEqual(r.fallbacks["mortgage_rates"], 0)
                rep = bt.evaluate(env.store, r, list(env.workers))
                self.assertEqual(rep["model_fallbacks"]["housing_prices"], len(r.cases))
                self.assertIn("WARNING: model calls failed", bt.format_report(rep))
            finally:
                env.close()


class LedgerTests(unittest.TestCase):
    def test_record_then_score_when_matured(self):
        with tempfile.TemporaryDirectory() as d:
            env = Env(d, clock=Clock(datetime(2024, 6, 15, 12, tzinfo=timezone.utc)))
            try:
                sup = Supervisor(env.store, env.workers, max_attempts=1)
                ledger = Ledger(env.store)
                # a live run on this fake clock sees data up to the clock; record it
                outlook = sup.run(LOCS[0])
                self.assertTrue(ledger.record(outlook, "baseline", env.truth))
                self.assertEqual(len(ledger.entries()), 1)
                res = ledger.score({LOCS[0].key: LOCS[0]}, env.truth)
                self.assertEqual(res["scored"], 1)       # fixture data runs to 2026, so matured
                row = res["rows"][0]
                self.assertIn(row["actual"], M.LABELS)
                self.assertIn("baseline", res["by_model"])
            finally:
                env.close()

    def test_unmatured_entry_is_pending(self):
        with tempfile.TemporaryDirectory() as d:
            env = Env(d, clock=Clock(datetime(2026, 9, 25, 12, tzinfo=timezone.utc)))
            try:
                outlook = Supervisor(env.store, env.workers, max_attempts=1).run(LOCS[0])
                ledger = Ledger(env.store)
                ledger.record(outlook, "baseline", env.truth)
                res = ledger.score({LOCS[0].key: LOCS[0]}, env.truth)
                self.assertEqual((res["scored"], res["pending"]), (0, 1))
            finally:
                env.close()


class CliTests(unittest.TestCase):
    def test_backtest_track_and_ledger_commands(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "cfg"
            cfg.mkdir()
            shutil.copy("config/models.yaml", cfg / "models.yaml")
            (cfg / "cities.yaml").write_text("cities:\n" + "".join(
                f"  - city: {c}\n    state: {s}\n" for c, s in NAMES))
            base = ["--env-file", d + "/.env", "--config-dir", str(cfg), "--db", d + "/c.db",
                    "--log", d + "/l.jsonl"]
            op = opener_for(zillow_csv())
            out = io.StringIO()
            rc = cli.main(base + ["backtest", "--registry", "cities.yaml", "--start", "2019-01-31",
                                  "--end", "2025-03-31", "--split", "2022-06-30"],
                          fred_source=fake_fred, open_url=op, out=out)
            text = out.getvalue()
            self.assertEqual(rc, 0, text)
            self.assertIn("Caveats", text)
            self.assertIn("[out_of_sample]", text)
            saved = list((Path(d) / "eval").glob("backtest_baseline_*.json"))
            self.assertEqual(len(saved), 1)
            self.assertIn("variants", json.loads(saved[0].read_text()))

            out = io.StringIO()
            rc = cli.main(base + ["track", "--registry", "cities.yaml", "--locations", "seattle-wa"],
                          fred_source=fake_fred, open_url=op, out=out)
            self.assertEqual(rc, 0, out.getvalue())
            self.assertIn("recorded", out.getvalue())
            out = io.StringIO()
            rc = cli.main(base + ["ledger-score", "--registry", "cities.yaml"],
                          fred_source=fake_fred, open_url=op, out=out)
            self.assertEqual(rc, 0, out.getvalue())
            self.assertIn("not matured", out.getvalue())

    def test_bad_inputs_are_reported(self):
        with tempfile.TemporaryDirectory() as d:
            out = io.StringIO()
            rc = cli.main(["--env-file", d + "/.env", "--config-dir", "config", "--db", d + "/c.db",
                           "--log", d + "/l.jsonl", "backtest", "--locations", "nope-xx",
                           "--start", "2020-01-31", "--end", "2021-01-31"],
                          fred_source=fake_fred, out=out)
            self.assertEqual(rc, 2)
            self.assertIn("Unknown location", out.getvalue())


if __name__ == "__main__":
    unittest.main()
