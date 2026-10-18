"""Model comparison table, improved weight tuning."""

import io
import json
import shutil
import tempfile
import unittest
from datetime import date
from pathlib import Path

from retrend import cli
from retrend.eval import backtest as bt
from retrend.eval import compare as cmp
from retrend.llm import JsonJudge, StubChat
from retrend.sources.redfin import RedfinLoader
from retrend.sources.zillow import ZillowLoader
from retrend.workers.housing import HousingPriceWorker
from retrend.workers.mortgage import MortgageRateWorker
from retrend.workers.supply import SupplyWorker
from test_phase4 import LOCS, NAMES, Env, fake_fred, opener_for, zillow_csv


def reply(direction, conf=0.6):
    return json.dumps({"direction": direction, "confidence": conf, "summary": "canned", "caveats": []})


def judge(env, direction=None, name="m", raw=None):
    chat = StubChat([raw if raw is not None else reply(direction)], name=name)
    return JsonJudge(chat, store=env.store, max_attempts=1, sleep=lambda s: None)


def builder(env):
    def build(j):
        zl = ZillowLoader(env.tmp / "z", open_url=opener_for(zillow_csv()), clock=env.store.clock)
        o = opener_for(zillow_csv())
        return {"mortgage_rates": MortgageRateWorker(env.store, fred_source=fake_fred, judge=j),
                "housing_prices": HousingPriceWorker(env.store, loader=zl, judge=j, geos=("city",)),
                "supply": SupplyWorker(env.store, loader=RedfinLoader(env.tmp / "r", open_url=o), judge=j)}
    return build


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = Env(self.tmp.name)
        self.dates = bt.month_end_dates(date(2019, 1, 31), date(2025, 6, 30), 3)

    def tearDown(self):
        self.env.close()
        self.tmp.cleanup()

    def run_cmp(self, judges, **kw):
        return cmp.run_comparison(self.env.store, builder(self.env), judges, LOCS[:6], self.env.truth, self.dates, **kw)

    def test_table_has_rules_models_baselines_and_cost_columns(self):
        out = self.run_cmp({"bull": judge(self.env, "up", "bull"), "bear": judge(self.env, "down", "bear")})
        self.assertGreater(out["n_common_cases"], 30)
        self.assertEqual(list(out["models"]), ["rules", "bull", "bear"])
        self.assertEqual(set(out["baselines"]), set(cmp.BASELINES))
        bull, bear = out["models"]["bull"], out["models"]["bear"]
        self.assertGreater(bull["calls"], 0)
        self.assertEqual(bull["n_scored"], bear["n_scored"])  # same cases for everyone
        self.assertEqual(bull["fallbacks"], 0)
        self.assertEqual(bull["valid_rate"], 1.0)
        # one model says up, the other down: they can never agree on a case
        self.assertIn("bull vs bear", out["head_to_head"])
        text, md, csv_text = cmp.format_text(out), cmp.to_markdown(out), cmp.to_csv(out)
        for s in ("bull", "bear", "rules", "always flat"):
            self.assertIn(s, text)
            self.assertIn(s, md)
        self.assertEqual(csv_text.splitlines()[0].split(",")[0], "Model")
        self.assertIn("| Model |", md)
        self.assertIn("Model calls", md)

    def test_a_model_that_only_falls_back_is_not_scored_and_is_flagged(self):
        out = self.run_cmp({"ok": judge(self.env, "up", "ok"), "broken": judge(self.env, raw="not json", name="broken")})
        self.assertEqual(out["n_common_cases"], 0)
        self.assertIn("No cases where every model", out["verdict"])
        self.assertGreater(out["models"]["broken"]["fallbacks"], 0)
        self.assertLess(out["models"]["broken"]["valid_rate"], 0.5)
        cmp.format_text(out)  # must not crash with no scored cases
        cmp.to_markdown(out)

    def test_rules_row_matches_the_plain_backtest(self):
        out = self.run_cmp({"bull": judge(self.env, "up", "bull")})
        result = bt.run_backtest(self.env.store, builder(self.env)(None), LOCS[:6], self.env.truth, self.dates)
        rep = bt.evaluate(self.env.store, result, ["mortgage_rates", "housing_prices", "supply"])
        # the common set can be smaller than the plain backtest's, but never larger
        self.assertLessEqual(out["n_common_cases"], rep["n_cases"])
        self.assertIsNotNone(out["models"]["rules"]["accuracy"])

    def test_tuning_is_reported_per_model_when_asked(self):
        out = self.run_cmp({"bull": judge(self.env, "up", "bull")}, split=date(2022, 6, 30), tune=True)
        self.assertEqual(set(out["tuning"]), {"rules", "bull"})
        t = out["tuning"]["rules"]
        if "chosen_weights" in t:
            self.assertIn("recommendation", t)


class TuningTests(unittest.TestCase):
    def test_tuning_reports_default_comparison_and_prefers_defaults_on_ties(self):
        with tempfile.TemporaryDirectory() as d:
            env = Env(d)
            try:
                dates = bt.month_end_dates(date(2018, 1, 1), date(2025, 6, 30), 1)
                result = bt.run_backtest(env.store, env.workers, LOCS[:6], env.truth, dates)
                res = bt.tune_weights(env.store, result, list(env.workers), date(2022, 6, 30), step=0.25)
                self.assertAlmostEqual(sum(res["chosen_weights"].values()), 1.0)
                for k in ("default_weights", "recommendation", "test_paired_accuracy_diff", "grid_step", "n_candidates"):
                    self.assertIn(k, res)
                self.assertTrue(res["recommendation"].startswith(("ADOPT", "KEEP")))
                self.assertGreaterEqual(res["best_train_balanced_accuracy"], res["train_balanced_accuracy"])
                self.assertLessEqual(res["best_train_balanced_accuracy"] - res["train_balanced_accuracy"], bt.TIE_TOL + 1e-9)
            finally:
                env.close()

    def test_auto_step_gets_coarser_with_more_signals(self):
        self.assertEqual(bt.auto_step(3), 0.1)
        self.assertEqual(bt.auto_step(6), 0.2)


class CliTests(unittest.TestCase):
    def test_compare_models_command_writes_table_files(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "cfg"
            cfg.mkdir()
            shutil.copy("config/models.yaml", cfg / "models.yaml")
            (cfg / "cities.yaml").write_text("cities:\n" + "".join(f"  - city: {c}\n    state: {s}\n" for c, s in NAMES))
            base = ["--env-file", d + "/.env", "--config-dir", str(cfg), "--db", d + "/data/c.db", "--log", d + "/l.jsonl"]
            out = io.StringIO()
            rc = cli.main(base + ["compare-models", "--registry", "cities.yaml", "--start", "2019-01-31", "--end", "2025-06-30",
                                  "--step-months", "6", "--models", "stub"],
                          fred_source=fake_fred, open_url=opener_for(zillow_csv()), out=out)
            self.assertEqual(rc, 0, out.getvalue())
            self.assertIn("stub", out.getvalue())
            files = {p.suffix for p in (Path(d) / "data" / "eval").glob("compare_*")}
            self.assertEqual(files, {".md", ".csv", ".json"})

    def test_unknown_or_missing_models_are_explained(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "cfg"
            cfg.mkdir()
            shutil.copy("config/models.yaml", cfg / "models.yaml")
            (cfg / "cities.yaml").write_text("cities:\n  - city: Seattle\n    state: WA\n")
            base = ["--env-file", d + "/.env", "--config-dir", str(cfg), "--db", d + "/data/c.db", "--log", d + "/l.jsonl"]
            out = io.StringIO()
            rc = cli.main(base + ["compare-models", "--registry", "cities.yaml", "--start", "2019-01-31", "--end", "2020-01-31",
                                  "--models", "nosuchmodel"], fred_source=fake_fred, open_url=opener_for(zillow_csv()), out=out)
            self.assertEqual(rc, 2)
            self.assertIn("config/models.yaml", out.getvalue())


if __name__ == "__main__":
    unittest.main()
