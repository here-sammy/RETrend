"""Parallel model calls for backtests (prefetch, then replay from the cache)."""

import json
import tempfile
import threading
import time
import unittest
from datetime import date

from retrend.eval import backtest as bt
from retrend.eval import compare as cmp
from retrend.eval.prefetch import warm_cache
from retrend.llm import JsonJudge, StubChat
from test_phase4 import LOCS, Env
from test_phase15 import builder, reply


class SlowChat(StubChat):
    def __init__(self, *a, fail_on=None, **k):
        super().__init__(*a, **k)
        self.threads = set()
        self.fail_on = fail_on

    def complete(self, messages):
        self.threads.add(threading.get_ident())
        self.calls.append(list(messages))
        time.sleep(0.01)
        if self.fail_on and self.fail_on in messages[0][1]:
            raise RuntimeError("boom")
        return reply("up")


def mk(env, **kw):
    return JsonJudge(SlowChat(name="m", **kw), store=env.store, max_attempts=1, sleep=lambda s: None)


class ParallelTests(unittest.TestCase):
    def fresh(self):
        tmp = tempfile.TemporaryDirectory()
        env = Env(tmp.name)
        self.addCleanup(tmp.cleanup)
        self.addCleanup(env.close)
        return env

    def setUp(self):
        self.env = self.fresh()
        self.dates = bt.month_end_dates(date(2019, 1, 31), date(2025, 6, 30), 6)
        self.locs = LOCS[:4]

    def run_bt(self, j, parallel):
        workers = builder(self.env)(j)
        if parallel > 1:
            warm_cache(j, lambda: bt.run_backtest(self.env.store, workers, self.locs, self.env.truth, self.dates),
                       parallel=parallel)
        return bt.run_backtest(self.env.store, workers, self.locs, self.env.truth, self.dates)

    def key(self, r):
        return [(c.loc.key, c.as_of, {n: (x.direction, x.judged_by) for n, x in c.reports.items()})
                for c in r.cases]

    def test_same_results_as_one_at_a_time(self):
        seq = self.run_bt(mk(self.env), 1)
        self.env = self.fresh()  # empty cache, so the parallel run really asks the model
        par = self.run_bt(mk(self.env), 4)
        self.assertEqual(self.key(seq), self.key(par))

    def test_threads_used_and_no_repeat_calls(self):
        j = mk(self.env)
        self.run_bt(j, 4)
        self.assertGreater(len(j.chat.threads), 1)
        calls = j.stats["calls"]
        self.assertEqual(j.stats["cache_hits"] > 0, True)
        self.assertEqual(calls, len({json.dumps(c) for c in j.chat.calls}))

    def test_collect_mode_makes_no_model_calls(self):
        j = mk(self.env)
        workers = builder(self.env)(j)
        j.collect = []
        bt.run_backtest(self.env.store, workers, self.locs, self.env.truth, self.dates)
        self.assertTrue(j.collect)
        self.assertEqual(j.stats["calls"], 0)

    def test_failed_prefetch_is_left_to_the_normal_run(self):
        j = mk(self.env, fail_on="supply and demand indicators")
        r = self.run_bt(j, 4)
        self.assertTrue(r.cases)
        ok, total = r.coverage["supply"]
        self.assertLess(ok, total)  # the failed questions are not cached, so they fail again as before
        self.assertEqual(r.coverage["housing_prices"][0], r.coverage["housing_prices"][1])

    def test_parallel_one_is_a_noop(self):
        self.assertIsNone(warm_cache(mk(self.env), lambda: None, parallel=1))

    def test_comparison_accepts_parallel(self):
        out = cmp.run_comparison(self.env.store, builder(self.env), {"m": mk(self.env)}, self.locs,
                                 self.env.truth, self.dates, parallel=3)
        self.assertIn("m", out["models"])


    def test_models_are_asked_at_the_same_time(self):
        a, b = mk(self.env), mk(self.env)
        a.chat.name, b.chat.name = "a", "b"
        a.chat.cache_id, b.chat.cache_id = "stub:a", "stub:b"
        out = cmp.run_comparison(self.env.store, builder(self.env), {"a": a, "b": b}, self.locs,
                                 self.env.truth, self.dates, parallel=3)
        self.assertEqual(set(out["models"]), {"rules", "a", "b"})
        self.assertGreater(a.stats["calls"], 0)
        self.assertGreater(b.stats["calls"], 0)
        self.assertFalse(a.chat.threads & b.chat.threads)  # separate pools, so separate threads

    def test_per_model_limit_from_models_yaml(self):
        import tempfile, pathlib
        from retrend.config import load_models
        f = pathlib.Path(tempfile.mkdtemp()) / "m.yaml"
        f.write_text("models:\n  g:\n    spec: 'google_genai:x'\n    parallel: 8\n  q:\n    spec: 'ollama:q'\n")
        m = load_models(f)
        self.assertEqual((m["g"].parallel, m["q"].parallel), (8, None))
        f.write_text("models:\n  g:\n    spec: 'google_genai:x'\n    parallel: 0\n")
        with self.assertRaises(ValueError):
            load_models(f)


class CacheKeyTests(unittest.TestCase):
    def test_speed_only_params_do_not_change_the_cache_key(self):
        import sys, types
        from retrend.config import ModelConfig
        from retrend.llm import LangChainChat

        class Fake:
            def __init__(self, **kw): pass
        mod = types.ModuleType("fakechat"); mod.Fake = Fake; sys.modules["fakechat"] = mod
        mk_ = lambda **p: LangChainChat(ModelConfig(name="x", spec="fake:m", class_path="fakechat.Fake", params=p)).cache_id
        self.assertEqual(mk_(temperature=0.2), mk_(temperature=0.2, keep_alive="24h"))
        self.assertNotEqual(mk_(temperature=0.2), mk_(temperature=0.5))


class ThrottleTests(unittest.TestCase):
    def test_rpm_spaces_requests_and_rate_limits_pause_longer(self):
        sleeps = []
        j = JsonJudge(StubChat([RuntimeError("429 Resource exhausted"), reply("up")]), max_attempts=2,
                      sleep=sleeps.append)
        j.rpm = 30
        from retrend.workers.judging import Judgment
        j.judge("s", "u", Judgment)
        self.assertTrue(any(s >= 20 for s in sleeps))  # rate limit: waited a real pause
        j._next_slot = time.monotonic() + 100  # next slot is far away -> must wait about that long
        sleeps.clear()
        j.judge("s", "u2", Judgment)
        self.assertTrue(any(s > 50 for s in sleeps))

    def test_rpm_validation(self):
        import tempfile, pathlib
        from retrend.config import load_models
        f = pathlib.Path(tempfile.mkdtemp()) / "m.yaml"
        f.write_text("models:\n  g:\n    spec: 'x:y'\n    rpm: 0\n")
        with self.assertRaises(ValueError):
            load_models(f)


class BreakerTests(unittest.TestCase):
    def test_all_failures_stop_the_run_with_the_real_error(self):
        from retrend.llm import PrefetchAborted
        from retrend.workers.judging import Judgment
        j = JsonJudge(StubChat([RuntimeError("404 model not found: nope")]), max_attempts=1, sleep=lambda s: None)
        reqs = [("s", f"u{i}", Judgment) for i in range(40)]
        with self.assertRaises(PrefetchAborted) as cm:
            j.prefetch(reqs, workers=2)
        self.assertIn("model not found", str(cm.exception))

    def test_some_successes_do_not_trip_it(self):
        from retrend.workers.judging import Judgment
        ok = reply("up")
        j = JsonJudge(StubChat([ok, RuntimeError("x")] * 20), max_attempts=1, sleep=lambda s: None)
        out = j.prefetch([("s", f"u{i}", Judgment) for i in range(12)], workers=1)
        self.assertGreater(out["done"], 0)


if __name__ == "__main__":
    unittest.main()
