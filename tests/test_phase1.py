"""Phase 1 checks: FRED client, metrics, model layer, mortgage worker, CLI.

No network and no real model is used anywhere in these tests.
Run with:  PYTHONPATH=src:tests python3 -m unittest discover -s tests -v
"""

import io
import json
import os
import re
import sys
import tempfile
import types
import unittest
import urllib.parse
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from unittest import mock

from helpers import T0, Clock
from pydantic import ValidationError

from retrend import cli
from retrend.config import ModelConfig, load_dotenv, load_models
from retrend.fred import FredClient, FredError, Observation, parse_observations
from retrend.llm import (
    DEFAULT_STUB_RESPONSE,
    JsonJudge,
    JudgeError,
    LangChainChat,
    StubChat,
    extract_json_object,
    message_text,
    strip_reasoning,
)
from retrend.location import Location
from retrend.logging_utils import get_logger
from retrend.schema import WorkerReport
from retrend.store import RawStore
from retrend.workers.base import NoDataAsOf
from retrend.workers.mortgage import MortgageRateWorker, RateJudgment
from retrend.workers.mortgage_metrics import (
    baseline_direction,
    compute_metrics,
    monthly_payment_per_100k,
)

ROOT = Path(__file__).resolve().parents[1]
SEATTLE = Location(city="Seattle", state="WA")
AUSTIN = Location(city="Austin", state="TX")
SECRET = "abcd1234abcd1234abcd1234abcd1234"

N = 70
LATEST = T0.date()  # a Thursday, like the real survey dates


def rate_at(weeks_ago: int) -> float:
    """6.00% for a year or more ago, then rising 2 bp a week up to 7.04% now."""
    return round(6.0 + max(0, 52 - weeks_ago) * 0.02, 2)


def series(n: int = N, end: date = LATEST) -> list[Observation]:
    return [Observation(end - timedelta(weeks=n - 1 - i), rate_at(n - 1 - i)) for i in range(n)]


def payload_from(obs: list[Observation]) -> dict:
    return {
        "series_id": "MORTGAGE30US",
        "as_of": None,
        "observations": [{"date": o.day.isoformat(), "value": o.value} for o in obs],
    }


def fred_json(obs: list[Observation], missing_on: date | None = None) -> str:
    """The shape FRED actually returns: string values, "." for a missing reading."""
    rows = [
        {
            "realtime_start": "2026-10-13",
            "realtime_end": "2026-10-13",
            "date": o.day.isoformat(),
            "value": "." if o.day == missing_on else f"{o.value:.2f}",
        }
        for o in obs
    ]
    return json.dumps({"count": len(rows), "observations": rows})


def end_of_day(day: date) -> datetime:
    return datetime.combine(day, time(23, 59, 59), tzinfo=timezone.utc)


def reply(direction="up", confidence=0.6, summary="Rates rose.", caveats=None) -> str:
    return json.dumps({"direction": direction, "confidence": confidence,
                       "summary": summary, "caveats": caveats or []})


class FredClientTests(unittest.TestCase):
    def client(self, status=200, body=None, **kw):
        self.urls = []

        def http_get(url, timeout):
            self.urls.append(url)
            return status, body if body is not None else fred_json(series(3))

        return FredClient(SECRET, http_get=http_get, **kw)

    def params(self):
        return dict(urllib.parse.parse_qsl(urllib.parse.urlparse(self.urls[-1]).query))

    def test_live_request_has_no_realtime_parameters(self):
        self.client().observations("MORTGAGE30US", observation_start=date(2021, 10, 1))
        p = self.params()
        self.assertEqual(p["series_id"], "MORTGAGE30US")
        self.assertEqual(p["file_type"], "json")
        self.assertEqual(p["observation_start"], "2021-10-01")
        self.assertNotIn("realtime_start", p)
        self.assertNotIn("observation_end", p)
        self.assertTrue(self.urls[-1].endswith(f"api_key={SECRET}"))

    def test_as_of_request_is_pinned_to_that_vintage_and_capped(self):
        self.client().observations("MORTGAGE30US", as_of=date(2023, 6, 30))
        p = self.params()
        self.assertEqual(p["realtime_start"], "2023-06-30")
        self.assertEqual(p["realtime_end"], "2023-06-30")
        self.assertEqual(p["observation_end"], "2023-06-30")

    def test_missing_values_become_none_and_are_dropped_by_parse(self):
        obs = series(3)
        payload = self.client(body=fred_json(obs, missing_on=obs[1].day)).observations("X")
        self.assertEqual([r["value"] for r in payload["observations"]][1], None)
        self.assertEqual(len(parse_observations(payload)), 2)
        self.assertNotIn("realtime_start", json.dumps(payload))  # would defeat dedup

    def test_http_error_never_leaks_the_key(self):
        body = json.dumps({"error_code": 400, "error_message": f"Bad key {SECRET}"})
        with self.assertRaises(FredError) as ctx:
            self.client(status=400, body=body).observations("X")
        self.assertNotIn(SECRET, str(ctx.exception))
        self.assertIn("400", str(ctx.exception))

    def test_transport_error_never_leaks_the_key(self):
        def boom(url, timeout):
            raise OSError(f"cannot reach {url}")

        with self.assertRaises(FredError) as ctx:
            FredClient(SECRET, http_get=boom).observations("X")
        self.assertNotIn(SECRET, str(ctx.exception))
        self.assertTrue(ctx.exception.__suppress_context__)

    def test_bad_shapes_and_missing_key(self):
        with self.assertRaises(FredError):
            self.client(body='{"nope": 1}').observations("X")
        with self.assertRaises(FredError):
            self.client(body="not json").observations("X")
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(FredError) as ctx:
                FredClient()
        self.assertIn("FRED_API_KEY", str(ctx.exception))


class MetricsTests(unittest.TestCase):
    def test_payment_formula(self):
        self.assertAlmostEqual(monthly_payment_per_100k(6.0), 599.55, places=2)
        self.assertAlmostEqual(monthly_payment_per_100k(7.0), 665.30, places=2)
        self.assertAlmostEqual(monthly_payment_per_100k(0.0), 100000 / 360, places=2)

    def test_metrics_on_known_series(self):
        m = compute_metrics(series(), LATEST)
        self.assertEqual(m["latest_rate_pct"], 7.04)
        self.assertEqual(m["latest_obs_date"], LATEST.isoformat())
        self.assertEqual(m["days_since_latest_obs"], 0)
        self.assertEqual(m["change_4w_bp"], 8.0)
        self.assertEqual(m["change_13w_bp"], 26.0)
        self.assertEqual(m["change_52w_bp"], 104.0)
        self.assertEqual(m["range_52w_low_pct"], 6.0)
        self.assertEqual(m["range_52w_high_pct"], 7.04)
        self.assertEqual(m["percentile_52w"], 1.0)
        self.assertEqual(m["window_years"], 1.3)
        self.assertEqual(m["payment_per_100k"], round(monthly_payment_per_100k(7.04), 2))
        expected = round((monthly_payment_per_100k(7.04) / monthly_payment_per_100k(6.0) - 1) * 100, 1)
        self.assertEqual(m["payment_change_52w_pct"], expected)
        self.assertGreater(m["payment_change_52w_pct"], 0)
        self.assertEqual(m["baseline_direction"], "down")  # +26 bp over 13 weeks

    def test_short_history_gives_none_not_wrong_numbers(self):
        m = compute_metrics(series(n=8), LATEST)
        self.assertEqual(m["change_4w_bp"], 8.0)
        self.assertIsNone(m["change_13w_bp"])
        self.assertIsNone(m["change_52w_bp"])
        self.assertIsNone(m["range_52w_low_pct"])
        self.assertIsNone(m["payment_change_52w_pct"])
        self.assertEqual(m["baseline_direction"], "flat")

    def test_look_back_across_a_gap_is_not_used(self):
        obs = series()
        gappy = [o for o in obs if not (LATEST - timedelta(weeks=20) < o.day < LATEST - timedelta(weeks=8))]
        m = compute_metrics(gappy, LATEST)
        self.assertIsNone(m["change_13w_bp"])  # nearest older reading is too far from the target
        self.assertEqual(m["change_4w_bp"], 8.0)

    def test_baseline_thresholds(self):
        self.assertEqual(baseline_direction(25.0)[0], "down")
        self.assertEqual(baseline_direction(24.9)[0], "flat")
        self.assertEqual(baseline_direction(-25.0)[0], "up")
        self.assertEqual(baseline_direction(None)[0], "flat")


class TextHelperTests(unittest.TestCase):
    def test_strip_reasoning(self):
        self.assertEqual(strip_reasoning('<think>hmm {x}</think>{"a": 1}'), '{"a": 1}')
        self.assertEqual(strip_reasoning('opening tag was dropped</think>{"a": 1}'), '{"a": 1}')
        self.assertEqual(strip_reasoning("<think>never closes {\"a\": 1}"), "")
        self.assertEqual(strip_reasoning('{"a": 1}'), '{"a": 1}')

    def test_extract_json_object(self):
        self.assertEqual(extract_json_object('{"a": 1}'), {"a": 1})
        self.assertEqual(extract_json_object('Sure!\n```json\n{"a": 1}\n```\nDone'), {"a": 1})
        self.assertEqual(extract_json_object('Result: {"a": {"b": 1}} trailing'), {"a": {"b": 1}})
        self.assertEqual(extract_json_object('{not json} {"a": 1}'), {"a": 1})
        for bad in ("no json here", "[1, 2]", ""):
            with self.assertRaises(ValueError):
                extract_json_object(bad)

    def test_message_text_across_langchain_versions(self):
        class Prop:
            text = "from property"

        class Method:
            def text(self):
                return "from method"

        class Blocks:
            content = [{"type": "thinking", "thinking": "skip me"},
                       {"type": "text", "text": "a"}, "b", {"type": "text", "text": "c"}]

        class Plain:
            content = "plain"

        self.assertEqual(message_text(Prop()), "from property")
        self.assertEqual(message_text(Method()), "from method")
        self.assertEqual(message_text(Blocks()), "abc")
        self.assertEqual(message_text(Plain()), "plain")
        self.assertEqual(message_text("raw string"), "raw string")


class RateJudgmentTests(unittest.TestCase):
    def test_normalizes_and_tidies(self):
        j = RateJudgment.model_validate(
            {"direction": " UP ", "confidence": "0.7", "summary": "ok", "caveats": "one",
             "unexpected": "ignored"}
        )
        self.assertEqual((j.direction, j.confidence, j.caveats), ("up", 0.7, ["one"]))
        many = RateJudgment.model_validate(
            {"direction": "flat", "confidence": 0.5, "summary": "ok",
             "caveats": ["a", " ", "b", "c", "d"]}
        )
        self.assertEqual(many.caveats, ["a", "b", "c"])

    def test_rejects_bad_values(self):
        for bad in (
            {"direction": "sideways", "confidence": 0.5, "summary": "x"},
            {"direction": "up", "confidence": 70, "summary": "x"},
            {"direction": "up", "confidence": 0.5, "summary": ""},
            {"direction": "up", "confidence": 0.5},
        ):
            with self.assertRaises(ValidationError):
                RateJudgment.model_validate(bad)


class JudgeTests(unittest.TestCase):
    def make(self, responses, *, store=None, name="m", **kw):
        self.sleeps = []
        chat = StubChat(responses, name=name)
        return JsonJudge(chat, store=store, sleep=self.sleeps.append, **kw), chat

    def test_valid_first_try(self):
        judge, chat = self.make([reply()])
        result = judge.judge("sys", "usr", RateJudgment)
        self.assertEqual((result.value.direction, result.attempts, result.cached), ("up", 1, False))
        self.assertEqual(judge.stats["successes"], 1)

    def test_tolerates_think_tags_and_fences(self):
        judge, _ = self.make([f"<think>hmm {{no}}</think>\n```json\n{reply()}\n```"])
        self.assertEqual(judge.judge("s", "u", RateJudgment).value.confidence, 0.6)

    def test_retries_with_feedback(self):
        judge, chat = self.make(["not json at all", reply(confidence=1.7), reply()])
        result = judge.judge("s", "u", RateJudgment)
        self.assertEqual(result.attempts, 3)
        self.assertEqual(judge.stats["parse_failures"], 2)
        self.assertEqual(judge.stats["calls"], 3)
        last = chat.calls[-1]
        self.assertEqual(last[-1][0], "user")
        self.assertIn("not valid", last[-1][1])
        self.assertEqual(last[-2][0], "assistant")

    def test_gives_up_after_max_attempts_and_caches_nothing(self):
        with RawStore(clock=Clock()) as store:
            judge, chat = self.make(["bad", "bad", "bad"], store=store)
            with self.assertRaises(JudgeError) as ctx:
                judge.judge("s", "u", RateJudgment)
            self.assertEqual(ctx.exception.attempts, 3)
            self.assertEqual(judge.stats["failures"], 1)
            self.assertEqual(len(chat.calls), 3)
            self.assertEqual(self.sleeps, [])  # parse failures are not rate limits: no waiting

    def test_cache_hit_skips_the_model_and_is_per_model(self):
        with RawStore(clock=Clock()) as store:
            judge, chat = self.make([reply()], store=store)
            judge.judge("s", "u", RateJudgment)
            again = judge.judge("s", "u", RateJudgment)
            self.assertTrue(again.cached)
            self.assertEqual(len(chat.calls), 1)
            self.assertEqual(judge.stats["cache_hits"], 1)
            # A new judge over the same store and model also hits the cache...
            judge2, chat2 = self.make([reply()], store=store)
            self.assertTrue(judge2.judge("s", "u", RateJudgment).cached)
            self.assertEqual(len(chat2.calls), 0)
            # ...but a different model, or a different prompt, does not.
            other, other_chat = self.make([reply("flat")], store=store, name="other")
            self.assertEqual(other.judge("s", "u", RateJudgment).value.direction, "flat")
            judge.judge("s", "different prompt", RateJudgment)
            self.assertEqual(len(chat.calls), 2)

    def test_provider_errors_back_off_then_succeed(self):
        judge, _ = self.make([RuntimeError("connection reset"), reply()], base_delay=2.0)
        judge.judge("s", "u", RateJudgment)
        self.assertEqual(self.sleeps, [2.0])
        self.assertEqual(judge.stats["provider_errors"], 1)

    def test_rate_limit_errors_wait_longer(self):
        judge, _ = self.make([RuntimeError("429 rate limited"), reply()], base_delay=2.0)
        judge.judge("s", "u", RateJudgment)
        self.assertEqual(self.sleeps, [20.0])

    def test_provider_errors_exhaust_attempts(self):
        judge, _ = self.make([RuntimeError("down")] * 3, base_delay=1.0)
        with self.assertRaises(JudgeError) as ctx:
            judge.judge("s", "u", RateJudgment)
        self.assertEqual(self.sleeps, [1.0, 2.0])  # no pointless sleep after the last attempt
        self.assertIn("RuntimeError: down", str(ctx.exception))
        self.assertEqual(judge.stats["provider_errors"], 3)


class ConfigAndChatTests(unittest.TestCase):
    def test_dotenv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text('# c\n\nRT_A=1\nexport RT_B = two \nRT_C="quoted value"\nbad line\nRT_D=x\n')
            with mock.patch.dict(os.environ, {"RT_D": "from_env"}, clear=True):
                loaded = load_dotenv(path)
                self.assertEqual(sorted(loaded), ["RT_A", "RT_B", "RT_C"])
                self.assertEqual(os.environ["RT_B"], "two")
                self.assertEqual(os.environ["RT_C"], "quoted value")
                self.assertEqual(os.environ["RT_D"], "from_env")  # existing env wins
            self.assertEqual(load_dotenv(Path(tmp) / "missing.env"), [])

    def test_shipped_models_yaml(self):
        models = load_models(ROOT / "config" / "models.yaml")
        self.assertEqual(set(models), {"qwen", "llama", "gemma"})
        self.assertEqual(models["qwen"].provider, "ollama")
        self.assertEqual(models["qwen"].model_id, "qwen3:8b")  # tag keeps its own colon

    def test_models_yaml_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.yaml"
            for text in ("models:\n  x:\n    spec: nocolon\n",
                         "models:\n  baseline:\n    spec: 'a:b'\n",
                         "models:\n  x:\n    spec: 'a:b'\n    params: [1]\n"):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    load_models(path)

    def test_ollama_provider_maps_to_chatollama(self):
        seen = {}

        class ChatOllama:
            def __init__(self, **kw):
                seen.update(kw)

            def invoke(self, messages):
                return types.SimpleNamespace(text="hello", content="hello")

        fake = types.ModuleType("langchain_ollama")
        fake.ChatOllama = ChatOllama
        with mock.patch.dict(sys.modules, {"langchain_ollama": fake}):
            chat = LangChainChat(ModelConfig("q", "ollama:qwen3:8b", params={"temperature": 0.2}))
            self.assertEqual(chat.complete([("user", "hi")]), "hello")
        self.assertEqual(seen, {"model": "qwen3:8b", "temperature": 0.2})
        self.assertIn("qwen3:8b", chat.cache_id)

    def test_chat_errors_are_clear(self):
        with self.assertRaises(ValueError):
            LangChainChat(ModelConfig("x", "nope:m"))
        with self.assertRaises(RuntimeError) as ctx:
            LangChainChat(ModelConfig("x", "p:m", class_path="not_a_real_module_xyz.Chat"))
        self.assertIn("Install", str(ctx.exception))


class StoreExtensionTests(unittest.TestCase):
    def test_data_as_of_can_be_computed_from_the_payload(self):
        with RawStore(clock=Clock()) as store:
            record = store.fetch_or_cache(
                "fred", "k", lambda: {"d": "2026-09-24"}, max_age=timedelta(days=1),
                data_as_of=lambda p: date.fromisoformat(p["d"]),
            )
            self.assertEqual(record.data_as_of, date(2026, 9, 24))


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.store = RawStore(clock=self.clock)
        self.calls = []
        self.obs = series()

    def tearDown(self):
        self.store.close()

    def source(self, series_id, **kwargs):
        self.calls.append(kwargs)
        return payload_from(self.obs)  # note: ignores as_of, like a misbehaving source

    def worker(self, responses=None, *, judge="stub", **kw):
        self.chat = StubChat(responses or [reply()], name="fake-model")
        if judge == "stub":
            judge = JsonJudge(self.chat, store=self.store, sleep=lambda s: None)
        return MortgageRateWorker(self.store, fred_source=self.source, judge=judge, **kw)

    def test_live_report(self):
        report = self.worker().run(SEATTLE)
        WorkerReport.model_validate_json(report.model_dump_json())
        self.assertEqual(report.worker, "mortgage_rates")
        self.assertEqual(report.direction, "up")  # the model's call, not the baseline's
        self.assertEqual(report.confidence, 0.6)
        self.assertEqual(report.judged_by, "fake-model")
        self.assertEqual(report.summary, "Rates rose.")
        self.assertEqual(report.geography_match, "approximate")
        self.assertEqual(report.data_as_of, LATEST)
        self.assertEqual(report.key_metrics["latest_rate_pct"], 7.04)
        self.assertEqual(report.key_metrics["baseline_direction"], "down")
        self.assertEqual(report.key_metrics["judge_attempts"], 1)
        self.assertIn("National average", report.caveats[0])
        self.assertTrue(report.sources[0].url.startswith("https://fred.stlouisfed.org/series/"))
        self.assertEqual(self.calls[0]["as_of"], None)
        self.assertEqual(self.store.get_latest("fred", "MORTGAGE30US|live").data_as_of, LATEST)

    def test_model_caveats_are_appended_after_code_caveats(self):
        report = self.worker([reply(caveats=["lagged effect"])]).run(SEATTLE)
        self.assertEqual(report.caveats[-1], "lagged effect")

    def test_methodology_change_caveat_when_the_window_spans_it(self):
        report = self.worker().run(SEATTLE)  # 70 weeks back is mid-2025: does not span Nov 2022
        self.assertFalse(any("methodology" in c for c in report.caveats))
        self.obs = [Observation(date(2022, 1, 6) + timedelta(weeks=i), 5.0) for i in range(300)]
        self.clock.now = datetime.combine(self.obs[-1].day, time(12), tzinfo=timezone.utc)
        self.assertTrue(any("methodology" in c for c in self.worker().run(SEATTLE).caveats))

    def test_as_of_uses_a_pinned_vintage_and_never_sees_later_readings(self):
        cutoff = LATEST - timedelta(weeks=10)
        report = self.worker().run(SEATTLE, as_of=end_of_day(cutoff))
        self.assertEqual(self.calls[0]["as_of"], cutoff)
        self.assertEqual(self.calls[0]["observation_end"], cutoff)
        self.assertEqual(report.data_as_of, cutoff)
        self.assertEqual(report.key_metrics["latest_rate_pct"], rate_at(10))
        self.assertEqual(report.key_metrics["change_52w_bp"], round((rate_at(10) - 6.0) * 100, 1))

    def test_as_of_pulls_are_cached_separately_and_forever(self):
        cutoff = LATEST - timedelta(weeks=10)
        w = self.worker()
        w.run(SEATTLE, as_of=end_of_day(cutoff))
        self.clock.advance(days=400)
        w.run(SEATTLE, as_of=end_of_day(cutoff))
        self.assertEqual(len(self.calls), 1)
        w.run(SEATTLE)  # live pull is a different cache entry
        self.assertEqual(len(self.calls), 2)

    def test_naive_as_of_is_rejected_and_empty_data_raises(self):
        with self.assertRaises(ValueError):
            self.worker().run(SEATTLE, as_of=datetime(2026, 1, 1))
        self.obs = []
        with self.assertRaises(NoDataAsOf):
            self.worker().run(SEATTLE)
        self.obs = series()
        with self.assertRaises(NoDataAsOf):  # as-of before any reading exists
            self.worker().run(SEATTLE, as_of=end_of_day(date(2000, 1, 1)))

    def test_stale_data_is_flagged(self):
        later = end_of_day(LATEST + timedelta(days=30))
        report = self.worker().run(SEATTLE, as_of=later)
        self.assertTrue(any("30 days old" in c for c in report.caveats))

    def test_second_run_uses_both_caches_and_one_model_call_serves_every_city(self):
        w = self.worker()
        w.run(SEATTLE)
        second = w.run(AUSTIN)
        self.assertEqual(len(self.calls), 1)  # FRED pull cached
        self.assertEqual(len(self.chat.calls), 1)  # same prompt: one model call
        self.assertTrue(second.key_metrics["judge_cached"])
        self.assertEqual(second.location, AUSTIN)

    def test_prompt_has_no_location_dates_or_baseline(self):
        self.worker().run(SEATTLE)
        system, user = self.chat.calls[0][0][1], self.chat.calls[0][1][1]
        for text in (system, user):
            self.assertIsNone(re.search(r"\d{4}-\d{2}-\d{2}", text))
            self.assertNotIn("Seattle", text)
        self.assertNotIn("baseline", user.lower())
        self.assertIn('"latest_rate_pct": 7.04', user)

    def test_baseline_only_mode(self):
        report = self.worker(judge=None).run(SEATTLE)
        self.assertEqual(report.direction, "down")
        self.assertEqual(report.confidence, 0.3)
        self.assertEqual(report.judged_by, "rule:baseline-13w")
        self.assertTrue(any("Rule-based" in c for c in report.caveats))

    def test_model_failure_falls_back_to_baseline_and_says_so(self):
        w = self.worker([RuntimeError("ollama is down")] * 3)
        report = w.run(SEATTLE)
        self.assertEqual(report.direction, report.key_metrics["baseline_direction"])
        self.assertIn("fallback from fake-model", report.judged_by)
        self.assertTrue(any("failed" in c for c in report.caveats))

    def test_no_fallback_lets_the_error_through(self):
        w = self.worker([RuntimeError("down")] * 3, fallback_to_baseline=False)
        with self.assertRaises(JudgeError):
            w.run(SEATTLE)


class KeyLeakTests(unittest.TestCase):
    def test_the_api_key_never_reaches_reports_store_or_logs(self):
        urls = []

        def http_get(url, timeout):
            urls.append(url)
            return 200, fred_json(series())

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "run.jsonl"
            logger = get_logger("retrend.test_leak", log_path)
            store = RawStore(clock=Clock(), logger=logger)
            client = FredClient(SECRET, http_get=http_get)
            judge = JsonJudge(StubChat([reply()]), store=store, logger=logger)
            report = MortgageRateWorker(
                store, logger, fred_source=client.observations, judge=judge
            ).run(SEATTLE)
            for h in logger.handlers:
                h.flush()
            self.assertIn(SECRET, urls[0])  # the key really was sent to FRED
            stored = store.get_latest("fred", "MORTGAGE30US|live").text()
            for blob in (report.model_dump_json(), stored, log_path.read_text()):
                self.assertNotIn(SECRET, blob)
            store.close()
            get_logger("retrend.test_leak")  # release the file handle


class CliTests(unittest.TestCase):
    def run_cli(self, *argv):
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            base = ["--db", f"{tmp}/data/cache.db", "--log", f"{tmp}/logs/run.jsonl",
                    "--config-dir", str(ROOT / "config"), "--env-file", f"{tmp}/none.env"]
            code = cli.main(
                base + list(argv),
                fred_source=lambda series_id, **kw: payload_from(series()),
                out=out,
            )
            saved = sorted(p.name for p in Path(tmp).glob("data/reports/*/*.json"))
            get_logger("retrend.cli")  # release the log file handle
        return code, out.getvalue(), saved

    def test_baseline_and_stub_side_by_side(self):
        code, text, saved = self.run_cli("mortgage", "seattle-wa", "--models", "baseline,stub")
        self.assertEqual(code, 0, text)
        self.assertIn("baseline", text)
        self.assertIn("rule:baseline-13w", text)
        self.assertIn("stub", text)
        self.assertEqual(saved, ["mortgage_live_baseline.json", "mortgage_live_stub.json"])

    def test_multiple_as_of_dates(self):
        code, text, saved = self.run_cli(
            "mortgage", "seattle-wa", "--models", "baseline", "--as-of", "2026-06-25", "2026-08-13"
        )
        self.assertEqual(code, 0, text)
        self.assertEqual(saved, ["mortgage_2026-06-25_baseline.json", "mortgage_2026-08-13_baseline.json"])

    def test_fred_check(self):
        code, text, _ = self.run_cli("fred-check")
        self.assertEqual(code, 0)
        self.assertIn("Fetched 70 weekly observations", text)
        self.assertIn("latest_rate_pct: 7.04", text)

    def test_unknown_inputs_exit_with_2(self):
        self.assertEqual(self.run_cli("mortgage", "atlantis-xx")[0], 2)
        self.assertEqual(self.run_cli("mortgage", "seattle-wa", "--models", "nonsense")[0], 2)


if __name__ == "__main__":
    unittest.main()
