"""Phase 7: question planner, grounded answers, runner, local server."""

import http.client
import json
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from retrend.location import load_registry
from retrend.outlook import Outlook
from retrend.ui import answer as A
from retrend.ui import planner as P
from retrend.ui.planner import Plan, PlanDraft, build_plan, rule_plan
from retrend.ui.runner import QueryRunner
from retrend.ui.server import JobThread, serve
from retrend.store import RawStore
from test_phase4 import fake_fred, opener_for, zillow_csv

REG = load_registry("config/backtest_cities.yaml")


class FakeJudge:
    model_name = "fake"

    def __init__(self, value=None, exc=None):
        self.value, self.exc = value, exc

    def judge(self, system, user, schema):
        if self.exc:
            raise self.exc
        return SimpleNamespace(value=self.value, model="fake")


class PlannerTests(unittest.TestCase):
    def keys(self, q):
        return [l.key for l in rule_plan(q, REG).locations]

    def test_single_city_outlook_uses_all_signals(self):
        p = rule_plan("Will home prices rise in Seattle over the next year?", REG)
        self.assertEqual((p.intent, [l.key for l in p.locations]), ("outlook", ["seattle-wa"]))
        self.assertEqual(p.workers, P.KNOWN_WORKERS)

    def test_any_city_question_runs_every_signal(self):
        for q in ("Are mortgage rates hurting buyers in Boston, MA?", "Is Austin, TX a buyer's market?",
                  "Is school enrollment growing in Denver, CO?"):
            p = rule_plan(q, REG)
            self.assertEqual((p.intent, p.workers), ("outlook", P.KNOWN_WORKERS), q)

    def test_compare_in_order_of_appearance(self):
        p = rule_plan("Compare Denver, Austin and Phoenix", REG)
        self.assertEqual((p.intent, self.keys("Compare Denver, Austin and Phoenix")),
                         ("compare", ["denver-co", "austin-tx", "phoenix-az"]))

    def test_unlisted_city_with_state_and_filler_words(self):
        self.assertEqual(self.keys("Is Springfield, IL a buyer's market?"), ["springfield-il"])
        self.assertEqual(self.keys("Will prices rise in Salt Lake City, Utah?"), ["salt-lake-city-ut"])

    def test_no_city_asks_for_one(self):
        p = rule_plan("Will prices go up?", REG)
        self.assertEqual(p.intent, "clarify")
        self.assertIn("Which city", p.clarification)

    def test_unrelated_is_out_of_scope(self):
        self.assertEqual(rule_plan("What is the weather like?", REG).intent, "out_of_scope")

    def test_unsupported_topics_are_flagged_not_dropped(self):
        p = rule_plan("Is Seattle safe, and should I buy a house there?", REG)
        self.assertEqual(p.intent, "outlook")
        self.assertIn("crime", p.unsupported)
        self.assertIn("buy or sell advice", p.unsupported)
        self.assertIn("a ZIP code that could not be matched to a city", rule_plan("What about 98101?", REG).unsupported)

    def test_model_cannot_invent_city_or_worker(self):
        draft = PlanDraft(intent="outlook", locations=[{"city": "Atlantis", "state": "WA"},
                                                       {"city": "Seattle", "state": "WA"}],
                          workers=["supply", "crime"], reason="r")
        p = build_plan("Will prices rise in Seattle?", REG, FakeJudge(draft))
        self.assertEqual([l.key for l in p.locations], ["seattle-wa"])
        self.assertEqual(p.workers, P.KNOWN_WORKERS)
        self.assertTrue(any("Atlantis" in n for n in p.notes))
        self.assertTrue(any("crime" in n for n in p.notes))

    def test_model_unsupported_topics_still_unioned(self):
        draft = PlanDraft(intent="outlook", locations=[{"city": "Seattle", "state": "WA"}], workers=P.ALL_WORKERS)
        p = build_plan("Seattle prices and crime?", REG, FakeJudge(draft))
        self.assertIn("crime", p.unsupported)

    def test_model_failure_falls_back_to_rules(self):
        p = build_plan("Will prices rise in Seattle?", REG, FakeJudge(exc=ValueError("bad json")))
        self.assertEqual([l.key for l in p.locations], ["seattle-wa"])
        self.assertIn("model failed", p.planner)
        self.assertTrue(any("ValueError" in n for n in p.notes))


def outlook_json():
    from datetime import datetime, timezone
    return Outlook.model_validate({
        "location": {"city": "Seattle", "state": "WA"}, "direction": "up", "score": 0.5, "confidence": 0.6,
        "uncertainty": "moderate", "drivers": [], "dissent": [], "missing_workers": {}, "findings": [],
        "narrative": "Seattle looks upward.", "weights": {}, "generated_at": datetime.now(timezone.utc)})


class AnswerTests(unittest.TestCase):
    facts = {"places": [{"score": 0.5, "metrics": {"zhvi_yoy_pct": 4.2}}]}

    def test_grounding_accepts_numbers_in_facts_and_query(self):
        self.assertIsNone(A.check_grounded("Home values grew 4.2% and the score is 0.5.", self.facts, "q"))
        self.assertIsNone(A.check_grounded("Over 6 to 12 months.", self.facts, "q"))

    def test_grounding_rejects_invented_numbers_and_advice(self):
        self.assertIn("not in the data", A.check_grounded("Prices will grow 9.9%.", self.facts, "q"))
        self.assertIn("advice", A.check_grounded("You should buy now.", self.facts, "q"))

    def test_compose_uses_model_when_grounded_else_template(self):
        plan = Plan(intent="outlook", workers=["supply"], unsupported=["crime"])
        good = FakeJudge(A.AnswerDraft(answer="The score is 0.5 so pressure is upward."))
        text, by, _ = A.compose("q", plan, [outlook_json()], {}, {}, good)
        self.assertEqual(by, "fake")
        self.assertIn("does not cover crime", text)
        bad = FakeJudge(A.AnswerDraft(answer="Prices will rise 77%."))
        text, by, notes = A.compose("q", plan, [outlook_json()], {}, {}, bad)
        self.assertEqual(by, "template")
        self.assertTrue(any("rejected" in n for n in notes))
        self.assertIn("Seattle looks upward.", text)
        self.assertIn("Seattle, WA: ", text)
        text, by, notes = A.compose("q", plan, [outlook_json()], {}, {}, FakeJudge(exc=RuntimeError("x")))
        self.assertEqual(by, "template")


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        cfg = d / "cfg"
        cfg.mkdir()
        for f in ("models.yaml", "test_cities.yaml", "backtest_cities.yaml"):
            shutil.copy(f"config/{f}", cfg / f)
        self.db = str(d / "data" / "cache.db")
        holder = {}

        def make():
            store = RawStore(self.db)
            holder["s"] = store
            return QueryRunner(config_dir=cfg, db_path=self.db, store=store, logger=None,
                               fred_source=fake_fred, open_url=opener_for(zillow_csv()))

        self.jobs = JobThread(make, close=lambda: holder["s"].close())
        self.httpd = serve(self.jobs, port=0, warnings=["demo warning"], monthly_dir=d / "data" / "monthly",
                           config_dir=cfg)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.jobs.stop()
        self.jobs.join(5)
        self.tmp.cleanup()

    def req(self, method, path, body=None, headers=None, host=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        h = {"Host": host or f"127.0.0.1:{self.port}", **(headers or {})}
        c.request(method, path, body=body, headers=h)
        r = c.getresponse()
        data = r.read()
        c.close()
        return r.status, data

    def ask(self, q, **kw):
        body = json.dumps({"query": q, "model": "baseline", **kw})
        return self.req("POST", "/api/ask", body, {"Content-Type": "application/json"})

    def test_page_and_config(self):
        st, data = self.req("GET", "/")
        self.assertEqual(st, 200)
        self.assertIn(b"Ask about a housing market", data)
        st, data = self.req("GET", "/api/config")
        cfg = json.loads(data)
        self.assertEqual(cfg["models"][0], "baseline")
        self.assertEqual(cfg["warnings"], ["demo warning"])
        labels = cfg["model_labels"]
        self.assertEqual(set(labels), set(cfg["models"]))
        self.assertIn("gemma", cfg["models"])
        self.assertIn("local", labels["gemma"])
        self.assertIn("gemma3:4b", labels["gemma"])

    def test_rejects_bad_host_content_type_and_size(self):
        self.assertEqual(self.req("GET", "/api/config", host="evil.example")[0], 403)
        self.assertEqual(self.req("POST", "/api/ask", "{}", {"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.ask("x" * 501)[0], 400)
        self.assertEqual(self.req("POST", "/api/ask", "x" * 9000, {"Content-Type": "application/json"})[0], 413)
        st, _ = self.req("POST", "/api/ask", json.dumps({"query": "hi", "model": "nope"}),
                         {"Content-Type": "application/json"})
        self.assertEqual(st, 400)
        st, _ = self.req("POST", "/api/ask", "{}", {"Content-Type": "application/json", "Origin": "http://evil.example"})
        self.assertEqual(st, 403)

    def events(self, q, **kw):
        st, data = self.ask(q, **kw)
        self.assertEqual(st, 200)
        return [json.loads(l) for l in data.decode().splitlines() if l.strip()]

    def test_full_question_streams_plan_workers_and_answer(self):
        evs = self.events("Will home prices rise in Seattle, WA over the next year?")
        kinds = [e["event"] for e in evs]
        self.assertEqual(kinds[0], "plan")
        self.assertIn("worker_start", kinds)
        self.assertEqual(kinds[-1], "done")
        done = [e for e in evs if e["event"] == "worker_done"]
        ok = {e["worker"] for e in done if e["ok"]}
        self.assertIn("housing_prices", ok)
        self.assertIn("mortgage_rates", ok)
        answer = next(e for e in evs if e["event"] == "answer")
        self.assertEqual(answer["written_by"], "template")
        self.assertEqual(answer["outlooks"][0]["location"]["city"], "Seattle")
        self.assertIn("Seattle", answer["answer"])

    def test_model_writer_is_used_and_checked(self):
        self.jobs.ready.wait(5)
        self.jobs.runner._judges["qwen:long"] = FakeJudge(
            A.AnswerDraft(answer="Seattle looks like it is heading up, with a score of 1.0."))
        evs = self.events("Will home prices rise in Seattle, WA?", writer="qwen")
        ans = next(e for e in evs if e["event"] == "answer")
        self.assertEqual(ans["written_by"], "fake")
        self.assertIn("heading up", ans["answer"])
        self.jobs.runner._judges["qwen:long"] = FakeJudge(A.AnswerDraft(answer="It will rise 88%."))
        ans = next(e for e in self.events("Will home prices rise in Seattle, WA?", writer="qwen")
                   if e["event"] == "answer")
        self.assertEqual(ans["written_by"], "template")
        self.assertTrue(any("rejected" in n for n in ans["notes"]))

    def test_bad_writer_name_is_rejected(self):
        self.assertEqual(self.ask("hi there", writer="baseline")[0], 400)
        self.assertEqual(self.ask("hi there", writer="nope")[0], 400)

    def test_every_worker_runs_for_a_city_question(self):
        evs = self.events("Are mortgage rates a problem in Seattle, WA?")
        ran = {e["worker"] for e in evs if e["event"] == "worker_start"}
        self.assertEqual(ran, set(P.KNOWN_WORKERS))
        ans = next(e for e in evs if e["event"] == "answer")
        self.assertIn("Overall trend", ans["answer"])

    def test_clarify_and_out_of_scope_run_no_workers(self):
        for q in ("Will prices go up?", "What is the weather like?"):
            evs = self.events(q)
            self.assertFalse(any(e["event"] == "worker_start" for e in evs))
            self.assertTrue(next(e for e in evs if e["event"] == "answer")["answer"])

    def test_history_is_saved(self):
        self.events("Will home prices rise in Seattle, WA?")
        st, data = self.req("GET", "/api/history")
        items = json.loads(data)["items"]
        self.assertEqual(items[0]["query"], "Will home prices rise in Seattle, WA?")

    def test_unknown_route(self):
        self.assertEqual(self.req("GET", "/etc/passwd")[0], 404)


if __name__ == "__main__":
    unittest.main()
