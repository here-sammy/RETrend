"""Runs one question end to end and emits progress events.

One QueryRunner owns the RawStore (sqlite connections are thread-bound), so the server feeds
it from a single worker thread.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import replace
from datetime import timezone
from pathlib import Path
from typing import Callable

from retrend.config import load_dotenv, load_models
from retrend.llm import JsonJudge, LangChainChat, StubChat
from retrend.location import Location, load_registry
from retrend.outlook import Outlook
from retrend.supervisor import Supervisor
from retrend.ui.answer import compose, school_answer, school_card
from retrend.sources.zipcodes import ZipError, ZipResolver
from retrend.ui.planner import WORKER_LABELS, Plan, build_plan

MAX_PLACES = 8
MAX_PLACES_RULES = 12  # every signal runs for every place, so keep comparisons small
HISTORY_SOURCE = "ui_query"
Emit = Callable[[dict], None]


class Traced:
    """Wraps a worker so each run is reported as events."""

    def __init__(self, inner, emit: Emit, sink: dict, errors: dict, reports: dict | None = None):
        self.inner, self.emit, self.sink, self.errors = inner, emit, sink, errors
        self.reports = reports if reports is not None else {}
        self.name = inner.name

    def run(self, location: Location, *, as_of=None):
        self.emit({"event": "worker_start", "worker": self.name, "place": location.key})
        t0 = time.perf_counter()
        try:
            rep = self.inner.run(location, as_of=as_of)
        except Exception as exc:
            ms = round((time.perf_counter() - t0) * 1000)
            self.errors[f"{self.name}@{location.key}"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            self.emit({"event": "worker_done", "worker": self.name, "place": location.key, "ok": False,
                       "ms": ms, "error": f"{type(exc).__name__}: {str(exc)[:160]}"})
            raise
        self.sink.setdefault(location.key, {})[self.name] = dict(rep.key_metrics)
        self.reports[(self.name, location.key)] = rep
        self.emit({"event": "worker_done", "worker": self.name, "place": location.key, "ok": True,
                   "ms": round((time.perf_counter() - t0) * 1000), "direction": rep.direction,
                   "confidence": rep.confidence, "summary": rep.summary,
                   "data_as_of": rep.data_as_of.isoformat(), "judged_by": rep.judged_by})
        return rep


class QueryRunner:
    def __init__(self, *, config_dir, db_path, store, logger, fred_source, open_url=None, fred_search=None):
        self.config_dir = Path(config_dir)
        self.db_path = db_path
        self.store, self.logger = store, logger
        self.fred_source, self.open_url, self.fred_search = fred_source, open_url, fred_search
        self.zips = ZipResolver(Path(db_path).parent / "files" / "zip")
        self.models = load_models(self.config_dir / "models.yaml")
        self.registry = {}
        for f in ("test_cities.yaml", "backtest_cities.yaml"):
            p = self.config_dir / f
            if p.exists():
                self.registry.update(load_registry(p))
        self._judges: dict[str, JsonJudge | None] = {}

    def judge(self, name: str, long_output: bool = False) -> JsonJudge | None:
        """long_output: the answer writer needs room for several paragraphs, so token limits
        set for the short per-signal replies are raised for it."""
        if name == "baseline":
            return None
        key = name + (":long" if long_output else "")
        if key not in self._judges:
            if name == "stub":
                chat = StubChat()
            else:
                cfg = self.models[name]
                if long_output:
                    params = dict(cfg.params)
                    if "num_predict" in params:
                        params["num_predict"] = max(int(params["num_predict"]), 1200)
                    if "num_ctx" in params:
                        params["num_ctx"] = max(int(params["num_ctx"]), 4096)
                    cfg = replace(cfg, params=params)
                chat = LangChainChat(cfg)
            self._judges[key] = JsonJudge(chat, store=self.store, logger=self.logger)
        return self._judges[key]

    def model_names(self) -> list[str]:
        return ["baseline", *self.models]

    def history(self, limit: int = 5) -> list[dict]:
        """The most recent `limit` different questions, newest first."""
        out, seen = [], set()
        for key in sorted(self.store.keys(HISTORY_SOURCE), reverse=True):
            recs = self.store.history(HISTORY_SOURCE, key)
            if not recs:
                continue
            try:
                rec = recs[-1].json()
            except Exception:
                continue
            norm_q = " ".join(str(rec.get("query", "")).lower().split())
            if not norm_q or norm_q in seen:
                continue
            seen.add(norm_q)
            out.append(rec)
            if len(out) >= limit:
                break
        return out

    def _build_workers(self, judge, emit, sink, errors, reports=None):
        from retrend.cli import _build_workers
        raw = _build_workers(judge, self.store, self.logger, self.fred_source, self.open_url, self.db_path, with_schools=True,
                           with_jobs=True, with_permits=True, with_news=True, fred_search=self.fred_search, config_dir=self.config_dir)
        return {k: Traced(v, emit, sink, errors, reports) for k, v in raw.items()}

    def handle(self, query: str, model: str, judge_signals: bool, emit: Emit, writer: str = "template") -> dict:
        t_start = time.perf_counter()
        if model not in self.model_names():
            model = "baseline"
        planner_judge = None
        try:
            planner_judge = self.judge(model)
        except Exception as exc:
            emit({"event": "note", "text": f"Model {model} could not be loaded ({type(exc).__name__}); using rules."})
            model = "baseline"
        planning_query, zip_notes = self._resolve_zips(query)
        plan = build_plan(planning_query, self.registry, planner_judge)
        plan.notes = zip_notes + plan.notes
        writer_judge = None
        if writer != "template" and plan.workers != ["schools"]:  # a school-only lookup needs no writer
            try:
                writer_judge = self.judge(writer, long_output=True)
            except Exception as exc:
                emit({"event": "note", "text": f"Answer writer {writer} could not be loaded ({type(exc).__name__}); using the template."})
        if plan.compare_all:
            plan.locations = list(load_registry(self.config_dir / "test_cities.yaml").values())
        cap = MAX_PLACES if (judge_signals and planner_judge) else MAX_PLACES_RULES
        if len(plan.locations) > cap:
            plan.notes.append(f"Limited to the first {cap} places.")
            plan.locations = plan.locations[:cap]
        emit({"event": "plan", "plan": plan.model_dump(mode="json"),
              "worker_labels": dict(WORKER_LABELS)})

        result: dict = {"plan": plan.model_dump(mode="json"), "outlooks": [], "answer": "", "written_by": "template",
                        "notes": list(plan.notes), "errors": {}}
        if plan.intent in ("clarify", "out_of_scope") or not plan.locations:
            if plan.intent == "out_of_scope":
                msg = ("This tool answers questions about housing-market signals for U.S. cities. "
                       "Try 'Will home prices rise in Seattle, WA?'")
            else:
                msg = plan.clarification or "Name a city and state."
            result["answer"] = msg
            emit({"event": "answer", **{k: result[k] for k in ("answer", "written_by")}, "notes": result["notes"]})
            return self._finish(query, model, judge_signals, plan, result, t_start, emit)

        if plan.workers == ["schools"]:
            return self._schools_only(query, model, judge_signals, plan, result, t_start, emit)

        sink: dict = {}
        errors: dict = {}
        worker_judge = planner_judge if judge_signals else None
        outlooks: list[Outlook] = []
        reports: dict = {}
        for loc in plan.locations:
            emit({"event": "place_start", "place": loc.key, "name": f"{loc.city}, {loc.state}"})
            workers = self._build_workers(worker_judge, emit, sink, errors, reports)
            sup = Supervisor(self.store, workers, logger=self.logger, narrator=None,
                             expected=list(plan.workers))
            try:
                o = sup.run(loc)
            except Exception as exc:
                errors[loc.key] = f"{type(exc).__name__}: {str(exc)[:160]}"
                emit({"event": "place_done", "place": loc.key, "ok": False, "error": errors[loc.key]})
                continue
            outlooks.append(o)
            emit({"event": "place_done", "place": loc.key, "ok": True, "direction": o.direction,
                  "score": o.score, "confidence": o.confidence})
        from retrend.supervisor import DEFAULT_WEIGHTS
        context_only = bool(outlooks) and all(DEFAULT_WEIGHTS.get(w, 0.0) <= 0 for w in plan.workers)
        emit({"event": "composing"})
        text, by, notes = compose(query, plan, outlooks, sink, errors, writer_judge)
        if context_only:
            notes.append("The signals asked for (mortgage rates, school enrollment) are context only and do not vote, so no price outlook is given.")
        headlines = {}
        for (w, key), rep in reports.items():
            if w == "news":
                try:
                    items = json.loads(rep.key_metrics.get("headlines_json") or "[]")
                except ValueError:
                    items = []
                if items:
                    headlines[f"{rep.location.city}, {rep.location.state}"] = items
        result.update(headlines=headlines)
        result.update(outlooks=[o.model_dump(mode="json") for o in outlooks], answer=text, written_by=by,
                      errors=errors, context_only=context_only, notes=result["notes"] + notes)
        emit({"event": "answer", "answer": text, "written_by": by, "notes": result["notes"],
              "outlooks": result["outlooks"], "headlines": headlines})
        return self._finish(query, model, judge_signals, plan, result, t_start, emit)

    # -- tab lookups (plain data, no model) ----------------------------------------------

    def _resolve_zips(self, query: str) -> tuple[str, list[str]]:
        """Swap each ZIP in the question for its 'City, ST'. -> (question to plan with, notes)."""
        try:
            text, places, unresolved = self.zips.rewrite(query)
        except ZipError as exc:  # e.g. an unreadable ZIP file
            return query, [f"ZIP codes could not be looked up: {exc}"]
        notes = [f"ZIP {p.zip} is in {p.city}, {p.state}, so I used {p.city}, {p.state}." for p in places]
        if unresolved:
            if not self.zips.available():
                notes.append("ZIP codes need a ZIP file. Run 'retrend zip-fetch' once to download it, then ask again.")
            else:
                notes.append("ZIP " + ", ".join(unresolved) + " is not in the ZIP file.")
        return text, notes

    def _location(self, city: str, state: str, zip_code: str = "") -> Location:
        if zip_code.strip():
            try:
                place = self.zips.resolve(zip_code)
            except ZipError as exc:
                raise ValueError(str(exc)) from None
            city, state = place.city, place.state
        city = " ".join(city.split())
        if not city or len(city) > 60 or not re.fullmatch(r"[A-Za-z][A-Za-z .'\-]*", city):
            raise ValueError("enter a city name using letters only")
        state = state.strip().upper()
        if not re.fullmatch(r"[A-Z]{2}", state):
            raise ValueError("enter the state as a two-letter code, such as WA")
        city = " ".join(w[:1].upper() + w[1:] for w in city.split())
        loc = Location(city=city, state=state)
        return self.registry.get(loc.key, loc)

    def schools_view(self, city: str, state: str, zip_code: str = "") -> dict:
        loc = self._location(city, state, zip_code)
        reports: dict = {}
        worker = self._build_workers(None, lambda e: None, {}, {}, reports)["schools"]
        worker.run(loc)
        card = school_card(loc, reports[("schools", loc.key)])
        card["city"], card["state"], card["zip"] = loc.city, loc.state, zip_code.strip()
        return card

    def prices_view(self, city: str, state: str, zip_code: str = "") -> dict:
        from retrend.ui.prices import price_view
        loc = self._location(city, state, zip_code)
        from retrend.cli import _build_workers
        workers = _build_workers(None, self.store, self.logger, self.fred_source, self.open_url,
                                 self.db_path)
        dom = None
        try:
            dom = workers["supply"].dom_series(loc)
        except Exception:  # days on market is optional; the chart works without it
            pass
        view = price_view(loc, workers["housing_prices"].series_bundle(loc), dom)
        view.update(city=loc.city, state=loc.state, zip=zip_code.strip())
        return view

    def _schools_only(self, query, model, judge_signals, plan, result, t_start, emit):
        """A plain look-up of enrollment history: no outlook, no model needed."""
        sink: dict = {}
        errors: dict = {}
        reports: dict = {}
        cards = []
        for loc in plan.locations:
            emit({"event": "place_start", "place": loc.key, "name": f"{loc.city}, {loc.state}"})
            worker = self._build_workers(None, emit, sink, errors, reports)["schools"]
            try:
                worker.run(loc)
            except Exception:
                continue  # Traced already reported it and recorded the error
            cards.append(school_card(loc, reports[("schools", loc.key)]))
        text = school_answer(cards, errors)
        result.update(answer=text, written_by="direct lookup", errors=errors, schools=cards)
        emit({"event": "answer", "answer": text, "written_by": "direct lookup", "notes": result["notes"],
              "outlooks": [], "schools": cards})
        return self._finish(query, model, judge_signals, plan, result, t_start, emit)

    def _finish(self, query, model, judge_signals, plan, result, t_start, emit):
        now = self.store.clock()
        stats = {}
        record = {"at": now.astimezone(timezone.utc).isoformat(), "query": query, "model": model,
                  "judge_signals": judge_signals, "intent": plan.intent,
                  "places": [l.key for l in plan.locations], "workers": plan.workers,
                  "answer": result["answer"][:1500]}
        try:
            self.store.put(HISTORY_SOURCE, now.strftime("%Y%m%dT%H%M%S%f"), record)
        except Exception:
            pass
        emit({"event": "done", "seconds": round(time.perf_counter() - t_start, 1)})
        result["seconds"] = round(time.perf_counter() - t_start, 1)
        return result
