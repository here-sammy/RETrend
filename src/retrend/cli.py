"""Command line: check the FRED key, or run the mortgage worker with one or more models.

    retrend fred-check
    retrend mortgage seattle-wa --models baseline
    retrend mortgage seattle-wa --models stub               # whole pipeline, no model needed
    retrend mortgage seattle-wa --models qwen,llama         # local models via Ollama
    retrend mortgage seattle-wa --models qwen --as-of 2023-06-30 2024-06-28
    retrend mortgage seattle-wa --models qwen --source mcp  # fetch through the MCP server

(Also runnable as ``python -m retrend.cli`` without installing the console script.)
"""

from __future__ import annotations

import argparse
import json
import re
from time import perf_counter
import sys
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Callable, TextIO

from retrend.config import ModelConfig, RESERVED_NAMES, load_dotenv, load_models
from retrend.fred import FredClient, FredError, parse_observations
from retrend.llm import JsonJudge, LangChainChat, StubChat
from retrend.location import load_registry
from retrend.logging_utils import get_logger
from retrend.store import RawStore
from retrend.sources.files import SourceError
from retrend.sources.redfin import RedfinLoader
from retrend.sources.zillow import ZillowLoader
from retrend.workers.housing import HousingPriceWorker
from retrend.workers.mortgage import SERIES_ID, MortgageRateWorker
from retrend.supervisor import Supervisor
from retrend.workers.supply import SupplyWorker
from retrend.workers.schools import SchoolsWorker
from retrend.sources.schools import SchoolsLoader
from retrend.workers.mortgage_metrics import compute_metrics


def _end_of_day(day: date) -> datetime:
    return datetime.combine(day, time(23, 59, 59), tzinfo=timezone.utc)


def _make_fred_source(kind: str):
    if kind == "mcp":
        from retrend.mcp_client import McpFredSource

        return McpFredSource()
    return FredClient().observations


def _make_fred_search(kind: str):
    """Series search needs direct FRED access (the MCP source only returns observations)."""
    return (lambda text: FredClient().search(text)) if kind != "mcp" else None


def _build_judge(name, models, store, logger) -> JsonJudge | None:
    if name == "baseline":
        return None
    if name == "stub":
        return JsonJudge(StubChat(), store=store, logger=logger)
    judge = JsonJudge(LangChainChat(models[name]), store=store, logger=logger)
    judge.parallel = models[name].parallel
    judge.rpm = models[name].rpm
    return judge


def _build_workers(judge, store, logger, source, open_url, db_path, *, with_schools: bool = False,
                   with_jobs: bool = False, with_permits: bool = False, with_news: bool = False,
                   fred_search=None, config_dir=None) -> dict:
    data_dir = Path(db_path).parent / "files"
    out = {
        "mortgage_rates": MortgageRateWorker(store, logger, fred_source=source, judge=judge),
        "housing_prices": HousingPriceWorker(
            store, logger, judge=judge,
            loader=ZillowLoader(data_dir / "zillow", open_url=open_url)),
        "supply": SupplyWorker(
            store, logger, judge=judge,
            loader=RedfinLoader(data_dir / "redfin", open_url=open_url)),
    }
    if with_schools:  # opt-in: needs a file you supply, so it is not part of every run
        out["schools"] = SchoolsWorker(store, logger, judge=judge, loader=SchoolsLoader(data_dir / "schools"))
    if with_news:  # opt-in: live-only, cannot be backtested
        from retrend.workers.news import NewsWorker

        out["news"] = NewsWorker(store, logger, open_url=open_url, judge=judge)
    if with_jobs or with_permits:  # opt-in: each needs the FRED key and a metro-series lookup
        from retrend.sources.fred_metro import MetroSeriesFinder, load_overrides
        from retrend.workers.jobs import JobsWorker
        from retrend.workers.permits import PermitsWorker

        finder = MetroSeriesFinder(store, fred_search, load_overrides(Path(config_dir or "config") / "fred_series.yaml"))
        if with_jobs:
            out["jobs"] = JobsWorker(store, logger, fred_source=source, finder=finder, judge=judge)
        if with_permits:
            out["permits"] = PermitsWorker(store, logger, fred_source=source, finder=finder, judge=judge)
    return out


def _extras(args) -> dict:
    """Opt-in signals chosen on the command line (--with-schools/--with-jobs/--with-permits)."""
    return {"with_schools": bool(getattr(args, "with_schools", False)),
            "with_jobs": bool(getattr(args, "with_jobs", False)),
            "with_permits": bool(getattr(args, "with_permits", False)),
            "with_news": bool(getattr(args, "with_news", False)),
            "fred_search": _make_fred_search(getattr(args, "source", "direct")),
            "config_dir": getattr(args, "config_dir", None)}


def _outlook(args, location, names, judges, runs, store, logger, source, open_url, say) -> int:
    out_dir = Path(args.db).parent / "reports" / location.key
    out_dir.mkdir(parents=True, exist_ok=True)
    for as_of in runs:
        for name in names:
            judge = judges[name]
            workers = _build_workers(judge, store, logger, source, open_url, args.db, **_extras(args))
            narrator = None if args.no_narrative_model else judge
            sup = Supervisor(store, workers, logger=logger, narrator=narrator)
            label = as_of.date().isoformat() if as_of else "live"
            if args.engine == "langgraph":
                from retrend.supervisor_graph import run_with_graph

                outlook = run_with_graph(sup, location, as_of)
            else:
                outlook = sup.run(location, as_of=as_of)
            say(f"=== {location.city}, {location.state} | model: {name} | as-of: {label} ===")
            say(f"Outlook ({outlook.horizon}): {outlook.direction.upper()}  "
                f"score {outlook.score:+.2f}  confidence {outlook.confidence:.2f}")
            say(outlook.narrative)
            say()
            for title, items in (("Drivers", outlook.drivers), ("Dissent", outlook.dissent)):
                for c in items:
                    label = "Signal" if outlook.direction == "flat" and title == "Drivers" \
                        else title[:-1] if title == "Drivers" else title
                    say(f"  {label}: {c.worker} "
                        f"{c.direction} (conf {c.reported_confidence:.2f} -> "
                        f"{c.effective_confidence:.2f}, share {c.contribution:+.2f}, data as of "
                        f"{c.data_as_of})")
            for c in outlook.context:
                say(f"  Context (not voting): {c.worker} {c.direction} (conf {c.reported_confidence:.2f})")
            for worker, why in outlook.missing_workers.items():
                say(f"  MISSING: {worker}: {why}")
            for f in outlook.findings:
                say(f"  [{f.kind}] {f.worker or ''} {f.detail}")
            say(f"\n{outlook.uncertainty}\n")
            (out_dir / f"outlook_{label}_{name}.json").write_text(
                outlook.model_dump_json(indent=2), encoding="utf-8")
            if args.json:
                say(outlook.model_dump_json(indent=2))
    say(f"Outlooks saved under {out_dir}")
    return 0


def _pick_locations(args, config_dir, say):
    registry = load_registry(config_dir / args.registry)
    keys = [k.strip() for k in args.locations.split(",") if k.strip()] or list(registry)
    unknown = [k for k in keys if k not in registry]
    if unknown:
        say(f"Unknown location(s) {unknown} in {args.registry}. Available: {sorted(registry)}")
        return None, registry
    return [registry[k] for k in keys], registry


def _monthly(args, config_dir, store, logger, source, open_url, say) -> int:
    from retrend import report as rp
    from retrend.eval.ledger import Ledger
    from retrend.eval.truth import ZhviTruth

    locs, _ = _pick_locations(args, config_dir, say)
    if locs is None:
        return 2
    models = load_models(config_dir / "models.yaml")
    name = args.models.strip()
    if name not in models and name not in RESERVED_NAMES:
        say(f"Unknown model {name!r}. Available: {sorted(models) + sorted(RESERVED_NAMES)}")
        return 2
    judge = _build_judge(name, models, store, logger)
    workers = _build_workers(judge, store, logger, source, open_url, args.db)
    truth = ZhviTruth(workers["housing_prices"].zhvi_series)
    ledger = Ledger(store)
    record = args.cmd == "monthly"
    now = store.clock()
    run_id = args.run_id or now.strftime("%Y-%m")
    rows = []
    for i, loc in enumerate(locs, 1):
        try:
            sup = Supervisor(store, workers, logger=logger, narrator=judge if args.narrative_model else None)
            outlook = sup.run(loc)
        except Exception as exc:  # one location must not sink the run
            rows.append(rp.Row(loc.key, loc.city, loc.state, None, error=f"{type(exc).__name__}: {str(exc)[:150]}"))
            say(f"[{i}/{len(locs)}] {loc.key}: FAILED {type(exc).__name__}")
            continue
        prev = rp.previous_snapshot(store, loc.key, name, run_id)
        row = rp.Row(loc.key, loc.city, loc.state, outlook, previous=prev)
        row.changes = rp.diff(outlook, prev)
        if record:
            rp.save_snapshot(store, outlook, name, run_id)
            try:
                ledger.record(outlook, name, truth, horizon_months=args.horizon)
            except Exception as exc:
                say(f"  (ledger not recorded for {loc.key}: {type(exc).__name__})")
        rows.append(row)
        say(f"[{i}/{len(locs)}] {loc.key}: {outlook.direction} {outlook.score:+.2f} conf {outlook.confidence:.2f}"
            + "".join(f"  ** {c.detail}" for c in row.changes if c.severity == "high"))
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.db).parent / "monthly" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.html").write_text(rp.to_html(rows, run_id, name, now), encoding="utf-8")
    (out_dir / "report.md").write_text(rp.to_markdown(rows, run_id, name), encoding="utf-8")
    (out_dir / "report.json").write_text(rp.to_json(rows, run_id, name, now), encoding="utf-8")
    (out_dir / "report.csv").write_text(rp.to_csv(rows), encoding="utf-8")
    s = rp.summarize(rows)
    say(f"\n{s['n_locations']} locations: {s['counts']}; failed {s['n_failed']}")
    if s["direction_changes"]:
        say(f"Direction changed since last run: {', '.join(s['direction_changes'])}")
    say(f"Report written to {out_dir} (open report.html)")
    return 0 if s["n_failed"] < s["n_locations"] else 1


def _eval_commands(args, config_dir, store, logger, source, open_url, say) -> int:
    from retrend.eval import backtest as bt
    from retrend.eval.ledger import Ledger
    from retrend.eval.truth import ZhviTruth

    locs, registry = _pick_locations(args, config_dir, say) if args.cmd != "ledger-score" \
        else (None, load_registry(config_dir / args.registry))
    if args.cmd != "ledger-score" and locs is None:
        return 2
    data_dir_db = args.db

    if args.cmd == "ledger-score":
        zloader = ZillowLoader(Path(args.db).parent / "files" / "zillow", open_url=open_url)
        housing = HousingPriceWorker(store, logger, judge=None, loader=zloader)
        res = Ledger(store).score(registry, ZhviTruth(housing.zhvi_series))
        say(f"Scored {res['scored']} ledger entries; {res['pending']} not matured yet.")
        for m, r in res.get("by_model", {}).items():
            say(f"  {m}: n={r['n']}, system accuracy {r['system_accuracy']}, "
                f"plain-trend baseline {r['baseline_accuracy']}")
        return 0

    if args.cmd == "schools-test":
        from retrend.eval.schools_test import format_schools_test, run_schools_test

        try:
            dates = bt.month_end_dates(date.fromisoformat(args.start), date.fromisoformat(args.end),
                                       args.step_months)
        except ValueError as exc:
            say(f"Error: {exc}")
            return 2
        workers = _build_workers(None, store, logger, None, open_url, data_dir_db, with_schools=True)
        truth = ZhviTruth(workers["housing_prices"].zhvi_series)
        res = run_schools_test(workers["schools"], truth, locs, dates, horizon=args.horizon, progress=say)
        say("\n" + format_schools_test(res))
        out_dir = Path(args.db).parent / "eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"schools_test_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
        path.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
        say(f"\nSaved {path}")
        return 0

    if args.cmd == "compare-models":
        return _compare_models(args, config_dir, store, logger, source, open_url, say, locs)

    models = load_models(config_dir / "models.yaml")
    name = args.models.strip()
    if name not in models and name not in RESERVED_NAMES:
        say(f"Unknown model {name!r}. Available: {sorted(models) + sorted(RESERVED_NAMES)}")
        return 2
    judge = _build_judge(name, models, store, logger)
    workers = _build_workers(judge, store, logger, source, open_url, data_dir_db, **_extras(args))
    truth = ZhviTruth(workers["housing_prices"].zhvi_series)

    if args.cmd == "track":
        ledger = Ledger(store)
        for loc in locs:
            sup = Supervisor(store, workers, logger=logger, narrator=None)
            outlook = sup.run(loc)
            ok = ledger.record(outlook, name, truth, horizon_months=args.horizon)
            say(f"{loc.key}: {outlook.direction} (conf {outlook.confidence:.2f}) "
                f"{'recorded' if ok else 'NOT recorded (no ZHVI origin)'}")
        return 0

    try:
        dates = bt.month_end_dates(date.fromisoformat(args.start), date.fromisoformat(args.end),
                                   args.step_months)
        split = date.fromisoformat(args.split) if args.split else None
    except ValueError as exc:
        say(f"Error: {exc}")
        return 2
    if name != "baseline":
        say("NOTE: a language model is making the judgments. It may remember what happened "
            "after these dates; treat results as an upper bound.\n")
    started = perf_counter()
    if name != "baseline":
        est = len(locs) * len(dates) * 2 + len(dates)  # housing + supply per case; mortgage once per date
        say(f"Up to ~{est} model calls ({len(locs)} locations x {len(dates)} dates). Calls already "
            "in the cache are free, so an interrupted run resumes where it stopped. To cut the "
            "time: --step-months 3 or 6, fewer --locations, a shorter --start/--end, and see "
            "README 'Speeding up a local model'.\n")

    def progress(msg: str) -> None:
        extra = ""
        if judge is not None:
            st = judge.stats
            extra = (f" | model calls {st['calls']}, errors {st['provider_errors']}, bad replies {st['parse_failures']}, "
                     f"cache hits {st['cache_hits']}, "
                     f"{(perf_counter() - started) / 60:.1f} min elapsed")
        say(msg + extra)

    if judge is not None and (getattr(args, "parallel", 1) > 1 or (judge.parallel or 1) > 1):
        from retrend.eval.prefetch import warm_cache
        from retrend.llm import PrefetchAborted
        try:
            warm_cache(judge, lambda: bt.run_backtest(store, workers, locs, truth, dates, horizon=args.horizon),
                       parallel=args.parallel, progress=progress)
        except PrefetchAborted as exc:
            say(f"\nStopped: {exc}")
            return 1
    result = bt.run_backtest(store, workers, locs, truth, dates, horizon=args.horizon,
                             logger=logger, progress=progress)
    if judge is not None:
        st = judge.stats
        say(f"\nModel call stats: calls {st['calls']}, successes {st['successes']}, parse failures "
            f"{st['parse_failures']}, provider errors {st['provider_errors']}, cache hits {st['cache_hits']}")
    names = list(workers)
    report = bt.evaluate(store, result, names, split=split)
    say("\n" + bt.format_report(report))
    if args.tune:
        if not split:
            say("\n--tune needs --split")
        else:
            report["tuning"] = bt.tune_weights(store, result, names, split)
            say("\nWeight tuning: " + json.dumps(report["tuning"], indent=2))
    out_dir = Path(args.db).parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = out_dir / f"backtest_{name}_{stamp}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    say(f"\nSaved {path}")
    return 0


def _compare_models(args, config_dir, store, logger, source, open_url, say, locs) -> int:
    from retrend.eval import backtest as bt
    from retrend.eval import compare as cmp
    from retrend.eval.truth import ZhviTruth

    models = load_models(config_dir / "models.yaml")
    names = [n.strip() for n in args.models.split(",") if n.strip() and n.strip() != "baseline"]
    unknown = [n for n in names if n not in models and n not in RESERVED_NAMES]
    if not names or unknown:
        say(f"Name the models to compare, for example --models qwen,llama,gemma. "
            f"{'Unknown: ' + str(unknown) + '. ' if unknown else ''}Available: {sorted(models) + sorted(RESERVED_NAMES)}. "
            "Add a model to config/models.yaml to make it available.")
        return 2
    try:
        dates = bt.month_end_dates(date.fromisoformat(args.start), date.fromisoformat(args.end), args.step_months)
        split = date.fromisoformat(args.split) if args.split else None
    except ValueError as exc:
        say(f"Error: {exc}")
        return 2
    if args.tune and not split:
        say("--tune needs --split")
        return 2
    judges = {}
    for n in names:
        try:
            judges[n] = _build_judge(n, models, store, logger)
        except Exception as exc:  # a missing package or key for one model should not stop the others
            say(f"Skipping {n}: {type(exc).__name__}: {str(exc)[:200]}")
    if not judges:
        return 2
    est = len(locs) * len(dates) * 2 + len(dates)
    say(f"NOTE: models may remember what happened after these dates, so treat accuracy as an upper bound.\n"
        f"Up to ~{est} model calls per model ({len(locs)} locations x {len(dates)} dates); calls already "
        "cached are free, so an interrupted run resumes.\n")
    extras = _extras(args)

    def build(judge):
        return _build_workers(judge, store, logger, source, open_url, args.db, **extras)

    truth = ZhviTruth(build(None)["housing_prices"].zhvi_series)
    out = cmp.run_comparison(store, build, judges, locs, truth, dates, horizon=args.horizon, split=split,
                             tune=args.tune, progress=say, parallel=args.parallel)
    say("\n" + cmp.format_text(out))
    out_dir = Path(args.db).parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    base = out_dir / f"compare_{stamp}"
    base.with_suffix(".json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    base.with_suffix(".md").write_text(cmp.to_markdown(out), encoding="utf-8")
    base.with_suffix(".csv").write_text(cmp.to_csv(out), encoding="utf-8")
    say(f"\nSaved {base}.md (table), .csv and .json")
    return 0


def _ui(args, config_dir, logger, fred_source, open_url, say) -> int:
    import os
    import webbrowser

    from retrend.ui.runner import QueryRunner
    from retrend.ui.server import JobThread, serve

    warnings = []
    if fred_source is None:
        if not os.environ.get("FRED_API_KEY"):
            warnings.append("FRED_API_KEY is not set, so the mortgage-rate signal will be unavailable. "
                            "Add it to your .env file.")

        def fred_source(*a, **k):  # built lazily so a missing key is a per-signal error, not a crash
            return FredClient().observations(*a, **k)

    holder = {}

    def make():
        store = RawStore(args.db, logger=logger)  # created on the job thread (sqlite is thread-bound)
        holder["store"] = store
        return QueryRunner(config_dir=config_dir, db_path=args.db, store=store, logger=logger,
                           fred_source=fred_source, open_url=open_url,
                           fred_search=lambda text: FredClient().search(text))

    jobs = JobThread(make, close=lambda: holder["store"].close() if "store" in holder else None)
    monthly = Path(args.db).parent / "monthly"
    httpd = serve(jobs, port=args.port, warnings=warnings, monthly_dir=monthly, config_dir=config_dir,
                  ready=lambda url: say(f"Open {url} in your browser. Press Ctrl+C to stop."))
    if not args.no_browser:
        webbrowser.open(f"http://127.0.0.1:{httpd.server_address[1]}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        say("Stopped.")
    finally:
        httpd.server_close()
        jobs.stop()
    return 0


def main(
    argv: list[str] | None = None,
    *,
    fred_source: Callable[..., dict] | None = None,
    out: TextIO | None = None,
    open_url: Callable[[str], object] | None = None,
) -> int:
    out = out or sys.stdout

    def say(text: str = "") -> None:
        print(text, file=out)

    parser = argparse.ArgumentParser(prog="retrend")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--db", default="data/cache.db")
    parser.add_argument("--log", default="logs/run.jsonl")
    sub = parser.add_subparsers(dest="cmd", required=True)

    check = sub.add_parser("fred-check", help="fetch MORTGAGE30US and print the metrics")
    check.add_argument("--source", choices=["direct", "mcp"], default="direct")

    for cmd, text in (("mortgage", "run the mortgage-rate worker"),
                      ("housing", "run the housing-price worker (Zillow)"),
                      ("supply", "run the supply/inventory worker (Redfin)")):
        mort = sub.add_parser(cmd, help=text)
        mort.add_argument("location", help="key from config/test_cities.yaml, e.g. seattle-wa")
        mort.add_argument("--models", default="baseline",
                          help="comma-separated names from models.yaml, plus 'baseline' (rule "
                               "only) and 'stub' (canned reply, no model)")
        mort.add_argument("--as-of", nargs="*", default=[], metavar="YYYY-MM-DD",
                          help="run as of a past date (mortgage: FRED point-in-time; housing/"
                               "supply: stored snapshot if any, else truncated and flagged)")
        mort.add_argument("--source", choices=["direct", "mcp"], default="direct",
                          help="mortgage only: how to fetch FRED")
        mort.add_argument("--no-fallback", action="store_true",
                          help="fail instead of falling back to the rule when a model fails")
        mort.add_argument("--json", action="store_true", help="print full reports as JSON")

    outl = sub.add_parser("outlook", help="run all workers and the supervisor")
    outl.add_argument("location")
    outl.add_argument("--models", default="baseline",
                      help="model(s) used by the workers AND to write the narrative; one "
                           "full run per name. 'baseline' = rules only, 'stub' = canned")
    outl.add_argument("--as-of", nargs="*", default=[], metavar="YYYY-MM-DD")
    outl.add_argument("--source", choices=["direct", "mcp"], default="direct")
    outl.add_argument("--engine", choices=["plain", "langgraph"], default="plain")
    outl.add_argument("--no-narrative-model", action="store_true",
                      help="use the template narrative even when a model is available")
    outl.add_argument("--json", action="store_true")

    bt = sub.add_parser("backtest", help="replay the system at past dates and score it")
    bt.add_argument("--registry", default="backtest_cities.yaml",
                    help="file in config/ listing the locations")
    bt.add_argument("--locations", default="", help="comma-separated keys to use (default: all)")
    bt.add_argument("--start", required=True, metavar="YYYY-MM-DD")
    bt.add_argument("--end", required=True, metavar="YYYY-MM-DD")
    bt.add_argument("--step-months", type=int, default=1)
    bt.add_argument("--horizon", type=int, default=12, help="months ahead to score")
    bt.add_argument("--models", default="baseline",
                    help="one name from models.yaml, or 'baseline' (rules only: cheap, no "
                         "model memory problem)")
    bt.add_argument("--split", metavar="YYYY-MM-DD",
                    help="report in-sample vs out-of-sample around this date")
    bt.add_argument("--tune", action="store_true", help="tune weights in-sample (needs --split)")
    bt.add_argument("--parallel", type=int, default=1, metavar="N",
                    help="ask the model N questions at once (default 1). Hosted APIs take 8 or more; a local "
                         "Ollama needs OLLAMA_NUM_PARALLEL=N set before it starts, or it just queues them")
    bt.add_argument("--source", choices=["direct", "mcp"], default="direct")

    cm = sub.add_parser("compare-models", help="run several models on the same backtest and print one comparison table")
    cm.add_argument("--registry", default="backtest_cities.yaml")
    cm.add_argument("--locations", default="")
    cm.add_argument("--start", required=True, metavar="YYYY-MM-DD")
    cm.add_argument("--end", required=True, metavar="YYYY-MM-DD")
    cm.add_argument("--step-months", type=int, default=3)
    cm.add_argument("--horizon", type=int, default=12)
    cm.add_argument("--parallel", type=int, default=1, metavar="N",
                    help="ask the model N questions at once (default 1). Hosted APIs take 8 or more; a local "
                         "Ollama needs OLLAMA_NUM_PARALLEL=N set before it starts, or it just queues them")
    cm.add_argument("--models", required=True, help="comma-separated names from models.yaml, e.g. qwen,llama,gemma")
    cm.add_argument("--split", metavar="YYYY-MM-DD", help="needed for --tune; scores the tuning out of sample")
    cm.add_argument("--tune", action="store_true", help="also tune the signal weights for each model (needs --split)")
    cm.add_argument("--source", choices=["direct", "mcp"], default="direct")
    for flag, what in (("--with-schools", "schools"), ("--with-jobs", "jobs"), ("--with-permits", "permits")):
        cm.add_argument(flag, action="store_true", help=f"also run the {what} worker")

    bt.add_argument("--with-schools", action="store_true",
                    help="also run the schools worker and add schools variants to the report")
    for flag, what in (("--with-jobs", "local jobs (metro unemployment and payrolls)"),
                       ("--with-permits", "building permits (set RETREND_PERMITS_RULE=demand or supply to take a side)")):
        bt.add_argument(flag, action="store_true", help=f"also run the {what} worker")
        outl.add_argument(flag, action="store_true", help=f"also run the {what} worker")
    outl.add_argument("--with-schools", action="store_true", help="also run the schools worker")
    outl.add_argument("--with-news", action="store_true", help="also run the news-tone worker (live only; GDELT)")

    st = sub.add_parser("schools-test", help="test whether school enrollment predicts later price growth")
    st.add_argument("--registry", default="backtest_cities.yaml")
    st.add_argument("--locations", default="")
    st.add_argument("--start", required=True, metavar="YYYY-MM-DD")
    st.add_argument("--end", required=True, metavar="YYYY-MM-DD")
    st.add_argument("--step-months", type=int, default=6)
    st.add_argument("--horizon", type=int, default=12)

    trk = sub.add_parser("track", help="record today's outlooks in the forward ledger")
    trk.add_argument("--registry", default="test_cities.yaml")
    trk.add_argument("--locations", default="")
    trk.add_argument("--models", default="baseline")
    trk.add_argument("--horizon", type=int, default=12)
    trk.add_argument("--source", choices=["direct", "mcp"], default="direct")

    for cmd, text in (("monthly", "run all locations, save the run, flag changes since last run, write a report"),
                      ("compare", "rank locations side by side (saves nothing)")):
        mp = sub.add_parser(cmd, help=text)
        mp.add_argument("--registry", default="backtest_cities.yaml")
        mp.add_argument("--locations", default="", help="comma-separated keys (default: all)")
        mp.add_argument("--models", default="baseline", help="one name from models.yaml, or baseline")
        mp.add_argument("--source", choices=["direct", "mcp"], default="direct")
        mp.add_argument("--narrative-model", action="store_true",
                        help="let the model write each city's narrative (slower); default is a template")
        mp.add_argument("--out-dir", default="", help="default: <db folder>/monthly/<run id>")
        mp.add_argument("--run-id", default="", help="default: current YYYY-MM")
        mp.add_argument("--horizon", type=int, default=12)

    sf = sub.add_parser("schools-fetch", help="download district enrollment history (Urban Institute, free) and save it")
    sf.add_argument("--start", type=int, default=None, help="first school (fall) year; default: 19 before --end (20 years)")
    sf.add_argument("--end", type=int, default=None, help="last school (fall) year; default: newest with data")
    sf.add_argument("--timeout", type=int, default=120, help="seconds to wait per request (default 120)")
    sf.add_argument("--force", action="store_true", help="re-download years already saved")
    sf.add_argument("--csv-out", default="", help="default: <db folder>/files/schools/urban_enrollment.csv")

    zf = sub.add_parser("zip-fetch", help="download the free ZIP code -> city table (GeoNames) and save it")
    zf.add_argument("--csv-out", default="", help="default: <db folder>/files/zip/us_zip.csv")

    uip = sub.add_parser("ui", help="open the question screen in your browser (local only)")
    uip.add_argument("--port", type=int, default=8765)
    uip.add_argument("--no-browser", action="store_true")

    lsc = sub.add_parser("ledger-score", help="score matured ledger entries")
    lsc.add_argument("--registry", default="test_cities.yaml")

    args = parser.parse_args(argv)
    load_dotenv(args.env_file)
    config_dir = Path(args.config_dir)
    logger = get_logger("retrend.cli", args.log)
    if args.cmd == "schools-fetch":
        from retrend.sources import urban
        from retrend.sources.urban import _getter, export_csv, fetch_years, latest_available_year

        urban.TIMEOUT = args.timeout

        try:
            end = args.end if args.end is not None else latest_available_year(_getter(open_url))
        except SourceError as exc:
            say(f"Error: {exc}")
            return 2
        start = args.start if args.start is not None else end - 19
        say(f"Fetching school years {start} to {end}.")
        summary = fetch_years(args.db, list(range(start, end + 1)), open_url=open_url,
                              say=say, force=args.force)
        out_csv = args.csv_out or str(Path(args.db).parent / "files" / "schools" / "urban_enrollment.csv")
        n = export_csv(args.db, out_csv)
        say(f"Saved {len(summary['fetched'])} year(s), skipped {len(summary['skipped'])}, "
            f"failed {len(summary['failed'])}. Wrote {n} rows to {out_csv}")
        return 0 if not summary["failed"] else 1
    if args.cmd == "zip-fetch":
        from retrend.sources.zipcodes import ZipError, fetch_zip_csv

        out_csv = args.csv_out or str(Path(args.db).parent / "files" / "zip" / "us_zip.csv")
        try:
            n = fetch_zip_csv(out_csv, open_url=open_url, say=say)
        except ZipError as exc:
            say(f"Error: {exc}")
            return 2
        say(f"Wrote {n} ZIP codes to {out_csv}")
        return 0
    if args.cmd == "ui":
        return _ui(args, config_dir, logger, fred_source, open_url, say)
    store = RawStore(args.db, logger=logger)
    try:

        try:
            source = None
            if args.cmd in ("fred-check", "mortgage", "outlook", "backtest", "compare-models", "track", "monthly", "compare"):
                source = fred_source or _make_fred_source(args.source)

            if args.cmd == "fred-check":
                payload = source(SERIES_ID)
                obs = parse_observations(payload)
                if not obs:
                    say("FRED returned no observations.")
                    return 1
                say(f"Fetched {len(obs)} weekly observations. Last five:")
                for o in obs[-5:]:
                    say(f"  {o.day}  {o.value:.2f}%")
                say("\nMetrics:")
                for key, value in compute_metrics(obs, date.today()).items():
                    say(f"  {key}: {value}")
                return 0

            if args.cmd in ("monthly", "compare"):
                return _monthly(args, config_dir, store, logger, source, open_url, say)
            if args.cmd in ("backtest", "compare-models", "track", "ledger-score", "schools-test"):
                return _eval_commands(args, config_dir, store, logger, source, open_url, say)

            locations = load_registry(config_dir / "test_cities.yaml")
            if re.fullmatch(r"\d{5}", args.location):  # a ZIP code: use its city
                from retrend.sources.zipcodes import ZipError, ZipResolver
                try:
                    place = ZipResolver(Path(args.db).parent / "files" / "zip").resolve(args.location)
                except ZipError as exc:
                    say(f"Error: {exc}")
                    return 2
                from retrend.location import Location
                guess = Location(city=place.city, state=place.state)
                locations = {**locations, args.location: locations.get(guess.key, guess)}
                say(f"ZIP {args.location} -> {place.city}, {place.state}")
            if args.location not in locations:
                say(f"Unknown location {args.location!r}. Available: {sorted(locations)}")
                return 2
            location = locations[args.location]
            models = load_models(config_dir / "models.yaml")
            names = [n.strip() for n in args.models.split(",") if n.strip()]
            unknown = [n for n in names if n not in models and n not in RESERVED_NAMES]
            if unknown:
                say(f"Unknown model(s) {unknown}. Available: {sorted(models) + sorted(RESERVED_NAMES)}")
                return 2

            judges = {n: _build_judge(n, models, store, logger) for n in names}
            runs = [_end_of_day(date.fromisoformat(d)) for d in args.as_of] or [None]
            if args.cmd == "outlook":
                return _outlook(args, location, names, judges, runs, store, logger, source,
                                open_url, say)
        except (FredError, SourceError, ValueError, RuntimeError) as exc:
            say(f"Error: {exc}")
            return 2

        rows, reports = [], []
        for as_of in runs:
            for name in names:
                fb = not args.no_fallback
                if args.cmd == "mortgage":
                    worker = MortgageRateWorker(store, logger, fred_source=source,
                                                judge=judges[name], fallback_to_baseline=fb)
                elif args.cmd == "housing":
                    worker = HousingPriceWorker(
                        store, logger, judge=judges[name], fallback_to_baseline=fb,
                        loader=ZillowLoader(Path(args.db).parent / "files" / "zillow",
                                            open_url=open_url))
                else:
                    worker = SupplyWorker(
                        store, logger, judge=judges[name], fallback_to_baseline=fb,
                        loader=RedfinLoader(Path(args.db).parent / "files" / "redfin",
                                            open_url=open_url))
                label = as_of.date().isoformat() if as_of else "live"
                try:
                    report = worker.run(location, as_of=as_of)
                except Exception as exc:  # report and keep going: other models may work
                    rows.append((name, label, f"ERROR: {type(exc).__name__}: {str(exc)[:90]}"))
                    continue
                reports.append((name, label, report))
                km = report.key_metrics
                rows.append((name, label, report.direction, f"{report.confidence:.2f}",
                             km["baseline_direction"],
                             "yes" if report.direction == km["baseline_direction"] else "no",
                             str(km["judge_attempts"]), "yes" if km["judge_cached"] else "no",
                             report.judged_by or ""))

        say(f"{'model':<10} {'as-of':<11} {'direction':<9} {'conf':<5} {'baseline':<9} "
            f"{'agrees':<7} {'tries':<5} {'cached':<6} judged_by")
        for row in rows:
            if len(row) == 3:
                say(f"{row[0]:<10} {row[1]:<11} {row[2]}")
            else:
                say(f"{row[0]:<10} {row[1]:<11} {row[2]:<9} {row[3]:<5} {row[4]:<9} {row[5]:<7} "
                    f"{row[6]:<5} {row[7]:<6} {row[8]}")
        say()
        for name, label, report in reports:
            say(f"[{name} | {label}] {report.summary}")
            for caveat in report.caveats:
                say(f"    - {caveat}")
        stats = {n: j.stats for n, j in judges.items() if j is not None}
        if stats:
            say("\nModel call stats (calls, parse_failures, provider_errors, cache_hits):")
            for n, s in stats.items():
                say(f"  {n}: {s['calls']}, {s['parse_failures']}, {s['provider_errors']}, "
                    f"{s['cache_hits']}")

        out_dir = Path(args.db).parent / "reports" / location.key
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, label, report in reports:
            (out_dir / f"{args.cmd}_{label}_{name}.json").write_text(
                report.model_dump_json(indent=2), encoding="utf-8"
            )
        say(f"\nReports saved under {out_dir}")
        if args.json:
            say(json.dumps([json.loads(r.model_dump_json()) for _, _, r in reports], indent=2))
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
