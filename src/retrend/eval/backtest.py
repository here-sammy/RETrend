"""Backtest runner: replay the system at past dates, score it against what happened.

Workers run once per (location, date); everything else (ablations, other weights) is
re-combined from the stored reports in code, so it is cheap. The honest limits are
listed in ``CAVEATS`` and printed with every report.
"""

from __future__ import annotations

import calendar
import itertools
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from statistics import mean
from typing import Callable

from retrend.eval import metrics as M
from retrend.eval.truth import ZhviTruth
from retrend.location import Location
from retrend.logging_utils import log_event
from retrend.schema import WorkerReport
from retrend.supervisor import DEFAULT_WEIGHTS, LEGACY_WEIGHTS, Supervisor, new_state

CAVEATS = [
    "Zillow and Redfin revise history and have no vintage API, so housing/supply inputs are "
    "the current files truncated to each date (not point-in-time). Mortgage rates are "
    "point-in-time. Results are optimistic to an unknown degree.",
    "Neighbouring dates share forward windows, and cities move with the national cycle, so "
    "the effective sample size is far smaller than the row count (see n_dates).",
    "If a language model made the judgments, it may remember what happened after the date. "
    "Treat model backtests as upper bounds; the forward ledger (retrend track) is the clean test.",
    "Ground truth is the Zillow Home Value Index, a smoothed modeled series, not sale prices.",
]


def month_end(y: int, m: int) -> date:
    return date(y, m, calendar.monthrange(y, m)[1])


def month_end_dates(start: date, end: date, step_months: int = 1) -> list[date]:
    out, y, m = [], start.year, start.month
    while month_end(y, m) <= end:
        out.append(month_end(y, m))
        m += step_months
        y, m = y + (m - 1) // 12, (m - 1) % 12 + 1
    return out


def end_of_day(d: date) -> datetime:
    return datetime.combine(d, time(23, 59, 59), tzinfo=timezone.utc)


@dataclass
class Case:
    loc: Location
    as_of: date
    origin: date
    fwd_pct: float
    truth: str
    trailing_pct: float
    reports: dict[str, WorkerReport]
    errors: dict[str, str]


@dataclass
class Backtest:
    horizon: int
    threshold: float
    cases: list[Case] = field(default_factory=list)
    coverage: dict[str, list[int]] = field(default_factory=dict)  # worker -> [ok, total]
    skipped: dict[str, int] = field(default_factory=dict)
    fallbacks: dict[str, int] = field(default_factory=dict)  # worker -> reports where a failed model call fell back to the rule


def run_backtest(
    store, workers: dict, locations: list[Location], truth: ZhviTruth, dates: list[date], *,
    horizon: int = 12, logger: logging.Logger | None = None,
    progress: Callable[[str], None] | None = None,
) -> Backtest:
    bt = Backtest(horizon=horizon, threshold=M.threshold_for(horizon))
    bt.coverage = {n: [0, 0] for n in workers}
    bt.fallbacks = {n: 0 for n in workers}
    bt.skipped = {"no_origin": 0, "not_matured": 0, "no_trailing": 0, "load_error": 0}
    sup = Supervisor(store, workers, logger=None, max_attempts=1)
    for loc in locations:
        try:
            truth.series(loc)
        except Exception as exc:
            bt.skipped["load_error"] += len(dates)
            log_event(logger, "backtest_location_skipped", location=loc.key, error=str(exc)[:200])
            if progress:
                progress(f"skip {loc.key}: {type(exc).__name__}: {str(exc)[:100]}")
            continue
        for d in dates:
            origin = truth.origin(loc, d)
            if origin is None:
                bt.skipped["no_origin"] += 1
                continue
            fwd = truth.forward_pct(loc, origin, horizon)
            if fwd is None:
                bt.skipped["not_matured"] += 1
                continue
            trailing = truth.trailing_pct(loc, origin)
            if trailing is None:
                bt.skipped["no_trailing"] += 1
                continue
            state = new_state(loc, end_of_day(d), d)
            sup.run_workers(state)
            for n in workers:
                bt.coverage[n][1] += 1
                bt.coverage[n][0] += n in state["reports"]
                if n in state["reports"] and "fallback" in (state["reports"][n].judged_by or ""):
                    bt.fallbacks[n] += 1
            bt.cases.append(Case(loc, d, origin.day, fwd, M.label(fwd, bt.threshold), trailing,
                                 state["reports"], state["errors"]))
        if progress:
            progress(f"done {loc.key} ({len(bt.cases)} cases so far)")
    return bt


# -- variants ---------------------------------------------------------------------------


def _combiner(store, names: list[str], weights: dict[str, float]) -> Supervisor:
    return Supervisor(store, {}, weights=weights, expected=names, max_attempts=1)


def predictions(store, bt: Backtest, worker_names: list[str],
                weights: dict[str, float] | None = None) -> dict[str, list[tuple[str, float, float, int]]]:
    """variant -> per-case (direction, score, confidence, n_signals), aligned with bt.cases."""
    weights = weights or DEFAULT_WEIGHTS
    variants: dict[str, Supervisor] = {"full": _combiner(store, worker_names, weights)}
    for w in worker_names:
        others = [n for n in worker_names if n != w]
        if others:
            variants[f"no_{w}"] = _combiner(store, others, weights)
        # a zero-weight signal would be all "flat" on its own, so give the one signal full weight
        variants[f"only_{w}"] = _combiner(store, [w], {**weights, w: 1.0})
    if weights == DEFAULT_WEIGHTS:  # keep the pre-backtest weights as a comparison variant
        variants["full_legacy_weights"] = _combiner(store, worker_names, LEGACY_WEIGHTS)
    if weights == DEFAULT_WEIGHTS:
        # does a small vote from an extra signal help? (they are weight 0 in the default mix)
        for extra in ("schools", "jobs", "permits"):
            if extra not in worker_names:
                continue
            for w_s in (0.1, 0.2):
                rest = (1 - w_s) / 2
                variants[f"full_plus_{extra}_{w_s}"] = _combiner(
                    store, worker_names, {**weights, "housing_prices": rest, "supply": rest, extra: w_s})
    out: dict[str, list] = {v: [] for v in variants}
    for c in bt.cases:
        for v, sup in variants.items():
            o = sup.combine(c.loc, c.reports, c.errors, end_of_day(c.as_of), c.as_of)
            n = len([x for x in c.reports if x in sup.expected])
            out[v].append((o.direction, o.score, o.confidence, n))
    out["base_trend"] = [(M.label(c.trailing_pct, M.threshold_for(12)), c.trailing_pct, 0.0, 0)
                         for c in bt.cases]
    out["base_flat"] = [("flat", 0.0, 0.0, 0) for _ in bt.cases]
    out["base_up"] = [("up", 1.0, 0.0, 0) for _ in bt.cases]
    return out


# -- evaluation -------------------------------------------------------------------------


def _score_variant(preds, cases) -> dict:
    d = [p[0] for p in preds]
    t = [c.truth for c in cases]
    f = [c.fwd_pct for c in cases]
    sc = [p[1] for p in preds]
    cs, nd = M.mean_spearman_by_group([c.as_of for c in cases], sc, f)
    return {
        "n": len(cases),
        "accuracy": M.accuracy(d, t),
        "balanced_accuracy": M.balanced_accuracy(d, t),
        "spearman_pooled": M.spearman(sc, f),
        "spearman_cross_section": cs,
        "cross_section_dates": nd,
        **M.direction_spread(d, f),
    }


def evaluate(store, bt: Backtest, worker_names: list[str], *, split: date | None = None,
             weights: dict[str, float] | None = None) -> dict:
    """Score every variant on the common set of cases where the full system had at least one
    signal. With ``split``: 'in_sample' = cases whose forward window ends before the split,
    'out_of_sample' = cases dated on/after it (cases in between are dropped as an embargo)."""
    preds = predictions(store, bt, worker_names, weights)
    keep = [i for i, p in enumerate(preds["full"]) if p[3] >= 1]
    cases = [bt.cases[i] for i in keep]
    preds = {v: [p[i] for i in keep] for v, p in preds.items()}
    report: dict = {
        "horizon_months": bt.horizon, "threshold_pct": bt.threshold,
        "n_cases": len(cases), "n_dates": len({c.as_of for c in cases}),
        "n_locations": len({c.loc.key for c in cases}),
        "label_mix": {l: sum(c.truth == l for c in cases) for l in M.LABELS},
        "coverage": {n: {"ok": ok, "total": tot} for n, (ok, tot) in bt.coverage.items()},
        "skipped": bt.skipped, "model_fallbacks": bt.fallbacks, "caveats": CAVEATS, "weights": weights or DEFAULT_WEIGHTS,
        "variants": {}, "split": split.isoformat() if split else None,
    }
    if not cases:
        report["verdict"] = "No scoreable cases: check data coverage and dates."
        return report

    def months_between(a: date, b: date) -> int:
        return (b.year - a.year) * 12 + b.month - a.month

    periods = {"all": list(range(len(cases)))}
    if split:
        periods["in_sample"] = [i for i, c in enumerate(cases)
                                if months_between(c.as_of, split) >= bt.horizon]
        periods["out_of_sample"] = [i for i, c in enumerate(cases) if c.as_of >= split]
    for period, idx in periods.items():
        if not idx:
            continue
        report["variants"][period] = {
            v: _score_variant([p[i] for i in idx], [cases[i] for i in idx])
            for v, p in preds.items()}
    focus = "out_of_sample" if split and "out_of_sample" in report["variants"] else "all"
    idx = periods[focus]
    report["calibration"] = M.calibration(
        [preds["full"][i][2] for i in idx],
        [preds["full"][i][0] == cases[i].truth for i in idx])
    report["confusion_full"] = M.confusion([preds["full"][i][0] for i in idx],
                                           [cases[i].truth for i in idx])
    report["paired_vs_trend"] = _paired(preds, cases, idx, bt.horizon)
    report["focus_period"] = focus
    report["verdict"] = _verdict(report, focus)
    return report


def paired_diff(rows: list[tuple[date, bool, bool]], horizon: int) -> dict:
    """rows = (date, A correct, B correct). Per-date mean of (A - B), then a block bootstrap over dates."""
    by_date: dict[date, list[float]] = {}
    for d, a, b in rows:
        by_date.setdefault(d, []).append(float(a) - float(b))
    dates = sorted(by_date)
    series = [mean(by_date[d]) for d in dates]
    if len(series) < 6:
        return {"n_dates": len(series), "note": "too few dates for an interval"}
    gap = max(1, round(mean((b.year - a.year) * 12 + b.month - a.month
                            for a, b in zip(dates, dates[1:]))))
    block = max(1, round(horizon / gap))
    m, lo, hi = M.block_bootstrap_ci(series, block)
    return {"mean_accuracy_diff": m, "ci90": [lo, hi], "n_dates": len(series), "block_len": block}


def _paired(preds, cases, idx, horizon) -> dict:
    """Full system versus the plain-trend baseline."""
    return paired_diff([(cases[i].as_of, preds["full"][i][0] == cases[i].truth,
                         preds["base_trend"][i][0] == cases[i].truth) for i in idx], horizon)


def _verdict(report: dict, focus: str) -> str:
    v = report["variants"].get(focus, {})
    full, trend = v.get("full"), v.get("base_trend")
    if not full or not trend:
        return "Not enough data for a verdict."
    p = report["paired_vs_trend"]
    head = (f"[{focus}] system accuracy {full['accuracy']:.1%} vs plain-trend baseline "
            f"{trend['accuracy']:.1%} (n={full['n']}, {p.get('n_dates', 0)} dates).")
    if "ci90" not in p:
        return head + " Too few dates to say whether the difference is real."
    lo, hi = p["ci90"]
    diff = p["mean_accuracy_diff"]
    if lo > 0:
        tail = f" The system beats the baseline by {diff * 100:.1f} points (90% CI {lo * 100:.1f} to {hi * 100:.1f})."
    elif hi < 0:
        tail = f" The system is WORSE than the baseline by {-diff * 100:.1f} points (90% CI {lo * 100:.1f} to {hi * 100:.1f})."
    else:
        tail = (f" The difference ({diff * 100:+.1f} points, 90% CI {lo * 100:.1f} to {hi * 100:.1f}) "
                "is within noise: cannot say the system beats the baseline.")
    naive = max((v[k]["balanced_accuracy"] or 0, k) for k in ("base_flat", "base_up"))
    if (full["balanced_accuracy"] or 0) <= naive[0]:
        tail += f" It also does not beat the naive '{naive[1]}' rule on balanced accuracy."
    return head + tail


# -- weight tuning ----------------------------------------------------------------------


def weight_grid(names: list[str], step: float = 0.1):
    n = round(1 / step)
    for combo in itertools.product(range(n + 1), repeat=len(names) - 1):
        if sum(combo) <= n:
            ws = list(combo) + [n - sum(combo)]
            yield {name: round(w * step, 3) for name, w in zip(names, ws)}


TIE_TOL = 0.01  # train scores this close to the best count as ties; the tie goes to the weights nearest the default


def auto_step(n_signals: int) -> float:
    """A finer grid with many signals is mostly a way to overfit (and slow), so coarsen it."""
    return 0.1 if n_signals <= 4 else 0.2


def tune_weights(store, bt: Backtest, worker_names: list[str], split: date,
                 step: float | None = None) -> dict:
    """Pick weights on the in-sample period only, then report the result out of sample
    next to the default weights, with a confidence interval on the difference. Near-ties
    on the training period go to the weights closest to the defaults, so a signal only
    earns a vote if it clearly helps. The honest answer is often 'keep the default'."""
    step = step or auto_step(len(worker_names))
    cases = [c for c in bt.cases if c.reports]

    def period(c: Case):
        gap = (split.year - c.as_of.year) * 12 + split.month - c.as_of.month
        return "train" if gap >= bt.horizon else "test" if c.as_of >= split else None

    train = [c for c in cases if period(c) == "train"]
    test = [c for c in cases if period(c) == "test"]
    if len(train) < 30 or len(test) < 30:
        return {"note": f"too few cases to tune (train {len(train)}, test {len(test)})"}

    def directions(ws, subset):
        sup = _combiner(store, worker_names, ws)
        return [sup.combine(c.loc, c.reports, c.errors, end_of_day(c.as_of), c.as_of).direction
                for c in subset]

    def acc(ws, subset):
        return M.balanced_accuracy(directions(ws, subset), [c.truth for c in subset]) or 0.0

    default = {n: DEFAULT_WEIGHTS.get(n, 0.0) for n in worker_names}
    total = sum(default.values()) or 1.0
    default = {n: w / total for n, w in default.items()}

    def distance(ws):
        return sum(abs(ws[n] - default[n]) for n in worker_names)

    scored = [(ws, acc(ws, train)) for ws in weight_grid(worker_names, step)]
    top = max(s for _, s in scored)
    best, best_score = min(((ws, s) for ws, s in scored if s >= top - TIE_TOL),
                           key=lambda t: (distance(t[0]), -t[1]))
    truth = [c.truth for c in test]
    d_new, d_def = directions(best, test), directions(DEFAULT_WEIGHTS, test)
    paired = paired_diff([(c.as_of, n == c.truth, o == c.truth)
                          for c, n, o in zip(test, d_new, d_def)], bt.horizon)
    test_new = M.balanced_accuracy(d_new, truth) or 0.0
    test_def = M.balanced_accuracy(d_def, truth) or 0.0
    if "ci90" in paired and paired["ci90"][0] > 0:
        rec = "ADOPT: the chosen weights beat the defaults out of sample by more than noise."
    elif "ci90" in paired and paired["ci90"][1] < 0:
        rec = "KEEP THE DEFAULTS: the chosen weights did clearly worse out of sample (overfit)."
    else:
        rec = "KEEP THE DEFAULTS: any out-of-sample difference is within noise."
    return {
        "chosen_weights": best, "default_weights": default, "grid_step": step, "n_candidates": len(scored),
        "train_balanced_accuracy": best_score, "best_train_balanced_accuracy": top,
        "test_balanced_accuracy_chosen": test_new, "test_balanced_accuracy_default": test_def,
        "test_paired_accuracy_diff": paired, "recommendation": rec,
        "n_train": len(train), "n_test": len(test),
        "note": "Chosen on train only; near-ties go to the weights closest to the defaults. Adopt the new "
                "weights only if the recommendation says ADOPT.",
    }


# -- text report ------------------------------------------------------------------------


def format_report(rep: dict) -> str:
    L = []
    L.append(f"Backtest: horizon {rep['horizon_months']} months, class threshold ±{rep['threshold_pct']:.1f}%")
    L.append(f"Cases {rep['n_cases']} over {rep['n_dates']} dates and {rep['n_locations']} locations; "
             f"outcomes {rep['label_mix']}")
    L.append("Worker coverage (reports produced / attempts):")
    for n, c in rep["coverage"].items():
        L.append(f"  {n:<16} {c['ok']}/{c['total']}")
    L.append(f"Skipped origins: {rep['skipped']}")
    fb = {n: c for n, c in rep.get("model_fallbacks", {}).items() if c}
    if fb:
        L.append(f"WARNING: model calls failed and the rule baseline was used instead: {fb}. "
                 "Those reports are NOT model judgments; fix the model setup before trusting this run.")
    L.append("")
    L.append(rep["verdict"])
    for period, table in rep["variants"].items():
        L.append(f"\n[{period}]")
        L.append(f"{'variant':<22}{'n':>6}{'acc':>8}{'bal.acc':>9}{'rho':>8}{'rho/date':>10}{'spread':>8}")
        order = sorted(table, key=lambda v: (v.startswith("base"), v != "full", v))
        for v in order:
            r = table[v]
            f = lambda x, w, p="": f"{x:>{w}{p}}" if x is not None else f"{'-':>{w}}"  # noqa: E731
            L.append(f"{v:<22}{r['n']:>6}{f(r['accuracy'], 8, '.3f')}{f(r['balanced_accuracy'], 9, '.3f')}"
                     f"{f(r['spearman_pooled'], 8, '.3f')}{f(r['spearman_cross_section'], 10, '.3f')}"
                     f"{f(r['spread'], 8, '.2f')}")
    if "calibration" in rep:
        L.append(f"\nCalibration of the full system's confidence [{rep['focus_period']}]:")
        for b in rep["calibration"]:
            L.append(f"  conf {b['confidence']}: n={b['n']}, accuracy={b['accuracy']}, mean conf={b['mean_confidence']}")
    L.append("\nCaveats:")
    L.extend(f"  - {c}" for c in rep["caveats"])
    return "\n".join(L)
