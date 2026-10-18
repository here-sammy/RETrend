"""Compare language models (and the plain rules) on the same backtest, in one table.

Every model judges the same workers at the same (location, date) cases. Scoring uses only the cases
where EVERY model produced real judgments, so no model is graded on easier cases than another and no
rule-based fallback is credited to a model. The table reports accuracy next to what it cost: calls, parse
failures, provider errors, fallbacks and time.
"""

from __future__ import annotations

import csv
import io
import time
from datetime import date
from typing import Callable

from retrend.eval import backtest as bt
from retrend.eval import metrics as M
from retrend.eval.truth import ZhviTruth
from retrend.supervisor import DEFAULT_WEIGHTS

RULES = "rules"
BASELINES = ("base_trend", "base_flat", "base_up")
BASELINE_LABELS = {"base_trend": "plain trend (last 12 months continues)", "base_flat": "always flat",
                   "base_up": "always up"}
STAT_KEYS = ("calls", "cache_hits", "parse_failures", "provider_errors", "failures")


def _key(c) -> tuple[str, date]:
    return (c.loc.key, c.as_of)


def _fallback_case(c) -> bool:
    return any("fallback" in (r.judged_by or "") for r in c.reports.values())


def run_comparison(store, build_workers: Callable, judges: dict, locs, truth: ZhviTruth, dates, *,
                   horizon: int = 12, split: date | None = None, tune: bool = False,
                   progress: Callable[[str], None] | None = None, clock=time.perf_counter,
                   parallel: int = 1) -> dict:
    """``judges`` maps model name -> JsonJudge. The rule baseline always runs first as the reference.
    ``build_workers(judge)`` returns the workers dict for that judge (None = rules)."""
    runs: dict[str, dict] = {}
    if parallel > 1 or any(j.parallel and j.parallel > 1 for j in judges.values()):
        # ask every model's questions at the same time, then each model's run replays from the cache
        from retrend.eval.prefetch import warm_many
        items = []
        for name, judge in judges.items():
            w = build_workers(judge)
            items.append((name, judge, lambda w=w: bt.run_backtest(store, w, locs, truth, dates, horizon=horizon)))
        warm_many(items, parallel=parallel, progress=progress)
    for name, judge in {RULES: None, **judges}.items():
        before = dict(judge.stats) if judge is not None else {}
        t0 = clock()
        workers = build_workers(judge)
        say = (lambda m, n=name: progress(f"[{n}] {m}")) if progress else None
        result = bt.run_backtest(store, workers, locs, truth, dates, horizon=horizon, progress=say)
        stats = {k: judge.stats[k] - before.get(k, 0) for k in STAT_KEYS} if judge is not None else {}
        runs[name] = {"bt": result, "names": list(workers), "seconds": clock() - t0, "stats": stats}

    # per-model predictions with the default weights, keyed by case
    table: dict[str, dict] = {}
    for name, r in runs.items():
        preds = bt.predictions(store, r["bt"], r["names"])
        r["preds"] = {v: {_key(c): p for c, p in zip(r["bt"].cases, ps)} for v, ps in preds.items()}
        r["fallback_keys"] = {_key(c) for c in r["bt"].cases if _fallback_case(c)}
        r["cases"] = {_key(c): c for c in r["bt"].cases}

    ref = runs[RULES]
    common = [k for k in ref["cases"]
              if all(k in r["cases"] and r["preds"]["full"][k][3] >= 1 and k not in r["fallback_keys"]
                     for r in runs.values())]
    common.sort(key=lambda k: (k[1], k[0]))
    cases = [ref["cases"][k] for k in common]
    threshold = ref["bt"].threshold
    out: dict = {"horizon_months": horizon, "threshold_pct": threshold, "n_common_cases": len(cases),
                 "n_dates": len({c.as_of for c in cases}), "n_locations": len({c.loc.key for c in cases}),
                 "label_mix": {l: sum(c.truth == l for c in cases) for l in M.LABELS},
                 "models": {}, "baselines": {}, "head_to_head": {}, "tuning": {}, "caveats": bt.CAVEATS,
                 "split": split.isoformat() if split else None}
    if not cases:
        out["verdict"] = ("No cases where every model produced real judgments. Check the model setup "
                          "(for example that Ollama is running) and the fallback counts below.")
    truth_labels = [c.truth for c in cases]
    fwd = [c.fwd_pct for c in cases]

    def score(directions, scores):
        cs, nd = M.mean_spearman_by_group([c.as_of for c in cases], scores, fwd)
        return {"accuracy": M.accuracy(directions, truth_labels),
                "balanced_accuracy": M.balanced_accuracy(directions, truth_labels),
                "spearman_per_date": cs, **M.direction_spread(directions, fwd)}

    correct: dict[str, list[bool]] = {}
    dirs: dict[str, list[str]] = {}
    for name, r in runs.items():
        ps = [r["preds"]["full"][k] for k in common]
        dirs[name] = [p[0] for p in ps]
        correct[name] = [d == t for d, t in zip(dirs[name], truth_labels)]
        row = score(dirs[name], [p[1] for p in ps]) if cases else {}
        total_cases = len(r["cases"])
        row.update({
            "n_scored": len(cases), "n_attempted": total_cases,
            "valid_rate": None if not total_cases else round(1 - len(r["fallback_keys"]) / total_cases, 3),
            "mean_confidence": round(sum(p[2] for p in ps) / len(ps), 3) if ps else None,
            "seconds": round(r["seconds"], 1), **{k: v for k, v in r["stats"].items()},
            "fallbacks": sum(r["bt"].fallbacks.values()),
        })
        if row.get("accuracy") is not None and row["mean_confidence"] is not None:
            row["overconfidence"] = round(row["mean_confidence"] - row["accuracy"], 3)
        calls = row.get("calls") or 0
        row["seconds_per_call"] = round(r["seconds"] / calls, 2) if calls else None
        out["models"][name] = row
    for name in runs:
        if name != RULES and cases:
            out["models"][name]["agreement_with_rules"] = round(
                sum(a == b for a, b in zip(dirs[name], dirs[RULES])) / len(cases), 3)
    for b in BASELINES:
        if not cases:
            break
        ps = [ref["preds"][b][k] for k in common]
        out["baselines"][b] = score([p[0] for p in ps], [p[1] for p in ps])
        correct[b] = [p[0] == t for p, t in zip(ps, truth_labels)]
        dirs[b] = [p[0] for p in ps]

    def diff(a: str, b: str) -> dict:
        return bt.paired_diff([(c.as_of, x, y) for c, x, y in zip(cases, correct[a], correct[b])], horizon)

    if cases:
        for name in runs:
            out["models"][name]["vs_plain_trend"] = diff(name, "base_trend")
            if name != RULES:
                out["models"][name]["vs_rules"] = diff(name, RULES)
        names = [n for n in runs if n != RULES]
        for a in names:
            for b in names:
                if a != b:
                    out["head_to_head"][f"{a} vs {b}"] = diff(a, b)
        out["verdict"] = _verdict(out)
    if tune and split:
        for name, r in runs.items():
            out["tuning"][name] = bt.tune_weights(store, r["bt"], r["names"], split)
    return out


def _fmt_ci(d: dict | None) -> str:
    if not d or "ci90" not in d:
        return "n/a"
    lo, hi = d["ci90"]
    return f"{d['mean_accuracy_diff'] * 100:+.1f} ({lo * 100:+.1f} to {hi * 100:+.1f})"


def _verdict(out: dict) -> str:
    models = {n: m for n, m in out["models"].items() if n != RULES and m.get("accuracy") is not None}
    if not models:
        return "Only the rules ran; add models to compare."
    best = max(models, key=lambda n: (models[n]["accuracy"], models[n]["balanced_accuracy"] or 0))
    m, rules = models[best], out["models"][RULES]
    vr = m.get("vs_rules", {})
    head = (f"Most accurate: {best} at {m['accuracy']:.1%} (rules {rules['accuracy']:.1%}) on "
            f"{out['n_common_cases']} common cases over {out['n_dates']} dates.")
    if "ci90" in vr:
        lo, hi = vr["ci90"]
        if lo > 0:
            return head + f" It beats the rules by {vr['mean_accuracy_diff'] * 100:.1f} points (90% CI {lo * 100:.1f} to {hi * 100:.1f})."
        if hi < 0:
            return head + " It is clearly WORSE than the rules."
        return head + " Its edge over the rules is within noise, so it is not shown to be better."
    return head + " Too few dates to say whether any difference is real."


# -- output ------------------------------------------------------------------------------------

COLUMNS = [
    ("Model", lambda n, m: n),
    ("Cases", lambda n, m: str(m["n_scored"])),
    ("Accuracy", lambda n, m: _pct(m.get("accuracy"))),
    ("Balanced acc.", lambda n, m: _pct(m.get("balanced_accuracy"))),
    ("Rank corr. / date", lambda n, m: _num(m.get("spearman_per_date"), 2)),
    ("Up-minus-down spread (pts)", lambda n, m: _num(m.get("spread"), 1)),
    ("vs rules (pts, 90% CI)", lambda n, m: "-" if n == RULES else _fmt_ci(m.get("vs_rules"))),
    ("vs plain trend (pts, 90% CI)", lambda n, m: _fmt_ci(m.get("vs_plain_trend"))),
    ("Agrees with rules", lambda n, m: _pct(m.get("agreement_with_rules"))),
    ("Mean confidence", lambda n, m: _num(m.get("mean_confidence"), 2)),
    ("Overconfidence", lambda n, m: _num(m.get("overconfidence"), 2, signed=True)),
    ("Valid judgments", lambda n, m: _pct(m.get("valid_rate"))),
    ("Model calls", lambda n, m: "-" if n == RULES else str(m.get("calls", 0))),
    ("Cache hits", lambda n, m: "-" if n == RULES else str(m.get("cache_hits", 0))),
    ("Parse failures", lambda n, m: "-" if n == RULES else str(m.get("parse_failures", 0))),
    ("Provider errors", lambda n, m: "-" if n == RULES else str(m.get("provider_errors", 0))),
    ("Fallbacks", lambda n, m: "-" if n == RULES else str(m.get("fallbacks", 0))),
    ("Minutes", lambda n, m: _num(m["seconds"] / 60, 1)),
    ("Sec / call", lambda n, m: "-" if n == RULES else _num(m.get("seconds_per_call"), 1)),
]


def _pct(x):
    return "-" if x is None else f"{x * 100:.1f}%"


def _num(x, nd, signed=False):
    if x is None:
        return "-"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def rows(out: dict) -> list[list[str]]:
    r = [[c for c, _ in COLUMNS]]
    for name, m in out["models"].items():
        r.append([f(name, m) for _, f in COLUMNS])
    for b, m in out["baselines"].items():
        r.append([BASELINE_LABELS[b], str(out["n_common_cases"]), _pct(m["accuracy"]), _pct(m["balanced_accuracy"]),
                  _num(m["spearman_per_date"], 2), _num(m["spread"], 1)] + ["-"] * (len(COLUMNS) - 6))
    return r


def to_csv(out: dict) -> str:
    buf = io.StringIO()
    csv.writer(buf).writerows(rows(out))
    return buf.getvalue()


def _md_table(table: list[list[str]]) -> str:
    head, body = table[0], table[1:]
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|",
                      *("| " + " | ".join(r) + " |" for r in body)])


def to_markdown(out: dict) -> str:
    L = [f"# Model comparison\n", out.get("verdict", ""), "",
         f"Horizon {out['horizon_months']} months, class threshold ±{out['threshold_pct']:.1f}%. "
         f"{out['n_common_cases']} common cases, {out['n_dates']} dates, {out['n_locations']} locations; "
         f"outcomes {out['label_mix']}.", "", _md_table(rows(out)), ""]
    if out["head_to_head"]:
        L += ["## Head to head (accuracy difference, points, 90% CI; row model minus column model)", ""]
        names = [n for n in out["models"] if n != RULES]
        table = [[""] + names]
        for a in names:
            table.append([a] + ["-" if a == b else _fmt_ci(out["head_to_head"].get(f"{a} vs {b}")) for b in names])
        L += [_md_table(table), ""]
    for name, t in out["tuning"].items():
        if "chosen_weights" in t:
            L += [f"## Weights tuned for {name}", "",
                  f"Chosen {t['chosen_weights']} (train balanced accuracy {t['train_balanced_accuracy']:.3f}); "
                  f"test {t['test_balanced_accuracy_chosen']:.3f} vs default {t['test_balanced_accuracy_default']:.3f}. "
                  f"{t['recommendation']}", ""]
    L += ["## How to read this", "",
          "- Scoring uses only cases where every model produced real judgments, so rows are comparable.",
          "- Rank corr. / date: how well the score orders cities by later price growth, averaged over dates.",
          "- Overconfidence: mean stated confidence minus accuracy; above zero means it claims more than it delivers.",
          "- Valid judgments: share of cases where no model call fell back to the rule. Low means a setup problem.",
          "", "## Caveats", "", *[f"- {c}" for c in out["caveats"]]]
    return "\n".join(L)


def format_text(out: dict) -> str:
    table = rows(out)
    widths = [max(len(r[i]) for r in table) for i in range(len(table[0]))]
    lines = [out.get("verdict", ""), "",
             f"{out['n_common_cases']} common cases, {out['n_dates']} dates, {out['n_locations']} locations; "
             f"outcomes {out['label_mix']}", ""]
    for r in table:
        lines.append("  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(r, widths))))
    for name, t in out["tuning"].items():
        if "chosen_weights" in t:
            lines.append(f"\nWeights for {name}: {t['chosen_weights']} -> {t['recommendation']}")
    return "\n".join(lines)
