"""Does school enrollment tell us anything about later home-price growth?

At each date, for every city that has a school-district match, take the enrollment change that
was public at that date, and the realized price change over the next ``horizon`` months. Then ask:

1. Raw: do cities with growing enrollment later see more price growth than the others?
   (average rank correlation across dates)
2. Controlled: same question after removing what recent price growth already explains
   (partial rank correlation). Price momentum may carry the same information, so this is the
   test that matters.
3. Direction: average later price change when the schools rule said up vs down.
"""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from statistics import mean
from typing import Callable

from retrend.eval import metrics as M
from retrend.location import Location

LIMITS = [
    "Few cities and overlapping 12-month windows mean a small effective sample: only a large effect "
    "can show up. 'No detectable relationship' is not 'no relationship'.",
    "School data is the current, revised file; each school year is treated as public from June 30 of the "
    "following year, which is an assumption.",
    "A district stands in for the city; district boundaries often differ from city limits.",
    "A relationship here would show correlation, not that enrollment drives prices (births, migration and "
    "school choice move both).",
]


def _end_of_day(d: date) -> datetime:
    return datetime.combine(d, time(23, 59, 59), tzinfo=timezone.utc)


def _series_ci(values: list[float], dates: list[date], horizon: int) -> dict:
    if len(values) < 6:
        return {"n_dates": len(values), "note": "too few dates for an interval"}
    gap = max(1, round(mean((b.year - a.year) * 12 + b.month - a.month for a, b in zip(dates, dates[1:]))))
    m, lo, hi = M.block_bootstrap_ci(values, max(1, round(horizon / gap)))
    return {"mean": m, "ci90": [lo, hi], "n_dates": len(values)}


def run_schools_test(schools_worker, truth, locations: list[Location], dates: list[date], *,
                     horizon: int = 12, min_cities: int = 8,
                     progress: Callable[[str], None] | None = None) -> dict:
    threshold = M.threshold_for(horizon)
    rows = []  # (date, loc, enrollment_yoy, trailing, fwd, schools_direction)
    unmatched: dict[str, str] = {}
    for loc in locations:
        try:
            truth.series(loc)
        except Exception as exc:
            unmatched[loc.key] = f"no price series ({type(exc).__name__})"
            continue
        n_before = len(rows)
        for d in dates:
            origin = truth.origin(loc, d)
            if origin is None:
                continue
            fwd, trailing = truth.forward_pct(loc, origin, horizon), truth.trailing_pct(loc, origin)
            if fwd is None or trailing is None:
                continue
            try:
                rep = schools_worker.run(loc, as_of=_end_of_day(d))
            except Exception as exc:
                if loc.key not in unmatched and len(rows) == n_before:
                    unmatched[loc.key] = f"{type(exc).__name__}: {str(exc)[:80]}"
                continue
            yoy = rep.key_metrics.get("enrollment_yoy_pct")
            if yoy is None:
                continue
            rows.append((d, loc, yoy, trailing, fwd, rep.direction))
        if len(rows) == n_before:
            unmatched.setdefault(loc.key, "no usable dates")
        if progress:
            progress(f"done {loc.key} ({len(rows)} rows so far)")

    matched = {r[1].key for r in rows}
    out: dict = {
        "horizon_months": horizon, "n_rows": len(rows), "n_cities_matched": len(matched),
        "n_cities_total": len(locations), "unmatched": unmatched, "limits": LIMITS,
    }
    by_date: dict[date, list] = {}
    for r in rows:
        by_date.setdefault(r[0], []).append(r)
    raw, ctl, used = [], [], []
    for d in sorted(by_date):
        g = by_date[d]
        if len(g) < min_cities:
            continue
        e, t, f = [x[2] for x in g], [x[3] for x in g], [x[4] for x in g]
        a, b = M.spearman(e, f), M.partial_spearman(e, f, t)
        if a is None or b is None:
            continue
        raw.append(a)
        ctl.append(b)
        used.append(d)
    out["n_dates"] = len(used)
    out["raw"] = _series_ci(raw, used, horizon)
    out["controlled"] = _series_ci(ctl, used, horizon)
    out["direction_spread"] = M.direction_spread([r[5] for r in rows], [r[4] for r in rows])
    out["verdict"] = _verdict(out)
    return out


def _verdict(o: dict) -> str:
    if o["n_cities_matched"] < 8 or o["n_dates"] < 6:
        return (f"Not enough data for a verdict: {o['n_cities_matched']} cities matched a school district "
                f"and {o['n_dates']} dates had at least 8 cities. See 'unmatched' for why cities dropped out.")
    raw, ctl = o["raw"], o["controlled"]
    if "ci90" not in raw:
        return "Too few dates for an interval."

    def read(r):
        lo, hi = r["ci90"]
        return "positive" if lo > 0 else "negative" if hi < 0 else "none"
    a, b = read(raw), read(ctl)
    head = (f"Average rank correlation between enrollment growth and the next {o['horizon_months']} months of "
            f"price growth: {raw['mean']:+.2f} (90% interval {raw['ci90'][0]:+.2f} to {raw['ci90'][1]:+.2f}); "
            f"after removing recent price momentum: {ctl['mean']:+.2f} "
            f"({ctl['ci90'][0]:+.2f} to {ctl['ci90'][1]:+.2f}).")
    if b == "positive":
        tail = " Enrollment adds information beyond price momentum. It is a candidate for a small voting weight; confirm with the backtest variants (`backtest --with-schools`)."
    elif a == "positive":
        tail = " Enrollment growth goes with later price growth, but the effect disappears once momentum is accounted for, so it adds nothing beyond the price trend. Keep it as context."
    elif a == "negative" or b == "negative":
        tail = " The relationship is negative, which is unexpected. Check the district matches and data before drawing any conclusion."
    else:
        tail = " No detectable relationship. Keep schools as context only (weight 0)."
    return head + tail


def format_schools_test(o: dict) -> str:
    L = [f"School enrollment test: horizon {o['horizon_months']} months",
         f"{o['n_cities_matched']} of {o['n_cities_total']} cities matched a district; {o['n_rows']} city-date rows over {o['n_dates']} usable dates"]
    if o["unmatched"]:
        L.append("Dropped: " + "; ".join(f"{k} ({v})" for k, v in list(o["unmatched"].items())[:10]))
    ds = o["direction_spread"]
    L.append(f"When the schools rule said up ({ds['n_up']} rows) later price change averaged {ds['mean_fwd_when_up']}%; "
             f"down ({ds['n_down']} rows) {ds['mean_fwd_when_down']}%; spread {ds['spread']}")
    L.append("")
    L.append(o["verdict"])
    L.append("\nLimits:")
    L.extend(f"  - {c}" for c in o["limits"])
    return "\n".join(L)
