"""Scoring functions. Plain code, no I/O."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import mean

LABELS = ("down", "flat", "up")


def label(pct: float, threshold: float) -> str:
    """Realized change -> class. ``threshold`` is in percent over the whole horizon."""
    return "up" if pct > threshold else "down" if pct < -threshold else "flat"


def threshold_for(horizon_months: int, annual_pct: float = 2.0) -> float:
    return annual_pct * horizon_months / 12.0


def accuracy(pred: list[str], truth: list[str]) -> float | None:
    return round(mean(p == t for p, t in zip(pred, truth)), 4) if pred else None


def balanced_accuracy(pred: list[str], truth: list[str]) -> float | None:
    """Mean recall over the classes that actually occur (so 'always up' cannot look good
    just because most outcomes were up)."""
    recalls = []
    for c in LABELS:
        idx = [i for i, t in enumerate(truth) if t == c]
        if idx:
            recalls.append(mean(pred[i] == c for i in idx))
    return round(mean(recalls), 4) if recalls else None


def confusion(pred: list[str], truth: list[str]) -> dict[str, dict[str, int]]:
    out = {t: {p: 0 for p in LABELS} for t in LABELS}
    for p, t in zip(pred, truth):
        out[t][p] += 1
    return out


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return round(num / den, 4) if den else None


def partial_spearman(xs: list[float], ys: list[float], zs: list[float]) -> float | None:
    """Rank correlation of x and y after removing what z explains in both."""
    rxy, rxz, ryz = spearman(xs, ys), spearman(xs, zs), spearman(ys, zs)
    if rxy is None or rxz is None or ryz is None:
        return None
    den = math.sqrt(max(0.0, (1 - rxz ** 2) * (1 - ryz ** 2)))
    return round((rxy - rxz * ryz) / den, 4) if den > 1e-9 else None


def mean_spearman_by_group(groups: list, xs: list[float], ys: list[float],
                           min_size: int = 8) -> tuple[float | None, int]:
    """Average cross-sectional Spearman: within each group (date), do higher scores go with
    higher realized changes? Isolates picking relative winners from calling the market."""
    by = defaultdict(lambda: ([], []))
    for g, x, y in zip(groups, xs, ys):
        by[g][0].append(x)
        by[g][1].append(y)
    vals = [v for g, (a, b) in by.items() if len(a) >= min_size
            and (v := spearman(a, b)) is not None]
    return (round(mean(vals), 4) if vals else None), len(vals)


def direction_spread(pred: list[str], fwd: list[float]) -> dict:
    """Average realized change when the call was up vs down."""
    up = [f for p, f in zip(pred, fwd) if p == "up"]
    down = [f for p, f in zip(pred, fwd) if p == "down"]
    return {
        "n_up": len(up), "n_down": len(down),
        "mean_fwd_when_up": round(mean(up), 2) if up else None,
        "mean_fwd_when_down": round(mean(down), 2) if down else None,
        "spread": round(mean(up) - mean(down), 2) if up and down else None,
    }


def calibration(conf: list[float], correct: list[bool], bins=(0.0, 0.3, 0.5, 0.7, 1.01)) -> list[dict]:
    out = []
    for lo, hi in zip(bins, bins[1:]):
        idx = [i for i, c in enumerate(conf) if lo <= c < hi]
        out.append({"confidence": f"{lo:.1f}-{min(hi, 1.0):.1f}", "n": len(idx),
                    "accuracy": round(mean(correct[i] for i in idx), 3) if idx else None,
                    "mean_confidence": round(mean(conf[i] for i in idx), 3) if idx else None})
    return out


def block_bootstrap_ci(values: list[float], block_len: int, *, n_boot: int = 2000,
                       level: float = 0.90, seed: int = 0) -> tuple[float, float, float]:
    """Mean and CI of a time-ordered series using a moving-block bootstrap, because
    neighbouring dates share forward windows and are not independent."""
    n = len(values)
    if n == 0:
        raise ValueError("no values")
    block_len = max(1, min(block_len, n))
    rng = random.Random(seed)
    starts = n - block_len + 1
    k = math.ceil(n / block_len)
    means = []
    for _ in range(n_boot):
        sample = []
        for _ in range(k):
            s = rng.randrange(starts)
            sample.extend(values[s:s + block_len])
        means.append(mean(sample[:n]))
    means.sort()
    lo = means[int((1 - level) / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 + level) / 2 * n_boot))]
    return round(mean(values), 4), round(lo, 4), round(hi, 4)
