"""Run a backtest's model calls in parallel, then replay it from the cache.

Pass 1 walks the whole backtest with the judge in "collect" mode: every question the model
would be asked is noted, and the rule fallback answers in its place (no model time). Then the
distinct, uncached questions are asked ``parallel`` at a time. The normal run that follows finds
every answer in the cache, so its results are exactly those of a one-at-a-time run.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from retrend.llm import BREAKER_FAILURES


def plan(judge, run_pass: Callable[[], object]) -> list:
    """Every model question ``run_pass`` would ask (the rules answer in their place)."""
    judge.collect = []
    try:
        run_pass()
        return judge.collect
    finally:
        judge.collect = None


def warm_cache(judge, run_pass: Callable[[], object], *, parallel: int,
               progress: Callable[[str], None] | None = None) -> dict | None:
    parallel = judge.parallel or parallel if judge is not None else parallel
    if judge is None or parallel <= 1:
        return None
    requests = plan(judge, run_pass)
    if progress:
        progress(f"{len(requests)} model questions planned; asking {parallel} at a time")
    return judge.prefetch(requests, workers=parallel, progress=progress)


def warm_many(items: list, *, parallel: int, progress: Callable[[str], None] | None = None) -> dict:
    """items = [(name, judge, run_pass)]. All models are asked at the same time, each with its own
    limit (``judge.parallel`` from models.yaml, else ``parallel``). Results are saved from this
    thread only."""
    jobs = []
    for name, judge, run_pass in items:
        if judge is None:
            continue
        todo = judge.uncached(plan(judge, run_pass))
        jobs.append((name, judge, judge.parallel or parallel, todo))
        if progress:
            progress(f"[{name}] {len(todo)} model questions to ask, {judge.parallel or parallel} at a time")
    pools, futures, left, out = [], {}, {}, {}
    try:
        for name, judge, n, todo in jobs:
            out[name] = {"requested": len(todo), "done": 0, "failed": 0}
            left[name] = len(todo)
            pool = ThreadPoolExecutor(max_workers=max(1, n))
            pools.append(pool)
            for key, args in todo.items():
                futures[pool.submit(judge._attempt_loop, *args)] = (name, judge, key)
        total = len(futures)
        for i, fut in enumerate(as_completed(futures), 1):
            name, judge, key = futures[fut]
            try:
                data = fut.result()
            except Exception as exc:
                out[name]["failed"] += 1
                if out[name]["done"] == 0 and out[name]["failed"] == BREAKER_FAILURES:
                    for f, (n2, _j, _k) in futures.items():  # stop this model only; the others carry on
                        if n2 == name:
                            f.cancel()
                    if progress:
                        progress(f"[{name}] STOPPED: the first {BREAKER_FAILURES} questions all failed. "
                                 f"Last error: {str(exc)[:300]}")
            else:
                out[name]["done"] += 1
                if judge.store is not None:
                    judge.store.put_llm_response(*key, json.dumps(data))
            if progress and i % 25 == 0:
                progress(f"asked {i}/{total} model questions across {len(jobs)} models: "
                         + ", ".join(f"{n} {o['done']} answered/{o['failed']} failed" for n, o in out.items()))
    finally:
        for pool in pools:
            pool.shutdown(wait=False, cancel_futures=True)
    return out
