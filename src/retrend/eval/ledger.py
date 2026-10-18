"""Forward-tracking ledger: record today's outlooks, score them when the future arrives.

This is the clean test: nothing was known at prediction time, a model cannot have
memorized the outcome, and the stored snapshot is exactly what the system said.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

from retrend.eval import metrics as M
from retrend.eval.truth import ZhviTruth
from retrend.location import Location
from retrend.outlook import Outlook
from retrend.store import RawStore

SOURCE = "ledger"


class Ledger:
    def __init__(self, store: RawStore) -> None:
        self.store = store

    def record(self, outlook: Outlook, model: str, truth: ZhviTruth, *,
               horizon_months: int = 12) -> bool:
        """Store the outlook plus the plain-trend baseline's call at the same moment.
        Returns False if there is no ZHVI origin to anchor the later scoring."""
        today = self.store.clock().date()
        origin = truth.origin(outlook.location, today)
        if origin is None:
            return False
        trailing = truth.trailing_pct(outlook.location, origin)
        payload = {
            "location_key": outlook.location.key, "model": model, "horizon_months": horizon_months, "recorded": today.isoformat(),
            "origin": origin.day.isoformat(), "origin_value": origin.value,
            "baseline_trend_pct": trailing,
            "baseline_trend_direction": M.label(trailing, M.threshold_for(12)) if trailing is not None else None,
            "outlook": json.loads(outlook.model_dump_json()),
        }
        self.store.put(SOURCE, f"{outlook.location.key}|{model}|{today}|h{horizon_months}",
                       payload, data_as_of=today)
        return True

    def entries(self) -> list[dict]:
        out = []
        for key in self.store.keys(SOURCE):
            rec = self.store.get_latest(SOURCE, key)
            if rec:
                out.append(rec.json())
        return out

    def score(self, locations: dict[str, Location], truth: ZhviTruth) -> dict:
        rows, pending = [], 0
        for e in self.entries():
            loc = locations.get(e["location_key"])
            if loc is None:
                pending += 1
                continue
            origin = next((p for p in truth.series(loc) if p.day.isoformat() == e["origin"]), None)
            fwd = truth.forward_pct(loc, origin, e["horizon_months"]) if origin else None
            if fwd is None:
                pending += 1
                continue
            lab = M.label(fwd, M.threshold_for(e["horizon_months"]))
            rows.append({"location": loc.key, "model": e["model"], "recorded": e["recorded"],
                         "predicted": e["outlook"]["direction"], "confidence": e["outlook"]["confidence"],
                         "baseline": e["baseline_trend_direction"], "realized_pct": fwd, "actual": lab})
        res: dict = {"scored": len(rows), "pending": pending, "rows": rows}
        if rows:
            by_model: dict[str, list[dict]] = {}
            for r in rows:
                by_model.setdefault(r["model"], []).append(r)
            res["by_model"] = {
                m: {"n": len(rs),
                    "system_accuracy": M.accuracy([r["predicted"] for r in rs], [r["actual"] for r in rs]),
                    "baseline_accuracy": M.accuracy([r["baseline"] for r in rs if r["baseline"]],
                                                    [r["actual"] for r in rs if r["baseline"]])}
                for m, rs in by_model.items()}
        return res
