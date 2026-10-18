"""Phase 6: compare locations, track how outlooks change between runs, render a report.

Pure functions plus a small snapshot store; nothing here calls a model or the network.
All text that came from a model or a data file is HTML-escaped before it reaches the page.
"""

from __future__ import annotations

import csv
import html
import io
import json
from dataclasses import dataclass, field
from datetime import datetime

from retrend.outlook import Outlook
from retrend.store import RawStore

SNAPSHOT_SOURCE = "outlook_run"
SCORE_MOVE = 0.15       # a score change at least this big is flagged
CONFIDENCE_MOVE = 0.15  # likewise for confidence
EVIDENCE_NOTE = (
    "Status of the evidence: in the first backtest this system did not show a measurable "
    "advantage over a plain trailing-trend rule, confidence values are uncalibrated, and the "
    "forward-tracking test is still maturing. Treat this as a structured summary of public "
    "data signals, not a forecast you can rely on, and not financial advice."
)


@dataclass
class Change:
    kind: str       # direction_change | score_move | confidence_move | worker_lost | worker_back | new
    severity: str   # high | medium | info
    detail: str


@dataclass
class Row:
    key: str
    city: str
    state: str
    outlook: Outlook | None
    previous: dict | None = None      # previous snapshot payload, if any
    changes: list[Change] = field(default_factory=list)
    error: str | None = None


# -- snapshots --------------------------------------------------------------------------


def save_snapshot(store: RawStore, outlook: Outlook, model: str, run_id: str) -> None:
    payload = {"run_id": run_id, "model": model,
               "outlook": json.loads(outlook.model_dump_json())}
    store.put(SNAPSHOT_SOURCE, f"{outlook.location.key}|{model}", payload,
              data_as_of=store.clock().date())


def previous_snapshot(store: RawStore, loc_key: str, model: str, run_id: str) -> dict | None:
    """Most recent saved run for this location and model from a different run (month)."""
    for rec in reversed(store.history(SNAPSHOT_SOURCE, f"{loc_key}|{model}")):
        payload = rec.json()
        if payload.get("run_id") != run_id:
            return payload
    return None


# -- change detection -------------------------------------------------------------------


def diff(outlook: Outlook, previous: dict | None) -> list[Change]:
    if previous is None:
        return [Change("new", "info", "first recorded run for this location")]
    old = previous["outlook"]
    out: list[Change] = []
    if old["direction"] != outlook.direction:
        out.append(Change("direction_change", "high",
                          f"{old['direction']} -> {outlook.direction} "
                          f"(score {old['score']:+.2f} -> {outlook.score:+.2f})"))
    elif abs(outlook.score - old["score"]) >= SCORE_MOVE:
        out.append(Change("score_move", "medium",
                          f"score {old['score']:+.2f} -> {outlook.score:+.2f}, direction still {outlook.direction}"))
    if abs(outlook.confidence - old["confidence"]) >= CONFIDENCE_MOVE:
        out.append(Change("confidence_move", "medium",
                          f"confidence {old['confidence']:.2f} -> {outlook.confidence:.2f}"))
    old_missing, new_missing = set(old.get("missing_workers", {})), set(outlook.missing_workers)
    for w in sorted(new_missing - old_missing):
        out.append(Change("worker_lost", "medium", f"{w} unavailable this run ({outlook.missing_workers[w][:80]})"))
    for w in sorted(old_missing - new_missing):
        out.append(Change("worker_back", "info", f"{w} available again"))
    return out


# -- ranking and summary ----------------------------------------------------------------


def rank(rows: list[Row]) -> list[Row]:
    ok = [r for r in rows if r.outlook is not None]
    ok.sort(key=lambda r: (-r.outlook.score, -r.outlook.confidence, r.key))
    return ok + sorted((r for r in rows if r.outlook is None), key=lambda r: r.key)


def summarize(rows: list[Row]) -> dict:
    ok = [r for r in rows if r.outlook is not None]
    counts = {d: sum(r.outlook.direction == d for r in ok) for d in ("up", "flat", "down")}
    return {
        "n_locations": len(rows), "n_failed": len(rows) - len(ok), "counts": counts,
        "direction_changes": [r.key for r in ok if any(c.kind == "direction_change" for c in r.changes)],
        "other_flags": [r.key for r in ok if any(c.severity == "medium" for c in r.changes)
                        and not any(c.kind == "direction_change" for c in r.changes)],
        "oldest_data": min((c.data_as_of for r in ok for c in r.outlook.drivers + r.outlook.dissent + r.outlook.context),
                           default=None),
    }


def _drivers_text(o: Outlook) -> str:
    return ", ".join(f"{c.worker} {c.direction} ({c.contribution:+.2f})" for c in o.drivers) or "none"


# -- renderers --------------------------------------------------------------------------


def to_csv(rows: list[Row]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["rank", "location", "direction", "score", "confidence", "previous_direction",
                "previous_score", "flags", "missing_signals", "drivers"])
    for i, r in enumerate(rank(rows), 1):
        o = r.outlook
        if o is None:
            w.writerow([i, r.key, "ERROR", "", "", "", "", r.error or "", "", ""])
            continue
        prev = r.previous["outlook"] if r.previous else {}
        w.writerow([i, r.key, o.direction, o.score, o.confidence, prev.get("direction", ""),
                    prev.get("score", ""), "; ".join(f"{c.kind}: {c.detail}" for c in r.changes if c.kind != "new"),
                    ", ".join(o.missing_workers), _drivers_text(o)])
    return buf.getvalue()


def to_json(rows: list[Row], run_id: str, model: str, generated: datetime) -> str:
    return json.dumps({
        "run_id": run_id, "model": model, "generated_at": generated.isoformat(),
        "summary": summarize(rows), "evidence_note": EVIDENCE_NOTE,
        "locations": [{
            "key": r.key, "error": r.error,
            "changes": [c.__dict__ for c in r.changes],
            "outlook": json.loads(r.outlook.model_dump_json()) if r.outlook else None,
        } for r in rank(rows)]}, indent=2, default=str)


def to_markdown(rows: list[Row], run_id: str, model: str) -> str:
    s = summarize(rows)
    L = [f"# Housing outlook comparison, run {run_id} (model: {model})", "",
         f"{s['n_locations']} locations: {s['counts']['up']} up, {s['counts']['flat']} flat, "
         f"{s['counts']['down']} down" + (f", {s['n_failed']} failed" if s["n_failed"] else "") + ".", ""]
    if s["direction_changes"]:
        L.append(f"**Direction changed since last run:** {', '.join(s['direction_changes'])}")
    if s["other_flags"]:
        L.append(f"**Other notable moves:** {', '.join(s['other_flags'])}")
    L += ["", "| # | Location | Outlook | Score | Conf. | Since last run |", "|---|---|---|---|---|---|"]
    for i, r in enumerate(rank(rows), 1):
        o = r.outlook
        if o is None:
            L.append(f"| {i} | {r.key} | ERROR | | | {r.error} |")
            continue
        flags = "; ".join(c.detail for c in r.changes) or "no change"
        L.append(f"| {i} | {r.city}, {r.state} | {o.direction} | {o.score:+.2f} | {o.confidence:.2f} | {flags} |")
    L += ["", EVIDENCE_NOTE]
    return "\n".join(L)


_CSS = """
:root{--bg:#fff;--fg:#1c1e21;--muted:#5f6670;--line:#dfe3e8;--up:#1a7f4b;--down:#b3261e;--flat:#6b7280;--flag:#fff4d6}
@media (prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e8eaed;--muted:#a0a7b1;--line:#2b3037;--up:#4cc38a;--down:#ff8a80;--flat:#9aa3af;--flag:#3a3220}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px}
main{max-width:980px;margin:0 auto}h1{font-size:1.4rem;margin:0 0 4px}.muted{color:var(--muted)}
table{border-collapse:collapse;width:100%;margin:16px 0}th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}
.up{color:var(--up);font-weight:600}.down{color:var(--down);font-weight:600}.flat{color:var(--flat);font-weight:600}
.flag{background:var(--flag);padding:1px 6px;border-radius:4px}details{margin:6px 0}summary{cursor:pointer}
.note{border-left:3px solid var(--line);padding:4px 12px;margin:20px 0;color:var(--muted)}
@media (max-width:640px){th:nth-child(5),td:nth-child(5){display:none}}
"""


def to_html(rows: list[Row], run_id: str, model: str, generated: datetime) -> str:
    e = html.escape
    s = summarize(rows)
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>Housing outlook comparison</title><style>{_CSS}</style></head><body><main>",
             f"<h1>Housing outlook comparison</h1><p class='muted'>Run {e(run_id)} &middot; model {e(model)} "
             f"&middot; generated {e(generated.strftime('%Y-%m-%d %H:%M UTC'))}</p>",
             f"<p>{s['n_locations']} locations: <span class='up'>{s['counts']['up']} up</span>, "
             f"<span class='flat'>{s['counts']['flat']} flat</span>, "
             f"<span class='down'>{s['counts']['down']} down</span>"
             + (f", {s['n_failed']} failed" if s["n_failed"] else "") + ".</p>"]
    if s["direction_changes"]:
        parts.append(f"<p><span class='flag'>Direction changed since last run:</span> {e(', '.join(s['direction_changes']))}</p>")
    if s["other_flags"]:
        parts.append(f"<p><span class='flag'>Other notable moves:</span> {e(', '.join(s['other_flags']))}</p>")
    parts.append("<table><thead><tr><th>#</th><th>Location</th><th>Outlook</th><th>Score</th>"
                 "<th>Confidence</th><th>Since last run</th></tr></thead><tbody>")
    for i, r in enumerate(rank(rows), 1):
        o = r.outlook
        if o is None:
            parts.append(f"<tr><td>{i}</td><td>{e(r.city)}, {e(r.state)}</td><td colspan='4'>Failed: {e(r.error or '')}</td></tr>")
            continue
        flags = "".join(f"<div><span class='flag'>{e(c.detail)}</span></div>" if c.severity != "info"
                        else f"<div class='muted'>{e(c.detail)}</div>" for c in r.changes) or "<span class='muted'>no change</span>"
        parts.append(f"<tr><td>{i}</td><td><details><summary>{e(r.city)}, {e(r.state)}</summary>"
                     f"<p>{e(o.narrative)}</p><p class='muted'>{e(o.uncertainty)}</p>"
                     + "".join(f"<div>Driver: {e(c.worker)} {e(c.direction)} (share {c.contribution:+.2f}, data as of {c.data_as_of})</div>" for c in o.drivers)
                     + "".join(f"<div>Dissent: {e(c.worker)} {e(c.direction)} (share {c.contribution:+.2f})</div>" for c in o.dissent)
                     + "".join(f"<div class='muted'>Context, not voting: {e(c.worker)} {e(c.direction)}</div>" for c in o.context)
                     + "".join(f"<div class='muted'>Missing: {e(w)} ({e(why[:120])})</div>" for w, why in o.missing_workers.items())
                     + "".join(f"<div class='muted'>[{e(f.kind)}] {e(f.worker or '')} {e(f.detail)}</div>" for f in o.findings if f.kind != 'retried')
                     + f"</details></td><td class='{o.direction}'>{e(o.direction)}</td><td>{o.score:+.2f}</td>"
                       f"<td>{o.confidence:.2f}</td><td>{flags}</td></tr>")
    parts.append(f"</tbody></table><div class='note'>{e(EVIDENCE_NOTE)}</div></main></body></html>")
    return "".join(parts)
