"""Compose the final answer from the workers' results.

The model (when selected) writes prose from a structured fact sheet; code then checks the
prose: every number must appear in the facts or the question, and it may not give buy/sell
advice. If the check fails, or no model is selected, a deterministic template is used and the
reason is reported.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, ConfigDict, Field

from retrend.llm import JsonJudge
from retrend.outlook import Outlook
from retrend.ui.planner import WORKER_LABELS, Plan
from retrend.workers.housing_metrics import PROMPT_METRIC_KEYS as HOUSING_KEYS
from retrend.workers.mortgage_metrics import PROMPT_METRIC_KEYS as MORTGAGE_KEYS
from retrend.workers.supply_metrics import PROMPT_METRIC_KEYS as SUPPLY_KEYS
from retrend.workers.schools_metrics import PROMPT_METRIC_KEYS as SCHOOLS_KEYS
from retrend.workers.jobs_metrics import PROMPT_METRIC_KEYS as JOBS_KEYS
from retrend.workers.permits_metrics import PROMPT_METRIC_KEYS as PERMITS_KEYS
from retrend.workers.news_metrics import PROMPT_METRIC_KEYS as NEWS_KEYS

METRIC_KEYS = {"housing_prices": HOUSING_KEYS, "supply": SUPPLY_KEYS, "mortgage_rates": MORTGAGE_KEYS, "schools": SCHOOLS_KEYS,
               "jobs": JOBS_KEYS, "permits": PERMITS_KEYS, "news": NEWS_KEYS}
ADVICE = re.compile(r"\b(you should (buy|sell|wait|invest)|i recommend (buying|selling)|good time to (buy|sell)|"
                    r"now is the time|guaranteed|will definitely)\b", re.I)
NUM = re.compile(r"\d+(?:\.\d+)?")

SYSTEM = """\
You write the answer a person reads in a housing-market tool. You get their QUESTION and FACTS \
(JSON) from specialist signals. Write like a knowledgeable friend explaining over coffee: plain, \
warm, direct, but thorough.

Structure, as plain paragraphs separated by blank lines (no bullets, no headings):
1. One sentence answering the question directly.
2. One short paragraph per signal, in this order: home prices and rents, supply and demand, \
mortgage rates, school enrollment, local jobs, building permits, news. Say what each one shows, \
using the numbers in its metrics, and whether it points up, down or sideways for prices. A \
signal listed under "context_only" informs the picture but does not vote; say so in a few words. \
If a signal is in "missing", say it was unavailable and why, in half a sentence.
3. A final paragraph beginning "Overall trend:" that weighs the signals against each other, names \
what supports the direction and what pulls the other way, and states the direction \
(up, down or flat) for the next 6 to 12 months.
4. One short closing sentence that this is a read on direction, not a prediction, and the \
confidence figure is uncalibrated. If "not_covered" is listed, add that this tool does not look at those.

For several places, cover each place in turn (or rank them if the question compares) before the \
overall paragraph.

Rules:
- Describe signals in everyday words, never as "workers", "drivers", "scores" or field names.
- Every number you write must appear in the facts. Round nothing yourself.
- Never give buy, sell or invest advice.

Reply with ONLY a JSON object: {"answer": "..."}
"""


class AnswerDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")
    answer: str = Field(min_length=1, max_length=6000)


def build_facts(plan: Plan, outlooks: list[Outlook], metrics: dict[str, dict[str, dict]],
                errors: dict[str, str]) -> dict:
    places = []
    for o in outlooks:
        key = o.location.key
        m = {}
        for w, vals in metrics.get(key, {}).items():
            keys = METRIC_KEYS.get(w, [])
            m[w] = {k: vals[k] for k in keys if vals.get(k) is not None}
        places.append({
            "place": f"{o.location.city}, {o.location.state}",
            "direction": o.direction, "score": round(o.score, 2), "confidence": round(o.confidence, 2),
            "uncertainty": o.uncertainty,
            "drivers": [{"signal": c.worker, "direction": c.direction, "summary": c.summary} for c in o.drivers],
            "dissent": [{"signal": c.worker, "direction": c.direction, "summary": c.summary} for c in o.dissent],
            "context_only": [{"signal": c.worker, "direction": c.direction, "summary": c.summary} for c in o.context],
            "missing": o.missing_workers, "metrics": m,
        })
    return {"signals_used": [WORKER_LABELS[w] for w in plan.workers], "places": places,
            "not_covered": plan.unsupported, "errors": errors}


def _tokens(text: str) -> set[float]:
    return {float(x) for x in NUM.findall(text.replace(",", ""))}


def check_grounded(answer: str, facts: dict, query: str) -> str | None:
    """None if fine, else the reason it was rejected."""
    if ADVICE.search(answer):
        return "it gave buy/sell advice"
    allowed = _tokens(json.dumps(facts)) | _tokens(query) | {6.0, 12.0}
    allowed |= {abs(x) for x in allowed}
    stray = sorted(t for t in _tokens(answer) if t not in allowed
                   and not any(abs(t - a) < 0.051 for a in allowed))
    if stray:
        return f"it used numbers not in the data: {stray[:4]}"
    return None


def _label(worker: str) -> str:
    return WORKER_LABELS.get(worker, worker).split(" (")[0]


def _sentence(text: str) -> str:
    text = text.strip()
    return text if not text or text[-1] in ".!?" else text + "."


def _place_text(o: Outlook) -> str:
    """Every signal in turn, then the overall trend."""
    by_worker = {c.worker: (c, kind) for kind, group in (("votes", o.drivers), ("votes", o.dissent), ("context", o.context))
                 for c in group}
    name = f"{o.location.city}, {o.location.state}"
    paras = []
    for w in WORKER_LABELS:
        if w in by_worker:
            c, kind = by_worker[w]
            tail = " This signal is context only and does not vote." if kind == "context" else ""
            paras.append(f"{_label(w)} ({c.direction}): {_sentence(c.summary)}{tail}")
        elif w in o.missing_workers:
            paras.append(f"{_label(w)}: not available ({o.missing_workers[w]}).")
    overall = _plain_text(o.narrative)
    paras.append(f"Overall trend for {name}: {o.direction} over the next 6 to 12 months. {overall}")
    return "\n\n".join(paras)


def _plain_text(text: str) -> str:
    for w, label in WORKER_LABELS.items():
        text = text.replace(w, label.split(" (")[0].lower())
    return text.strip()


def template_answer(plan: Plan, outlooks: list[Outlook]) -> str:
    parts = []
    if plan.intent == "compare" and len(outlooks) > 1:
        ranked = sorted(outlooks, key=lambda o: -o.score)
        parts.append("Ranked by the combined price signal, highest first: " + "; ".join(
            f"{o.location.city}, {o.location.state} ({o.direction}, score {o.score:+.2f})" for o in ranked) + ".")
        shown = ranked[:3]
    else:
        shown = outlooks
    for o in shown:
        parts.append((f"{o.location.city}, {o.location.state}\n\n" if len(shown) > 1 else "") + _place_text(o))
    if plan.unsupported:
        parts.append("This tool does not cover " + ", ".join(plan.unsupported) + "; that part of your question is not answered.")
    parts.append("These are signals about direction, not predictions, and the confidence figure is uncalibrated.")
    return "\n\n".join(p.strip() for p in parts if p)


def compose(query: str, plan: Plan, outlooks: list[Outlook], metrics: dict, errors: dict,
            judge: JsonJudge | None) -> tuple[str, str, list[str]]:
    """-> (answer text, written_by, notes)"""
    notes: list[str] = []
    if not outlooks:
        return ("No signals could be computed for this question. " +
                ("; ".join(f"{k}: {v}" for k, v in errors.items()) or ""), "template", notes)
    facts = build_facts(plan, outlooks, metrics, errors)
    if judge is not None:
        try:
            res = judge.judge(SYSTEM, f"QUESTION: {query}\n\nFACTS:\n{json.dumps(facts, indent=1)}", AnswerDraft)
            text = res.value.answer.strip()  # type: ignore[attr-defined]
            problem = check_grounded(text, facts, query)
            if problem is None:
                if plan.unsupported and "cover" not in text.lower():
                    text += " This tool does not cover " + ", ".join(plan.unsupported) + "."
                return text, res.model, notes
            notes.append(f"The model's answer was rejected because {problem}; a template answer is shown.")
        except Exception as exc:
            notes.append(f"The model could not write the answer ({type(exc).__name__}); a template answer is shown.")
    return template_answer(plan, outlooks), "template", notes


def school_card(loc, report) -> dict:
    """A plain data view of one district's enrollment, for the schools-only lookup."""
    hist = json.loads(report.key_metrics["enrollment_history_json"])
    rows = []
    for i, r in enumerate(hist):
        prev = hist[i - 1]["students"] if i and hist[i - 1]["year"] == r["year"] - 1 else None
        rows.append({"year": r["year"], "students": round(r["students"]),
                     "change_pct": round((r["students"] / prev - 1) * 100, 1) if prev else None})
    km = report.key_metrics
    return {"place": f"{loc.city}, {loc.state}", "city": loc.city, "state": loc.state, "district": km["district"], "history": rows,
            "latest_year": km["latest_year"], "latest_students": km["latest_students"],
            "yoy_pct": km.get("enrollment_yoy_pct"), "cagr3_pct": km.get("enrollment_3y_cagr_pct"),
            "change_5y_pct": km.get("enrollment_5y_change_pct"), "direction": report.direction,
            "summary": report.summary, "caveats": report.caveats[:3]}


def school_answer(cards: list[dict], errors: dict[str, str]) -> str:
    parts = []
    for c in cards:
        bits = [f"{c['district']} had {c['latest_students']:,} students in fall {c['latest_year']}"]
        if c["yoy_pct"] is not None:
            bits.append(f"{c['yoy_pct']:+.1f}% from the year before")
        s = bits[0] + (", " + bits[1] if len(bits) > 1 else "") + "."
        if c["cagr3_pct"] is not None:
            s += f" Over the last three years it changed {c['cagr3_pct']:+.1f}% a year on average."
        if c["change_5y_pct"] is not None:
            s += f" Over five years: {c['change_5y_pct']:+.1f}%."
        parts.append(s)
    for k, v in errors.items():
        parts.append(f"No school data for {k.split('@')[-1]}: {v}")
    parts.append("This is enrollment only: it says nothing about school quality, and it is a weak, indirect signal for home prices.")
    return " ".join(parts)
