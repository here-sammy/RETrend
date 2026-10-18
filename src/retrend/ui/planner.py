"""Turn a plain-language question into a plan: which places, which workers, what is unavailable.

A model (when one is selected) proposes the plan; code then VALIDATES it, so a model can never
invent a city that is not in the question, call a worker that does not exist, or hide that part
of the question is out of scope. Without a model, or if the model fails, a rule-based parser
makes the plan and says so.
"""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from retrend.llm import JsonJudge
from retrend.location import Location
from retrend.sources.matching import norm

STATE_NAMES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA",
    "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD", "massachusetts": "MA",
    "michigan": "MI", "minnesota": "MN", "mississippi": "MS", "missouri": "MO", "montana": "MT",
    "nebraska": "NE", "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
    "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV", "wisconsin": "WI",
    "wyoming": "WY", "district of columbia": "DC",
}
STATE_CODES = set(STATE_NAMES.values())

WORKER_LABELS = {
    "housing_prices": "Home prices and rents (Zillow)",
    "supply": "Supply and demand (Redfin)",
    "mortgage_rates": "Mortgage rates (Freddie Mac via FRED)",
    "schools": "School enrollment trends (district data)",
    "jobs": "Local jobs (metro unemployment and payrolls, BLS via FRED)",
    "permits": "Building permits (Census via FRED)",
    "news": "News tone and coverage (GDELT)",
}
ALL_WORKERS = ["housing_prices", "supply", "mortgage_rates"]  # used for general questions
EXTRA_WORKERS = ["schools", "jobs", "permits", "news"]  # each only runs when the question asks about it
KNOWN_WORKERS = ALL_WORKERS + EXTRA_WORKERS
ALL_REASON = "every signal is checked for each city: prices, supply, mortgage rates, schools, jobs, permits and news"

UNSUPPORTED = {
    "crime": r"\b(crime|crimes|safety|safe|police|violent)\b",
    "school quality or ratings": r"\b(school quality|school ratings?|best schools|good schools|great schools|greatschools|test scores|school rankings?)\b",
    "a ZIP code that could not be matched to a city": r"\b\d{5}\b",  # known ZIPs are turned into cities before planning
    "buy or sell advice": (r"\b(should i (buy|sell|invest|wait)|good time to (buy|sell)|worth (buying|selling)|"
                           r"is it a good investment|tell me what to do)\b"),
    "individual properties": r"\b(this (house|home|property)|my (house|home|property)|this address)\b",
}
_COMPARE = r"\b(compare|comparison|versus|vs\.?|rank|ranking|best|worst|top|bottom|which (city|cities|market|markets)|cities|markets)\b"
_PRICE_WORDS = r"\b(price|prices|pricing|value|values|appreciat\w*|appreciation|rent|rents|rental|go up|go down|rise|rising|fall|falling|drop|crash|boom|outlook|forecast|trend|trending|going to|will)\b"
_MORTGAGE_WORDS = r"\b(mortgage|mortgages|rate|rates|interest|financing|borrowing|afford\w*|monthly payment)\b"
_SUPPLY_WORDS = r"\b(inventory|supply|listings?|days on market|buyer'?s market|seller'?s market|competitive|tight|hot market|cooling|market heat|sales volume|homes sold)\b"
_HOUSING_WORDS = r"\b(home|homes|house|houses|housing|real estate|property|properties|market|markets|rent|rents|mortgage|price|prices|inventory|supply|buyer|buyers|seller|sellers|buy|sell)\b"
_PAST_YEAR = re.compile(r"\b(19\d\d|20[01]\d|202[0-5])\b")


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Literal["outlook", "compare", "signal", "clarify", "out_of_scope"]
    locations: list[Location] = Field(default_factory=list)
    workers: list[str] = Field(default_factory=list)
    reason: str = ""
    unsupported: list[str] = Field(default_factory=list)
    clarification: str | None = None
    planner: str = "rules"
    notes: list[str] = Field(default_factory=list)
    compare_all: bool = False


class PlanDraft(BaseModel):
    """What the model is asked to return. Everything is validated before use."""
    model_config = ConfigDict(extra="ignore")

    intent: Literal["outlook", "compare", "signal", "out_of_scope"] = "outlook"
    locations: list[dict] = Field(default_factory=list)
    workers: list[str] = Field(default_factory=list)
    reason: str = ""


PLANNER_SYSTEM = """\
You route questions for a housing-market signals tool. Read the user's question and reply with \
ONLY a JSON object, no markdown fences:
{"intent": "outlook|compare|signal|out_of_scope",
 "locations": [{"city": "...", "state": "XX"}],
 "workers": ["housing_prices", "supply", "mortgage_rates", "schools", "jobs", "permits", "news"],
 "reason": "one short sentence on why these workers"}

Every signal runs for every city, so you do not choose workers; always return all seven:
housing_prices, supply, mortgage_rates, schools, jobs, permits, news.

intent: "outlook" = one place's direction; "compare" = several places or ranking; "signal" = a question about one specific signal; "out_of_scope" = not about housing markets.
locations: only places named in the question, state as a two-letter USPS code. Empty list if none are named.
ZIP codes in the question have already been replaced by the city they belong to.
The tool CANNOT do: crime, school quality or ratings, individual properties, or buy/sell advice. Do not invent workers for those.
"""


# -- helpers ----------------------------------------------------------------------------


def _unsupported_topics(query: str) -> list[str]:
    q = query.lower()
    return [label for label, pat in UNSUPPORTED.items() if re.search(pat, q)]


def _state_code(text: str | None) -> str | None:
    if not text:
        return None
    t = text.strip().rstrip(".").lower()
    if t.upper() in STATE_CODES and len(t) == 2:
        return t.upper()
    return STATE_NAMES.get(norm(t))


FILLER = {"is", "are", "will", "what", "how", "why", "does", "do", "can", "could", "should", "in", "for",
          "about", "compare", "and", "or", "vs", "the", "of", "to", "whats", "which", "where", "when",
          "tell", "me", "show", "give", "was", "were", "has", "have", "if", "with", "versus"}


def _state_after(query: str, end: int) -> str | None:
    """A state name or code right after a city mention: 'Seattle, WA' / 'Seattle Washington'."""
    tail = re.match(r"\s*,?\s*([A-Za-z]+(?:\s+[A-Za-z]+){0,2})", query[end:end + 40])
    if not tail:
        return None
    parts = tail.group(1).split()
    for n in (3, 2, 1):  # longest state name first ('New Hampshire' before 'New')
        if len(parts) >= n:
            words = " ".join(parts[:n])
            if n == 1 and len(words) == 2 and words != words.upper():
                continue  # a two-letter code must be written in capitals ('in', 'or' are not states)
            code = _state_code(words)
            if code:
                return code
    return None


def find_locations(query: str, registry: dict[str, Location]) -> tuple[list[Location], list[str]]:
    """Places named in the query, in order of appearance. Returns (locations, problems)."""
    found: list[tuple[int, Location]] = []
    problems: list[str] = []
    taken: list[tuple[int, int]] = []

    def overlaps(a: int, b: int) -> bool:
        return any(a < y and x < b for x, y in taken)

    # 1. 'City, ST' / 'City, State' for any place (capitalized words, then a state)
    for m in re.finditer(r"\b([A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){0,2}),?\s+(?=[A-Za-z])", query):
        city_words = m.group(1).split()
        drop = 0
        while drop < len(city_words) and city_words[drop].lower() in FILLER:
            drop += 1  # leading filler such as 'Is' or 'Will'
        if drop == len(city_words):
            continue
        city = " ".join(city_words[drop:])
        start = m.start(1) + len(m.group(1)) - len(city)
        end = start + len(city)
        state = _state_after(query, end)
        if state and not overlaps(start, end + 1) and norm(city) not in STATE_NAMES:
            found.append((start, Location(city=city, state=state)))
            taken.append((start, end + 1))
    # 2. registry cities mentioned without a state
    for key, loc in registry.items():
        for m in re.finditer(r"\b" + re.escape(loc.city) + r"\b", query, flags=re.IGNORECASE):
            if overlaps(m.start(), m.end()):
                continue
            same = [l for l in registry.values() if norm(l.city) == norm(loc.city)]
            if len(same) > 1:
                problems.append(f"{loc.city} matches more than one place ({', '.join(l.state for l in same)}); say which state")
                continue
            found.append((m.start(), loc))
            taken.append((m.start(), m.end()))
    found.sort(key=lambda t: t[0])
    out, seen = [], set()
    for _, loc in found:
        canon = registry.get(loc.key, loc)  # prefer the registry entry (carries FIPS etc.)
        if canon.key not in seen:
            seen.add(canon.key)
            out.append(canon)
    return out, problems


_SCHOOLS_WORDS = r"\b(school|schools|enrollment|enrolment|students|school district|school-age|families with kids)\b"
_JOBS_WORDS = r"\b(jobs?|job market|job growth|employment|unemployment|unemployed|payrolls?|layoffs?|hiring|labor market|labour market|local economy|workers)\b"
_PERMITS_WORDS = r"\b(permits?|building permits?|construction|new construction|housing starts|new development|homebuilding|home building|builders?)\b"
_NEWS_WORDS = r"\b(news|headlines?|in the news|media|press coverage|coverage|sentiment)\b"
_EXTRA_PATTERNS = {"news": (_NEWS_WORDS, "housing news"), "schools": (_SCHOOLS_WORDS, "school enrollment"), "jobs": (_JOBS_WORDS, "local jobs"),
                   "permits": (_PERMITS_WORDS, "building permits")}


def select_workers(query: str) -> tuple[list[str], str, bool]:
    """-> (workers, reason, is_single_signal_question)."""
    workers, reason, single = _select_core(query)
    q = query.lower()
    asked = [w for w, (pat, _) in _EXTRA_PATTERNS.items() if re.search(pat, q)]
    if not asked:
        return workers, reason, single
    names = " and ".join(_EXTRA_PATTERNS[w][1] for w in asked)
    explicit = re.search(_PRICE_WORDS + "|" + _SUPPLY_WORDS + "|" + _MORTGAGE_WORDS, q)
    if not explicit:
        return asked, f"the question is about {names}", True
    added = [w for w in asked if w not in workers]
    if added:
        workers = workers + added
        reason += f"; {names} added because you mentioned it"
    return workers, reason, False


def _select_core(query: str) -> tuple[list[str], str, bool]:
    q = query.lower()
    wants = []
    if re.search(_PRICE_WORDS, q) and not re.search(r"\b(mortgage|rate|rates)\b", q):
        wants += ["housing_prices", "supply"]
    if re.search(r"\b(rent|rents|rental|home values?|house prices?|home prices?)\b", q) and "housing_prices" not in wants:
        wants.append("housing_prices")
    if re.search(_SUPPLY_WORDS, q) and "supply" not in wants:
        wants.append("supply")
    if re.search(_MORTGAGE_WORDS, q):
        wants.append("mortgage_rates")
    outlook_q = bool(re.search(r"\b(will|outlook|forecast|going to|next year|go up|go down|rise|fall|trend)\b", q))
    if outlook_q:
        for w in ("housing_prices", "supply"):
            if w not in wants:
                wants.append(w)
        if "mortgage_rates" not in wants:
            wants.append("mortgage_rates")
    if not wants:
        return list(ALL_WORKERS), "no single signal named, so all signals are used", False
    wants = [w for w in ALL_WORKERS if w in wants]
    if len(wants) == len(ALL_WORKERS):
        return wants, "a general question, so all signals are used", False
    names = ", ".join(WORKER_LABELS[w].split(" (")[0].lower() for w in wants)
    return wants, f"the question is about {names}", not outlook_q


def registry_cities(registry: dict[str, Location]) -> list[Location]:
    return list(registry.values())


# -- rule-based planner -----------------------------------------------------------------


def rule_plan(query: str, registry: dict[str, Location]) -> Plan:
    unsupported = _unsupported_topics(query)
    locs, problems = find_locations(query, registry)
    notes = []
    if _PAST_YEAR.search(query):
        notes.append("This tool answers about the current market only; questions about past dates are not supported here.")
    has_compare = bool(re.search(_COMPARE, query.lower()))
    housing_related = bool(re.search(_HOUSING_WORDS, query.lower())) or bool(locs)
    if not housing_related:
        return Plan(intent="out_of_scope", unsupported=unsupported, notes=notes,
                    reason="the question does not look like it is about a housing market")
    if problems and not locs:
        return Plan(intent="clarify", clarification=problems[0] + ".", unsupported=unsupported, notes=notes)
    _picked, _why, _single = select_workers(query)  # kept for reference; every signal runs for a city question
    workers, reason, single = list(KNOWN_WORKERS), ALL_REASON, False
    if not locs:
        if has_compare:
            return Plan(intent="compare", compare_all=True, workers=workers, reason=reason,
                        unsupported=unsupported, notes=notes + ["No places named; comparing the places in the project's city list."])
        return Plan(intent="clarify", unsupported=unsupported, notes=notes,
                    clarification="Which city should I look at? Name it with its state, for example 'Seattle, WA'.")
    if problems:
        notes.extend(problems)
    intent = "compare" if len(locs) > 1 or has_compare and len(locs) > 1 else ("signal" if single else "outlook")
    return Plan(intent=intent, locations=locs, workers=workers, reason=reason,
                unsupported=unsupported, notes=notes)


# -- model-assisted planner -------------------------------------------------------------


def _validate_draft(draft: PlanDraft, query: str, registry: dict[str, Location],
                    rule: Plan) -> Plan | None:
    """Keep only what the question supports. Returns None if the draft is unusable."""
    qn = norm(query)
    locs: list[Location] = []
    notes = list(rule.notes)
    for item in draft.locations:
        city = str(item.get("city", "")).strip()
        if not city or norm(city) not in qn:  # the model may not invent a place
            notes.append(f"Ignored '{city}' from the planner: it is not in your question.")
            continue
        state = _state_code(str(item.get("state", "")))
        if not state:
            same = [l for l in registry.values() if norm(l.city) == norm(city)]
            state = same[0].state if len(same) == 1 else None
        if not state:
            notes.append(f"Could not tell which state '{city}' is in.")
            continue
        loc = Location(city=city, state=state)
        locs.append(registry.get(loc.key, loc))
    workers = [w for w in draft.workers if w in KNOWN_WORKERS]
    dropped = [w for w in draft.workers if w not in KNOWN_WORKERS]
    if dropped:
        notes.append(f"Ignored unknown workers from the planner: {dropped}.")
    if not locs:
        locs = rule.locations
    workers = list(KNOWN_WORKERS)  # whatever the model proposed, all signals run for a city question
    if draft.intent == "out_of_scope" and not locs and not rule.locations:
        return Plan(intent="out_of_scope", unsupported=rule.unsupported, notes=notes,
                    reason=draft.reason or "not about a housing market")
    if not locs:
        if rule.intent == "compare":
            return rule.model_copy(update={"workers": workers, "notes": notes})
        return Plan(intent="clarify", unsupported=rule.unsupported, notes=notes,
                    clarification="Which city should I look at? Name it with its state, for example 'Seattle, WA'.")
    intent = "compare" if len(locs) > 1 else "outlook"
    return Plan(intent=intent, locations=locs, workers=workers, reason=ALL_REASON,
                unsupported=rule.unsupported, notes=notes)


def build_plan(query: str, registry: dict[str, Location], judge: JsonJudge | None) -> Plan:
    rule = rule_plan(query, registry)
    if judge is None:
        return rule
    if rule.intent == "out_of_scope" and not _unsupported_topics(query):
        # still let the model have a look: the rules are blunt about what 'housing related' means
        pass
    try:
        listed = ", ".join(sorted({l.city for l in registry.values()})[:60])
        result = judge.judge(PLANNER_SYSTEM, f"Question: {query}\n\nCities the tool commonly handles: {listed}",
                             PlanDraft)
        plan = _validate_draft(result.value, query, registry, rule)  # type: ignore[arg-type]
        if plan is None:
            return rule.model_copy(update={"notes": rule.notes + ["The planner's answer was unusable; used rules."]})
        plan.planner = result.model
        return plan
    except Exception as exc:
        return rule.model_copy(update={
            "notes": rule.notes + [f"The language model could not plan this ({type(exc).__name__}); used rules instead."],
            "planner": "rules (model failed)"})
