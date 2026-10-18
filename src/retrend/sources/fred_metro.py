"""Find the FRED series for a city's metro area (unemployment, payrolls, building permits).

FRED names these series after the metro area ("Unemployment Rate in Seattle-Tacoma-Bellevue, WA (MSA)")
with IDs that cannot be derived from a city name, so the finder searches FRED by title and keeps the
series whose metro name contains the city and whose state list contains the state. The result is cached
for a month. If the search picks the wrong series (or none), pin the right IDs yourself in
config/fred_series.yaml:

    seattle-wa:
      unemployment: SEAT653URN
      payrolls: SEAT653NA
      permits: SEAT653BPPRIV

The title patterns below are written from memory of FRED's naming and have only been run against fakes.
"""

from __future__ import annotations

import re
from datetime import timedelta
from pathlib import Path
from typing import Callable

import yaml

from retrend.location import Location
from retrend.sources.files import SourceError
from retrend.sources.matching import metro_matches, norm
from retrend.store import RawStore

SOURCE = "fred_metro_search"
MAX_AGE = timedelta(days=30)
KINDS = ("unemployment", "payrolls", "permits")
SEARCH_TEXT = {
    "unemployment": "Unemployment Rate in {city}",
    "payrolls": "All Employees Total Nonfarm in {city}",
    "permits": "Authorized by Building Permits {city}",
}
TITLE = {
    "unemployment": re.compile(r"^Unemployment Rate in (?P<area>.+), (?P<st>[A-Z]{2}(?:-[A-Z]{2})*) \((?:MSA|CBSA)\)$"),
    "payrolls": re.compile(r"^All Employees: Total Nonfarm (?:in|for) (?P<area>.+), (?P<st>[A-Z]{2}(?:-[A-Z]{2})*) \((?:MSA|CBSA)\)$"),
    "permits": re.compile(r"Authorized by Building Permits for (?P<area>.+), (?P<st>[A-Z]{2}(?:-[A-Z]{2})*) \((?:MSA|CBSA)\)$"),
}
_SUBTYPE = re.compile(r"\b\d\+?-?\s*(unit|units)\b|\bor more\b", re.I)  # '1-Unit', '5 Units or More': not the total


class NoMetroSeries(SourceError):
    pass


def load_overrides(path: str | Path) -> dict[str, dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return {}
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    out: dict[str, dict[str, str]] = {}
    for key, kinds in raw.items():
        if not isinstance(kinds, dict):
            raise SourceError(f"{p.name}: '{key}' must map unemployment/payrolls/permits to series IDs")
        bad = set(kinds) - set(KINDS)
        if bad:
            raise SourceError(f"{p.name}: unknown series kind(s) {sorted(bad)} for '{key}'; use {list(KINDS)}")
        out[str(key)] = {k: str(v).strip() for k, v in kinds.items()}
    return out


def match_rank(area: str, states: str, location: Location) -> int | None:
    """Lower is better; None = not this city's metro. 0 = the metro is named for the city alone."""
    if not metro_matches(f"{area}, {states}", location.city, location.state):
        return None
    return 0 if norm(area) == norm(location.city) else 1


def pick_series(kind: str, rows: list[dict], location: Location) -> dict | None:
    pat = TITLE[kind]
    best: tuple | None = None
    for i, r in enumerate(rows):
        title = r.get("title") or ""
        m = pat.search(title)
        if not m or r.get("frequency_short") != "M":
            continue
        if kind == "permits" and _SUBTYPE.search(title):
            continue
        rank = match_rank(m.group("area"), m.group("st"), location)
        if rank is None:
            continue
        sa = 0 if r.get("seasonal_adjustment_short") == "SA" else 1  # prefer seasonally adjusted
        cand = ((rank, sa, i), {"series_id": r["id"], "title": title, "area": m.group("area") + ", " + m.group("st"),
                                "seasonal_adjustment": r.get("seasonal_adjustment_short")})
        if best is None or cand[0] < best[0]:
            best = cand
    return best[1] if best else None


class MetroSeriesFinder:
    def __init__(self, store: RawStore, search: Callable[[str], list[dict]] | None,
                 overrides: dict[str, dict[str, str]] | None = None) -> None:
        self._store, self._search, self._overrides = store, search, overrides or {}

    def find(self, kind: str, location: Location) -> dict:
        """-> {"series_id", "title", "area", ...}. Raises NoMetroSeries with advice if none is found."""
        if kind not in KINDS:
            raise ValueError(f"unknown series kind {kind!r}")
        pinned = self._overrides.get(location.key, {}).get(kind)
        if pinned:
            return {"series_id": pinned, "title": f"{pinned} (set in config/fred_series.yaml)",
                    "area": f"{location.city}, {location.state}", "seasonal_adjustment": None, "pinned": True}
        if self._search is None:
            raise NoMetroSeries("no FRED search is available here (needs FRED_API_KEY)")
        rec = self._store.fetch_or_cache(
            SOURCE, f"{kind}|{location.key}",
            lambda: self._search(SEARCH_TEXT[kind].format(city=location.city)), max_age=MAX_AGE)
        found = pick_series(kind, rec.json(), location)
        if found is None:
            raise NoMetroSeries(
                f"no FRED {kind} series found for the {location.city}, {location.state} metro area. "
                f"If one exists, add its ID for '{location.key}' in config/fred_series.yaml.")
        return found
