"""Matching a Location to a data source's own region names.

"City" means different things in different sources, so every match reports its quality:
"exact" (the source has this city) or "approximate" (we used the surrounding metro, or
picked one of several regions with the same name).
"""

from __future__ import annotations

import difflib
import re


def norm(text: str) -> str:
    """Lower-case, no periods, single spaces: 'St. Louis' == 'st louis'."""
    return re.sub(r"\s+", " ", text.replace(".", " ")).strip().casefold()


def metro_matches(name: str, city: str, state: str) -> bool:
    """True if a metro name like 'Seattle, WA' or 'Minneapolis-St. Paul, MN-WI' has
    ``city`` as one of its principal cities and ``state`` among its states."""
    head, _, tail = name.rpartition(",")
    if not head:
        return False
    states = {s.strip().upper() for s in re.split(r"[-/ ]+", tail) if s.strip()}
    if state.upper() not in states:
        return False
    wanted = norm(city)
    return wanted == norm(head) or wanted in {norm(part) for part in re.split(r"[-/]", head)}


def place_matches(name: str, city: str, state: str) -> bool:
    """True for 'Seattle, WA' style place names."""
    head, _, tail = name.rpartition(",")
    return bool(head) and norm(head) == norm(city) and tail.strip().upper() == state.upper()


def closest_names(wanted: str, names: list[str], n: int = 5) -> list[str]:
    by_norm = {norm(x): x for x in names}
    return [by_norm[m] for m in difflib.get_close_matches(norm(wanted), list(by_norm), n=n)]
