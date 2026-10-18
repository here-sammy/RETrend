"""Public-school enrollment by district and year, from a CSV you download yourself.

Why a local file: enrollment comes from the federal Common Core of Data (NCES) and from state
education agencies, in many layouts, and the download links could not be verified from the build
environment. So, like Redfin, you save a file and the loader reads it by column NAME.

What the file must contain (one row per district per year, district TOTALS, not per grade or school):
  - a district name   (district, district_name, lea_name, agency_name, school_district, name)
  - a state           (state, st, stabr, state_name, lstate, state_abbr): code or full name
  - a year            (year, school_year, survey_year, fall_year): '2023', '2023-24' or '2023-2024'
  - a student count   (enrollment, students, student_count, membership, total_enrollment, total_students)

Save it as data/files/schools/<anything>.csv, or set SCHOOLS_FILE=C:\\path\\to\\file.csv.
If the file has several rows for one district and year (per-grade or per-school rows), loading stops
with an explanation instead of guessing, because summing them could double count.
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Any

from retrend.location import Location
from retrend.sources.files import SourceError
from retrend.sources.matching import closest_names, norm

ALIASES = {
    "name": ("district", "district_name", "lea_name", "agency_name", "school_district", "name",
             "leaname", "district_or_school", "region"),
    "state": ("state", "st", "stabr", "state_name", "lstate", "state_abbr", "state_code"),
    "year": ("year", "school_year", "survey_year", "fall_year", "schoolyear", "period"),
    "city": ("city", "city_location", "location_city", "mailing_city"),
    "students": ("enrollment", "students", "student_count", "membership", "total_enrollment",
                 "total_students", "member", "enrolled", "total_membership"),
}
STATE_NAMES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA", "colorado": "CO",
    "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA",
    "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD", "tennessee": "TN", "texas": "TX",
    "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC",
}
CODES = set(STATE_NAMES.values())


class NoDistrictMatch(SourceError):
    """The file has no district matching the location."""


def _key(h: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", h.strip().lower()).strip("_")


def parse_year(text: str | None) -> int | None:
    """Fall year of a school year: '2023-24' -> 2023, '2023' -> 2023."""
    m = re.match(r"\s*((?:19|20)\d\d)", text or "")
    return int(m.group(1)) if m else None


def parse_count(text: str | None) -> float | None:
    t = re.sub(r"[,\s]", "", text or "")
    if t in ("", "-", "NA", "N/A", "n/a", "NULL", "†", "‡", "–", "-1", "-2", "-9"):
        return None  # NCES uses negative codes for missing / not applicable
    try:
        v = float(t)
    except ValueError:
        return None
    return v if v > 0 else None


def parse_state(text: str | None) -> str | None:
    t = (text or "").strip()
    if t.upper() in CODES:
        return t.upper()
    return STATE_NAMES.get(norm(t))


def district_matches(name: str, city: str) -> bool:
    """'Seattle Public Schools', 'Seattle School District No. 1' and 'Austin ISD' all match their city.
    The city's words must appear in order, whole words, at the START of the district name
    (so 'Portland' matches 'Portland Public Schools' but not 'North Portland Charter')."""
    n, c = norm(name).split(), norm(city).split()
    return bool(c) and n[: len(c)] == c


def extract_district(lines: Any, location: Location, *, file_name: str = "", delimiter: str = ",") -> dict:
    reader = csv.reader(lines, delimiter=delimiter)
    header = next(reader, None)
    if not header:
        raise SourceError("school enrollment file is empty")
    cols = {_key(h): i for i, h in enumerate(header)}
    idx = {}
    for field, names in ALIASES.items():
        hit = next((cols[n] for n in names if n in cols), None)
        if hit is None and field == "city":
            continue  # optional
        if hit is None:
            raise SourceError(
                f"school enrollment file has no {field} column (looked for {list(names)}); "
                f"columns found: {[h for h in header][:12]}")
        idx[field] = hit
    by_district: dict[str, dict[int, float]] = {}
    seen: set[tuple[str, int]] = set()
    same_state: list[str] = []
    for row in reader:
        try:
            name, st = row[idx["name"]].strip(), parse_state(row[idx["state"]])
        except IndexError:
            continue
        if st != location.state or not name:
            continue
        same_state.append(name)
        city = row[idx["city"]].strip() if "city" in idx and idx["city"] < len(row) else ""
        if not (district_matches(name, location.city) or (city and norm(city) == norm(location.city))):
            continue
        year, count = parse_year(row[idx["year"]]), parse_count(row[idx["students"]])
        if year is None:
            continue
        if (name, year) in seen:
            raise SourceError(
                f"{file_name or 'the file'} has more than one row for '{name}' in {year}; it looks like "
                "per-grade or per-school rows. Use a district-level total file (one row per district and year).")
        seen.add((name, year))
        if count is not None:
            by_district.setdefault(name, {})[year] = count
    if not by_district:
        close = closest_names(location.city, sorted(set(same_state)))
        raise NoDistrictMatch(
            f"no school district starting with '{location.city}' in {location.state}"
            + (f"; closest names in {location.state}: {close}" if close else ""))
    # several matches: take the one with the most students in its latest year
    best = max(by_district, key=lambda d: by_district[d][max(by_district[d])])
    years = by_district[best]
    return {
        "source": "schools", "file": file_name, "district": best,
        "quality": "exact" if len(by_district) == 1 else "approximate",
        "n_candidates": len(by_district), "other_candidates": sorted(set(by_district) - {best})[:5],
        "rows": [{"year": y, "students": years[y]} for y in sorted(years)],
    }


class SchoolsLoader:
    def __init__(self, data_dir: str | Path) -> None:
        self._dir = Path(data_dir)

    def _file(self) -> Path:
        env = os.environ.get("SCHOOLS_FILE")
        if env:
            p = Path(env).expanduser()
            if not p.is_file():
                raise SourceError(f"SCHOOLS_FILE points to {p}, which does not exist")
            return p
        if self._dir.is_dir():
            for p in sorted(self._dir.iterdir()):
                if p.is_file() and p.suffix.lower() in (".csv", ".tsv", ".txt"):
                    return p
        raise SourceError(
            f"no school enrollment file. Save a district-level CSV in {self._dir} or set SCHOOLS_FILE "
            "(see the README, 'School enrollment').")

    def extract(self, location: Location, geo: str = "district") -> dict:
        path = self._file()
        try:
            with open(path, encoding="utf-8-sig", newline="") as f:
                first = f.readline()
                f.seek(0)
                delim = "\t" if first.count("\t") > first.count(",") else ","
                payload = extract_district(f, location, file_name=path.name, delimiter=delim)
        except (OSError, UnicodeDecodeError) as exc:
            raise SourceError(f"could not read {path.name}: {exc}") from None
        payload["file_url"] = f"file:{path.name}"
        return payload
