"""Zillow Research files: ZHVI (home values) and ZORI (rents).

Layout assumed (checked against Zillow's published files from memory, not against a live
download in the build environment): one row per region, a few identifier columns
(RegionName, StateName, SizeRank, ...), then one column per month named YYYY-MM-DD. We
find the date columns by their names and read identifiers by name, so extra or reordered
columns are fine; a missing RegionName column fails loudly.

URLs: the two Metro files are the ones linked on https://www.zillow.com/research/data/.
The two City files follow the same naming pattern but were not listed there; if one is
wrong, set ZILLOW_<DATASET>_<GEO>_URL (e.g. ZILLOW_ZHVI_CITY_URL) in .env.
"""

from __future__ import annotations

import csv
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from retrend.location import Location
from retrend.sources.files import SourceError, download_cached
from retrend.sources.matching import closest_names, metro_matches, norm

BASE = "https://files.zillowstatic.com/research/public_csvs"
DEFAULT_URLS = {
    ("zhvi", "metro"): f"{BASE}/zhvi/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
    ("zhvi", "city"): f"{BASE}/zhvi/City_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
    ("zori", "metro"): f"{BASE}/zori/Metro_zori_uc_sfrcondomfr_sm_month.csv",
    ("zori", "city"): f"{BASE}/zori/City_zori_uc_sfrcondomfr_sm_month.csv",
    # Days to pending (how long a listing takes to go under contract). Zillow lists only a metro-level,
    # mean, smoothed file on its data page. For a median file, set ZILLOW_DTP_METRO_URL in .env.
    ("dtp", "metro"): f"{BASE}/mean_doz_pending/Metro_mean_doz_pending_uc_sfrcondo_sm_month.csv",
}
DESCRIPTIONS = {
    "zhvi": "Zillow Home Value Index, all homes (SFR + condo), middle tier, smoothed, "
            "seasonally adjusted",
    "zori": "Zillow Observed Rent Index, all homes + multifamily, smoothed",
    "dtp": "Zillow days to pending, all homes, metro level, smoothed",
}
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class NoRegionMatch(SourceError):
    """The file has no region matching the location."""


def resolve_url(dataset: str, geo: str, overrides: dict | None = None) -> str:
    env_name = f"ZILLOW_{dataset.upper()}_{geo.upper()}_URL"
    url = (overrides or {}).get((dataset, geo)) or os.environ.get(env_name) or DEFAULT_URLS.get(
        (dataset, geo)
    )
    if not url:
        raise SourceError(f"no Zillow URL known for dataset={dataset!r} geo={geo!r}")
    return url


def _cell(row: list[str], cols: dict[str, int], name: str) -> str | None:
    i = cols.get(name)
    return row[i].strip() if i is not None and i < len(row) and row[i].strip() else None


def _to_float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def extract_region(
    lines: Any, location: Location, dataset: str, geo: str, *, file_name: str = ""
) -> dict:
    """Stream a Zillow CSV (an iterable of text lines) and return the best-matching
    region's series as a small JSON-able payload."""
    reader = csv.reader(lines)
    header = next(reader, None)
    if not header:
        raise SourceError("Zillow file is empty")
    cols = {name.strip(): i for i, name in enumerate(header)}
    if "RegionName" not in cols:
        raise SourceError(f"unexpected Zillow file layout; first columns: {header[:8]}")
    date_cols = [(i, h.strip()) for i, h in enumerate(header) if _DATE.match(h.strip())]
    if not date_cols:
        raise SourceError("Zillow file has no YYYY-MM-DD date columns")

    state = location.state
    candidates: list[tuple[float, list[str]]] = []
    same_state_names: list[str] = []
    for row in reader:
        name = _cell(row, cols, "RegionName")
        if not name:
            continue
        row_state = (_cell(row, cols, "StateName") or _cell(row, cols, "State") or "").upper()
        if geo == "city":
            if row_state == state:
                same_state_names.append(name)
            matched = row_state == state and norm(name) == norm(location.city)
        else:
            if (_cell(row, cols, "RegionType") or "").lower() == "country":
                continue
            matched = metro_matches(name, location.city, state)
            if not matched and name.rpartition(",")[2].strip().upper() == state:
                same_state_names.append(name)
        if matched:
            rank = _to_float(_cell(row, cols, "SizeRank") or "")
            candidates.append((rank if rank is not None else float("inf"), row))

    if not candidates:
        close = closest_names(location.city, same_state_names)
        raise NoRegionMatch(
            f"no {geo}-level Zillow region matches {location.city}, {state}"
            + (f"; closest names in {state}: {close}" if close else "")
        )

    candidates.sort(key=lambda c: c[0])
    row = candidates[0][1]
    observations = []
    for i, day in date_cols:
        value = _to_float(row[i]) if i < len(row) and row[i].strip() else None
        if value is not None:
            observations.append({"date": day, "value": value})
    return {
        "source": "zillow",
        "dataset": dataset,
        "geo": geo,
        "file": file_name,
        "region": {
            "name": _cell(row, cols, "RegionName"),
            "state": _cell(row, cols, "StateName") or _cell(row, cols, "State"),
            "region_id": _cell(row, cols, "RegionID"),
            "metro": _cell(row, cols, "Metro"),
            "county": _cell(row, cols, "CountyName"),
        },
        # A single city-level match is exact; anything else is a stand-in for the city.
        "quality": "exact" if geo == "city" and len(candidates) == 1 else "approximate",
        "n_candidates": len(candidates),
        "observations": observations,
    }


class ZillowLoader:
    """Downloads Zillow files (cached on disk) and extracts one region from them."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        open_url: Callable[[str], Any] | None = None,
        clock: Callable[[], datetime] | None = None,
        max_file_age: timedelta = timedelta(days=7),
        urls: dict | None = None,
    ) -> None:
        self._dir = Path(data_dir)
        self._open_url = open_url
        self._clock = clock
        self._max_age = max_file_age
        self._urls = urls

    def extract(self, location: Location, dataset: str, geo: str) -> dict:
        url = resolve_url(dataset, geo, self._urls)
        path = download_cached(
            url,
            self._dir,
            max_age=self._max_age,
            now=self._clock,
            open_url=self._open_url,
            hint=f"If the link has moved, set ZILLOW_{dataset.upper()}_{geo.upper()}_URL in .env "
                 "(links are on https://www.zillow.com/research/data/).",
        )
        with open(path, encoding="utf-8-sig", newline="") as f:
            payload = extract_region(f, location, dataset, geo, file_name=path.name)
        payload["file_url"] = url
        return payload
