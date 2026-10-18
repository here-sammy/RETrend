"""Download district enrollment from the Urban Institute Education Data Portal (no API key needed).

    retrend schools-fetch                          # newest year with data and the 9 before it
    retrend schools-fetch --start 2013 --end 2022

Each year is saved in the project database (data/cache.db) and all years are exported to
data/files/schools/urban_enrollment.csv, which the schools worker reads.

IMPORTANT: written from memory of the portal's documented layout, NOT tested against the live API
(the build environment cannot reach it). Assumed:
  GET https://educationdata.urban.org/api/v1/school-districts/ccd/directory/{year}/?fips={state fips}
  -> {"count": N, "next": url-or-null, "results": [{"leaid", "lea_name", "state_location", "year",
                                                    "enrollment", ...}]}
Negative numbers mean missing/suppressed. If a field is named differently, the first run stops with the
fields it found, and the names are a one-line change at the top of this file.
"""

from __future__ import annotations

import csv
import json
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Callable

from retrend.sources.files import SourceError

BASE = "https://educationdata.urban.org/api/v1/school-districts/ccd/directory"
F_ID, F_NAME, F_STATE, F_YEAR, F_COUNT = "leaid", "lea_name", "state_location", "year", "enrollment"
FIPS = {
    "AL": 1, "AK": 2, "AZ": 4, "AR": 5, "CA": 6, "CO": 8, "CT": 9, "DE": 10, "DC": 11, "FL": 12, "GA": 13,
    "HI": 15, "ID": 16, "IL": 17, "IN": 18, "IA": 19, "KS": 20, "KY": 21, "LA": 22, "ME": 23, "MD": 24,
    "MA": 25, "MI": 26, "MN": 27, "MS": 28, "MO": 29, "MT": 30, "NE": 31, "NV": 32, "NH": 33, "NJ": 34,
    "NM": 35, "NY": 36, "NC": 37, "ND": 38, "OH": 39, "OK": 40, "OR": 41, "PA": 42, "RI": 44, "SC": 45,
    "SD": 46, "TN": 47, "TX": 48, "UT": 49, "VT": 50, "VA": 51, "WA": 53, "WV": 54, "WI": 55, "WY": 56,
}
HEADER = ["District Name", "State", "School Year", "Enrollment", "LEAID"]
Opener = Callable[[str], bytes]
FIPS_CODES = FIPS


TIMEOUT = 120  # seconds per request; the portal can be slow, and unpublished years may hang
PAGE = 1000    # rows per request: smaller pages time out less than one huge response


def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "retrend/0.1 (school enrollment pull)"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.read()


def fetch_pages(url: str, get: Opener, *, pause: float = 0.2, retries: int = 3) -> list[dict]:
    """All results for a URL, following 'next' links."""
    out: list[dict] = []
    seen = set()
    while url:
        if url in seen:
            raise SourceError(f"paging loop at {url}")
        seen.add(url)
        for attempt in range(retries):
            try:
                body = json.loads(get(url))
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    return out  # no data for this year/state
                if attempt == retries - 1:
                    raise SourceError(f"HTTP {exc.code} from {url}") from None
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                if attempt == retries - 1:
                    raise SourceError(f"could not fetch {url}: {exc}") from None
            time.sleep(1.5 * (attempt + 1))
        out.extend(body.get("results") or [])
        url = body.get("next")
        if url:
            time.sleep(pause)
    return out


def rows_from_results(results: list[dict], state: str) -> list[list]:
    rows = []
    for r in results:
        if r and F_NAME not in r and F_COUNT not in r:
            raise SourceError(f"unexpected record layout; fields found: {sorted(r)[:15]}")
        count = r.get(F_COUNT)
        if not isinstance(count, (int, float)) or count <= 0:  # negative codes = missing/suppressed
            continue
        year = r.get(F_YEAR)
        rows.append([str(r.get(F_NAME, "")).strip(), r.get(F_STATE) or state, f"{year}-{str(int(year) + 1)[2:]}",
                     int(count), r.get(F_ID, "")])
    return rows


def latest_available_year(get: Opener, *, today: date | None = None) -> int:
    """Newest school year (fall year) that returns data, probing downward from last year."""
    y = (today or date.today()).year - 2  # the newest published year trails the calendar by 2+ years
    errors = []
    for cand in range(y, y - 6, -1):
        try:
            if fetch_pages(f"{BASE}/{cand}/?fips=53&limit=1", get, pause=0, retries=1):
                return cand
        except SourceError as exc:  # a timeout on one year (often an unpublished one) must not stop the search
            errors.append(f"{cand}: {exc}")
    raise SourceError("could not find a recent year with data (" + "; ".join(errors[:3]) +
                      "). Pass --end YEAR yourself, for example --end 2022.")


SOURCE = "urban_enrollment"


def _getter(open_url) -> Opener:
    if open_url is None:
        return _http_get
    return lambda url: open_url(url).read()


def fetch_years(db_path, years: list[int], *, open_url=None, say: Callable[[str], None] = print,
                force: bool = False, pause: float = 0.2) -> dict:
    """Download each school year (all districts) into the project's SQLite database. Years already
    saved are skipped unless ``force``. -> {"fetched": [...], "skipped": [...], "failed": [...]}"""
    from retrend.store import RawStore

    get = _getter(open_url)
    summary: dict = {"fetched": [], "skipped": [], "failed": []}
    with RawStore(db_path) as store:
        for yr in years:
            key = str(yr)
            if not force and store.get_latest(SOURCE, key) is not None:
                summary["skipped"].append(yr)
                continue
            try:
                results = fetch_pages(f"{BASE}/{yr}/?limit={PAGE}", get, pause=pause)
                rows = rows_from_results(results, "")
            except SourceError as exc:
                say(f"  {yr}: failed ({exc})")
                summary["failed"].append(yr)
                continue
            if not rows:
                say(f"  {yr}: no data returned")
                summary["failed"].append(yr)
                continue
            store.put(SOURCE, key, {"year": yr, "rows": rows})
            summary["fetched"].append(yr)
            say(f"  {yr}: {len(rows)} districts saved")
    return summary


def export_csv(db_path, out_csv) -> int:
    """Write every saved year to one CSV the schools worker can read. -> number of rows."""
    from retrend.store import RawStore

    rows: list[list] = []
    with RawStore(db_path) as store:
        for key in store.keys(SOURCE):
            rec = store.get_latest(SOURCE, key)
            if rec is not None:
                rows.extend(rec.json()["rows"])
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        w.writerows(sorted(rows, key=lambda r: (r[1], r[0], r[2])))
    tmp.replace(out)
    return len(rows)
