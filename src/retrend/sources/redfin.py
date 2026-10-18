"""Redfin Data Center market tracker files (gzipped TSV, one row per region and period).

Column names come from Redfin's published tracker layout as documented by a third-party
schema listing (snake_case names such as period_end, region, property_type, inventory,
months_of_supply, price_drops); metric definitions were checked against Redfin's own
definitions page. Neither was checked against a live download in the build environment,
so the loader reads columns by name, treats "NA" as missing, and fails loudly (listing
what it found) if a required column is absent or no rows match.

URLS ARE UNVERIFIED. Redfin's landing and downloads pages did not list the file links.
If a default fails, copy the current link from https://www.redfin.com/news/data-center/
and set REDFIN_METRO_URL / REDFIN_CITY_URL in .env. The city file is large (hundreds of
MB compressed); the metro file is much smaller, which is why metro is the default.
"""

from __future__ import annotations

import csv
import gzip
import io
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from retrend.location import Location
from retrend.sources.files import SourceError, download_cached
from retrend.sources.matching import closest_names, metro_matches, norm, place_matches

BASE = "https://redfin-public-data.s3.us-west-2.amazonaws.com/redfin_market_tracker"
DEFAULT_URLS = {
    "metro": f"{BASE}/redfin_metro_market_tracker.tsv000.gz",
    "city": f"{BASE}/city_market_tracker.tsv000.gz",
}
REGION_TYPES = {"metro": "metro", "city": "place"}
PROPERTY_TYPE = "All Residential"
REQUIRED = ("period_end", "region")  # region_type / property_type are optional columns
FIELDS = (
    "period_begin", "period_end", "median_sale_price", "homes_sold", "pending_sales",
    "new_listings", "inventory", "months_of_supply", "median_dom", "avg_sale_to_list",
    "sold_above_list", "price_drops", "off_market_in_two_weeks",
)
# Year-over-year columns some exports provide directly, in percent points (or days for DOM).
YOY_FIELDS = ("median_sale_price_yoy", "homes_sold_yoy", "pending_sales_yoy",
              "new_listings_yoy", "inventory_yoy", "median_dom_yoy")
_FALSE = {"f", "false", "0", "no", ""}


class NoRegionMatch(SourceError):
    """The file has no rows matching the location."""


def resolve_url(geo: str, overrides: dict | None = None) -> str:
    url = (overrides or {}).get(geo) or os.environ.get(f"REDFIN_{geo.upper()}_URL") or DEFAULT_URLS.get(geo)
    if not url:
        raise SourceError(f"no Redfin URL known for geo={geo!r}")
    return url


def _num(text: str | None) -> float | None:
    """Parse '1,234', '$500K'-free dollars and '12.5%' (returned as a fraction)."""
    if text is None:
        return None
    text = text.strip()
    if text in ("", "NA", "N/A", "nan", "-"):
        return None
    pct = text.endswith("%")
    text = text.replace("$", "").replace(",", "").rstrip("%").strip()
    try:
        value = float(text)
    except ValueError:
        return None
    return value / 100.0 if pct else value


ALIASES = {
    "period_end": ("period_end", "period_end_date", "month", "date", "period"),
    "period_begin": ("period_begin", "period_begin_date"),
    "frequency": ("frequency",),
    "median_sale_price_yoy": ("median_sale_price_nsa_yoy", "median_sale_price_yoy"),
    "homes_sold_yoy": ("homes_sold_yoy",),
    "pending_sales_yoy": ("pending_sales_yoy",),
    "new_listings_yoy": ("new_listings_yoy",),
    "inventory_yoy": ("active_listings_yoy", "inventory_yoy"),
    "median_dom_yoy": ("median_days_on_market_yoy_days", "median_dom_yoy"),
    "region": ("region", "region_name", "metro", "metro_name", "name"),
    "region_type": ("region_type",),
    "property_type": ("property_type",),
    "state_code": ("state_code", "state"),
    "median_sale_price": ("median_sale_price", "median_sale_price_nsa"),
    "homes_sold": ("homes_sold",),
    "pending_sales": ("pending_sales",),
    "new_listings": ("new_listings",),
    "inventory": ("inventory", "active_listings"),
    "months_of_supply": ("months_of_supply",),
    "median_dom": ("median_dom", "median_days_on_market", "median_days_on_market_days"),
    "avg_sale_to_list": ("avg_sale_to_list", "average_sale_to_list", "sale_to_list_ratio"),
    "sold_above_list": ("sold_above_list", "share_of_homes_sold_above_list"),
    "price_drops": ("price_drops", "share_of_listings_with_price_drops"),
    "off_market_in_two_weeks": ("off_market_in_two_weeks",),
}


def _pp(text: str | None) -> float | None:
    """A value already in percent points: '5.2' and '5.2%' both mean 5.2."""
    if text is None:
        return None
    v = _num(text.strip().rstrip("%") if text.strip().endswith("%") else text)
    return v


def normalize_header(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _canonical_headers(fieldnames: list[str]) -> list[str]:
    """Map whatever the file calls its columns onto the names this module uses."""
    lookup = {alias: canon for canon, aliases in ALIASES.items() for alias in aliases}
    out = []
    for name in fieldnames:
        n = normalize_header(name)
        out.append(lookup.get(n, n))
    return out


def _parse_period(text: str | None) -> str | None:
    """ISO date out of '2026-01-31', '1/31/2026', 'January 2026' (-> month end)."""
    from calendar import monthrange

    if not text:
        return None
    text = text.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass
    for fmt in ("%B %Y", "%b %Y", "%Y-%m", "%b-%y", "%B-%y"):
        try:
            d = datetime.strptime(text, fmt).date()
            return d.replace(day=monthrange(d.year, d.month)[1]).isoformat()
        except ValueError:
            pass
    return None


def extract_region(
    lines: Any, location: Location, geo: str, *, period_days: int = 30, file_name: str = "",
    delimiter: str = "\t",
) -> dict:
    """Stream a Redfin tracker (an iterable of text lines) and return one region's monthly
    series for the "All Residential" property type, as a small JSON-able payload."""
    reader = csv.DictReader(lines, delimiter=delimiter)
    if reader.fieldnames:
        reader.fieldnames = _canonical_headers(reader.fieldnames)
    missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
    if missing:
        raise SourceError(
            f"unexpected Redfin file layout: missing columns {missing}; "
            f"found {(reader.fieldnames or [])[:10]}"
        )
    want_type = REGION_TYPES[geo]
    has_rtype = "region_type" in (reader.fieldnames or [])
    has_ptype = "property_type" in (reader.fieldnames or [])
    has_duration = "period_duration" in (reader.fieldnames or [])
    has_sa = "is_seasonally_adjusted" in (reader.fieldnames or [])
    state = location.state

    by_region: dict[str, dict[str, dict]] = {}
    durations_seen: set[str] = set()
    names_in_state: list[str] = []
    for row in reader:
        if has_rtype:
            rtype = (row.get("region_type") or "").strip().lower()
            other = ("place", "city") if geo == "metro" else ("metro",)
            if any(o in rtype for o in other):
                continue
        region = (row.get("region") or "").strip()
        if geo == "city":
            row_state = (row.get("state_code") or "").strip().upper() or \
                region.rpartition(",")[2].strip().upper()
            matched = place_matches(region, location.city, state) or (
                row_state == state
                and norm(row.get("city") or region) == norm(location.city)
            )
            if row_state == state:
                names_in_state.append(region)
        else:
            base = re.sub(r"\s+metro area\s*$", "", region, flags=re.IGNORECASE)
            matched = metro_matches(base, location.city, state)
            if not matched and base.rpartition(",")[2].strip().upper() == state:
                names_in_state.append(region)
        if not matched:
            continue
        if has_ptype and (row.get("property_type") or "").strip() != PROPERTY_TYPE:
            continue
        duration = (row.get("period_duration") or "").strip() if has_duration else ""
        durations_seen.add(duration)
        if has_duration and duration != str(period_days):
            continue
        seasonal = has_sa and (row.get("is_seasonally_adjusted") or "").strip().lower() not in _FALSE
        record = {k: _parse_period(row.get(k)) if k.startswith("period_") else _num(row.get(k))
                  for k in FIELDS}
        for k in YOY_FIELDS:
            record[k] = _pp(row.get(k))
        record["_freq"] = (row.get("frequency") or "").strip().lower()
        if not record["period_end"]:
            continue
        slot = by_region.setdefault(region, {})
        key = record["period_end"]
        # Prefer the not-seasonally-adjusted row when both exist for a period.
        if key not in slot or (slot[key]["_seasonal"] and not seasonal):
            record["_seasonal"] = seasonal
            slot[key] = record

    if not by_region:
        close = closest_names(location.city, names_in_state)
        hint = f" (rows exist for period lengths {sorted(durations_seen)}, wanted {period_days})" \
            if durations_seen and str(period_days) not in durations_seen else ""
        raise NoRegionMatch(
            f"no {geo}-level Redfin rows match {location.city}, {state}, "
            f"{PROPERTY_TYPE}{hint}" + (f"; closest in {state}: {close}" if close else "")
        )

    region = max(by_region, key=lambda r: len(by_region[r]))
    rows = sorted(by_region[region].values(), key=lambda r: r["period_end"])
    if any(r["_freq"].startswith("month") for r in rows):  # drop weekly/other rows if mixed
        rows = [r for r in rows if r["_freq"].startswith("month")]
    for r in rows:
        r.pop("_seasonal", None)
        r.pop("_freq", None)
    return {
        "source": "redfin",
        "geo": geo,
        "file": file_name,
        "region": {"name": region},
        "property_type": PROPERTY_TYPE,
        "period_days": period_days,
        "quality": "exact" if geo == "city" and len(by_region) == 1 else "approximate",
        "n_candidates": len(by_region),
        "rows": rows,
    }


class RedfinLoader:
    """Downloads Redfin tracker files (cached on disk) and extracts one region."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        open_url: Callable[[str], Any] | None = None,
        clock: Callable[[], datetime] | None = None,
        max_file_age: timedelta = timedelta(days=7),
        urls: dict | None = None,
        period_days: int = 30,
    ) -> None:
        self._dir = Path(data_dir)
        self._open_url = open_url
        self._clock = clock
        self._max_age = max_file_age
        self._urls = urls
        self._period_days = period_days

    def _local_file(self, geo: str) -> Path | None:
        """A file you downloaded by hand: REDFIN_<GEO>_FILE, or a file in the data folder
        whose name contains 'all_cities' / 'all_metros' (or 'city' / 'metro')."""
        env = os.environ.get(f"REDFIN_{geo.upper()}_FILE")
        if env:
            p = Path(env).expanduser()
            if not p.is_file():
                raise SourceError(f"REDFIN_{geo.upper()}_FILE points to {p}, which does not exist")
            return p
        words = ("all_cities", "all_city") if geo == "city" else ("all_metros", "all_metro")
        if self._dir.is_dir():
            for p in sorted(self._dir.iterdir()):
                low = p.name.lower()
                if p.is_file() and not low.endswith(".part") and any(w in low for w in words):
                    return p
        return None

    def extract(self, location: Location, geo: str) -> dict:
        local = self._local_file(geo)
        if local is not None:
            with open(local, "rb") as f:
                gz = f.read(2) == b"\x1f\x8b"
            opener = gzip.open if gz else open
            try:
                with opener(local, "rt", encoding="utf-8-sig", newline="") as text:
                    first = text.readline()
                    text.seek(0)
                    delim = "\t" if first.count("\t") > first.count(",") else ","
                    payload = extract_region(text, location, geo, period_days=self._period_days,
                                             file_name=local.name, delimiter=delim)
            except (OSError, EOFError, UnicodeDecodeError) as exc:
                raise SourceError(f"could not read {local.name}: {exc}") from None
            payload["file_url"] = f"file:{local.name}"
            return payload
        url = resolve_url(geo, self._urls)
        path = download_cached(
            url,
            self._dir,
            max_age=self._max_age,
            now=self._clock,
            open_url=self._open_url,
            hint=f"The default Redfin link is unverified. Copy the current one from "
                 f"https://www.redfin.com/news/data-center/ and set REDFIN_{geo.upper()}_URL in .env.",
        )
        try:
            with gzip.open(path, "rb") as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
                payload = extract_region(
                    text, location, geo, period_days=self._period_days, file_name=path.name
                )
        except (OSError, EOFError) as exc:  # not gzip, or truncated
            raise SourceError(f"could not read {path.name} as gzip: {exc}") from None
        payload["file_url"] = url
        return payload
