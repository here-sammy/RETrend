"""ZIP code -> city and state.

    retrend zip-fetch      # download the free GeoNames US postal-code list and save data/files/zip/us_zip.csv

or save your own CSV in data/files/zip/ (or set ZIP_FILE=C:\\path\\to\\file.csv) with columns found by NAME:
  zip     (zip, zipcode, zip_code, postal_code, postcode)
  city    (city, primary_city, place, place_name, po_name)
  state   (state, state_id, st, state_code, admin_code1)

A ZIP maps to the city of its postal place name. ZIPs can cross city limits and some place names are
neighbourhoods or post offices, so the answer uses that city and says so.

``zip-fetch`` is written from memory of the GeoNames file layout (tab separated: country, postal code,
place name, state name, state code, county name, county code, ...) and was NOT run against the live
site (the build environment cannot reach it). If the layout differs, it stops and says what it saw.
"""

from __future__ import annotations

import csv
import io
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from retrend.sources.files import SourceError

GEONAMES_URL = "https://download.geonames.org/export/zip/US.zip"
ALIASES = {
    "zip": ("zip", "zipcode", "zip_code", "postal_code", "postcode", "zcta"),
    "city": ("city", "primary_city", "place", "place_name", "po_name", "usps_city"),
    "state": ("state", "state_id", "st", "state_code", "admin_code1", "state_abbr"),
}
HEADER = ["zip", "city", "state"]
ZIP_RE = re.compile(r"\b(\d{5})(?:-\d{4})?\b")


class ZipError(SourceError):
    pass


class ZipNotFound(ZipError):
    pass


@dataclass(frozen=True)
class ZipPlace:
    zip: str
    city: str
    state: str


def _key(h: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", h.strip().lower()).strip("_")


def _pick(headers: dict[str, str], field: str) -> str | None:
    for alias in ALIASES[field]:
        if alias in headers:
            return headers[alias]
    return None


def _read_csv(path: Path, table: dict[str, ZipPlace]) -> None:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = {_key(h): h for h in (reader.fieldnames or [])}
        cols = {k: _pick(headers, k) for k in ALIASES}
        missing = [k for k, v in cols.items() if v is None]
        if missing:
            raise ZipError(f"{path.name}: no column for {missing}. Columns found: {list(headers.values())}. "
                           "Rename them or run 'retrend zip-fetch'.")
        for row in reader:
            z = re.sub(r"\D", "", (row.get(cols["zip"]) or ""))[:5].zfill(5)
            city = " ".join((row.get(cols["city"]) or "").split())
            state = (row.get(cols["state"]) or "").strip().upper()
            if len(z) == 5 and city and re.fullmatch(r"[A-Z]{2}", state):
                table.setdefault(z, ZipPlace(z, city, state))


class ZipResolver:
    def __init__(self, folder: str | Path, env: dict | None = None) -> None:
        self._folder = Path(folder)
        self._env = os.environ if env is None else env
        self._table: dict[str, ZipPlace] | None = None

    def _files(self) -> list[Path]:
        explicit = self._env.get("ZIP_FILE")
        if explicit:
            return [Path(explicit)]
        return sorted(self._folder.glob("*.csv")) if self._folder.exists() else []

    def available(self) -> bool:
        return any(p.exists() for p in self._files())

    def _load(self) -> dict[str, ZipPlace]:
        if self._table is None:
            files = [p for p in self._files() if p.exists()]
            if not files:
                raise ZipError("No ZIP file found. Run 'retrend zip-fetch', or save a CSV with zip, city and "
                               f"state columns in {self._folder} (or set ZIP_FILE).")
            table: dict[str, ZipPlace] = {}
            for p in files:
                _read_csv(p, table)
            self._table = table
        return self._table

    def resolve(self, zip_code: str) -> ZipPlace:
        m = ZIP_RE.fullmatch(str(zip_code).strip())
        if not m:
            raise ZipError("a ZIP code is five digits, such as 98101")
        place = self._load().get(m.group(1))
        if place is None:
            raise ZipNotFound(f"ZIP {m.group(1)} is not in the ZIP file")
        return place

    def rewrite(self, text: str) -> tuple[str, list[ZipPlace], list[str]]:
        """Replace every known ZIP in ``text`` with 'City, ST'. -> (new text, places, unresolved ZIPs).
        If no ZIP file is installed the text is returned unchanged and the ZIPs are all unresolved."""
        zips = ZIP_RE.findall(text)
        if not zips:
            return text, [], []
        if not self.available():
            return text, [], list(dict.fromkeys(zips))
        places: list[ZipPlace] = []
        bad: list[str] = []

        def sub(m: re.Match) -> str:
            try:
                p = self.resolve(m.group(1))
            except ZipNotFound:
                bad.append(m.group(1))
                return m.group(0)
            places.append(p)
            return f"{p.city}, {p.state}"

        return ZIP_RE.sub(sub, text), places, list(dict.fromkeys(bad))


def parse_geonames(raw: bytes) -> list[tuple[str, str, str]]:
    """US.zip bytes -> sorted (zip, city, state) rows."""
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            name = next((n for n in z.namelist() if n.upper() == "US.TXT"), None)
            if name is None:
                raise ZipError(f"US.txt not found in the download; it contains {z.namelist()[:5]}")
            text = z.read(name).decode("utf-8")
    except zipfile.BadZipFile:
        raise ZipError("the download is not a zip file; got: " + raw[:80].decode("utf-8", "replace")) from None
    rows: dict[str, tuple[str, str, str]] = {}
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        zip5, city, state = parts[1].strip().zfill(5), parts[2].strip(), parts[4].strip().upper()
        if re.fullmatch(r"\d{5}", zip5) and city and re.fullmatch(r"[A-Z]{2}", state):
            rows.setdefault(zip5, (zip5, city, state))
    if len(rows) < 1000:
        first = text.splitlines()[:1]
        raise ZipError(f"only {len(rows)} ZIPs parsed; expected tens of thousands. First line seen: {first}")
    return sorted(rows.values())


def fetch_zip_csv(out_csv: str | Path, *, open_url: Callable | None = None,
                  say: Callable[[str], None] = print) -> int:
    say(f"Downloading {GEONAMES_URL}")
    try:
        if open_url is None:
            import urllib.request
            req = urllib.request.Request(GEONAMES_URL, headers={"User-Agent": "retrend/0.1 (zip table)"})
            with urllib.request.urlopen(req, timeout=120) as r:
                raw = r.read()
        else:
            raw = open_url(GEONAMES_URL).read()
    except OSError as exc:
        raise ZipError(f"download failed: {type(exc).__name__}: {exc}") from None
    rows = parse_geonames(raw)
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    return len(rows)
