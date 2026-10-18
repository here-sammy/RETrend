"""Canonical location key shared by every worker.

"City" is ambiguous: school districts, police jurisdictions, ZIP codes and metros do
not line up. Every worker accepts a ``Location`` and maps it to its own source's
geography, reporting how close that match is (see ``WorkerReport.geography_match``).
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

_FIVE_DIGITS = re.compile(r"\d{5}")


class Location(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    city: str
    state: str  # two-letter USPS code
    county_fips: str | None = None  # 5-digit state+county FIPS
    cbsa: str | None = None  # 5-digit metro/micro area code
    zips: tuple[str, ...] = ()

    @field_validator("city")
    @classmethod
    def _check_city(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("city must not be empty")
        return v

    @field_validator("state")
    @classmethod
    def _check_state(cls, v: str) -> str:
        v = v.strip().upper()
        if not re.fullmatch(r"[A-Z]{2}", v):
            raise ValueError("state must be a two-letter USPS code")
        return v

    @field_validator("county_fips", "cbsa")
    @classmethod
    def _check_code(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not _FIVE_DIGITS.fullmatch(v):
            raise ValueError("must be a 5-digit code (keep leading zeros, quote it in YAML)")
        return v

    @field_validator("zips")
    @classmethod
    def _check_zips(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        for z in v:
            if not _FIVE_DIGITS.fullmatch(z):
                raise ValueError(f"invalid ZIP code: {z!r}")
        return v

    @property
    def key(self) -> str:
        """Stable slug used as the cache and log key, e.g. ``seattle-wa``."""
        slug = re.sub(r"[^a-z0-9]+", "-", self.city.lower()).strip("-")
        return f"{slug}-{self.state.lower()}"


def load_registry(path: str | Path) -> dict[str, Location]:
    """Load a YAML file with a top-level ``cities`` list into ``{key: Location}``."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cities = (raw or {}).get("cities", [])
    registry: dict[str, Location] = {}
    for entry in cities:
        loc = Location.model_validate(entry)
        if loc.key in registry:
            raise ValueError(f"duplicate location key: {loc.key}")
        registry[loc.key] = loc
    return registry
