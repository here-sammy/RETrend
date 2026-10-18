"""The supervisor's output: a directional outlook with its evidence."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from retrend.location import Location
from retrend.schema import Direction, Source


class Contribution(BaseModel):
    """How much one worker's report moved the outlook."""

    model_config = ConfigDict(extra="forbid")

    worker: str
    direction: Direction
    reported_confidence: float
    effective_confidence: float  # after freshness / geography adjustments
    base_weight: float
    freshness_factor: float
    geography_factor: float
    contribution: float  # signed share of the final score; positive pushes prices up
    data_as_of: date
    summary: str
    sources: list[Source]


class Finding(BaseModel):
    """One thing the critic noticed."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["missing_worker", "stale_data", "approximate_geography", "conflict",
                  "rule_only", "not_point_in_time", "retried"]
    worker: str | None = None
    detail: str


class Outlook(BaseModel):
    model_config = ConfigDict(extra="forbid")

    location: Location
    horizon: str = "next 6 to 12 months"
    direction: Direction
    score: float = Field(ge=-1.0, le=1.0)  # -1 = strongly down, +1 = strongly up
    confidence: float = Field(ge=0.0, le=1.0)  # uncalibrated; see README
    uncertainty: str
    drivers: list[Contribution]  # agree with the outlook (or all, when flat), largest first
    dissent: list[Contribution]  # point the other way
    context: list[Contribution] = []  # zero-weight workers: shown, but they do not vote
    missing_workers: dict[str, str]  # worker -> why it is missing
    findings: list[Finding]
    narrative: str
    narrated_by: str | None = None
    weights: dict[str, float]
    as_of: date | None = None
    generated_at: datetime
