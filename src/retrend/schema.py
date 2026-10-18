"""The output contract every worker returns.

The supervisor never reads free text from a worker: it compares these structured
reports, weighting by confidence and freshness.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from retrend.location import Location

Direction = Literal["up", "flat", "down"]
GeographyMatch = Literal["exact", "approximate"]
MetricValue = int | float | str | bool | None


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    url: str | None = None
    retrieved: datetime

    @field_validator("retrieved")
    @classmethod
    def _require_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("retrieved must be timezone-aware")
        return v


class WorkerReport(BaseModel):
    """What every worker returns.

    ``direction`` is the price pressure this worker's signal implies over roughly the next
    6 to 12 months, on its own: "up" means supportive / upward pressure on prices, "down"
    means downward pressure. It is NOT the direction the underlying metric moved: rising
    mortgage rates and rising inventory both map to "down". The raw movement belongs in
    ``key_metrics``.

    ``confidence`` is the worker's own estimate. For LLM-judged workers it is the model's
    self-report and is uncalibrated until the backtest measures it.
    """

    model_config = ConfigDict(extra="forbid")

    worker: str = Field(min_length=1)
    location: Location
    signal: str = Field(min_length=1)
    direction: Direction
    confidence: float = Field(ge=0.0, le=1.0)
    key_metrics: dict[str, MetricValue]
    data_as_of: date
    geography_match: GeographyMatch
    caveats: list[str] = Field(default_factory=list)
    # Short plain-language interpretation for the final narrative. The supervisor weights
    # signals using the structured fields above, never this text.
    summary: str = Field(default="", max_length=1200)
    # At least one source: every conclusion must link back to evidence.
    sources: list[Source] = Field(min_length=1)
    # Which model (or rule) produced direction/confidence, for model comparison.
    judged_by: str | None = None
