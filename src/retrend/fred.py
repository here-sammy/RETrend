"""FRED (St. Louis Fed) observations client.

Deterministic data access, no LLM involved. Two details matter for this project:

* ``as_of`` uses FRED's real-time parameters (``realtime_start``/``realtime_end``), which
  return a series *as it was known on that date*. That makes backtests honest even for
  dates before this project existed: a Freddie Mac rate revised later is returned as it
  was first reported.
* The API key travels only in the request URL and is redacted from every error message.
  It is never stored, logged, or put in a report (report sources link to the public
  series page, not the API URL).
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from typing import Callable, NamedTuple

API_URL = "https://api.stlouisfed.org/fred/series/observations"
SEARCH_URL = "https://api.stlouisfed.org/fred/series/search"
ENV_KEY = "FRED_API_KEY"


class FredError(RuntimeError):
    """Any failure talking to FRED. Messages never contain the API key."""


class Observation(NamedTuple):
    day: date
    value: float


HttpGet = Callable[[str, float], "tuple[int, str]"]


def series_page_url(series_id: str) -> str:
    return f"https://fred.stlouisfed.org/series/{series_id}"


def redact(text: str, secret: str | None) -> str:
    return text.replace(secret, "***") if secret else text


def _urllib_get(url: str, timeout: float) -> tuple[int, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "retrend/0.0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:  # FRED reports bad requests as HTTP 400 + JSON
        return exc.code, exc.read().decode("utf-8", "replace")


def _to_float(value: object) -> float | None:
    """FRED marks a missing observation with "." (not documented on the endpoint page,
    so anything that is not a finite number is treated as missing)."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class FredClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        http_get: HttpGet = _urllib_get,
        timeout: float = 30.0,
    ) -> None:
        key = api_key or os.environ.get(ENV_KEY)
        if not key:
            raise FredError(
                f"{ENV_KEY} is not set. Put it in your .env file or environment "
                "(free key: https://fred.stlouisfed.org/docs/api/api_key.html)."
            )
        self._key = key
        self._http_get = http_get
        self._timeout = timeout

    def _get_json(self, base: str, params: dict) -> dict:
        url = f"{base}?{urllib.parse.urlencode(params)}&api_key={urllib.parse.quote(self._key)}"
        try:
            status, body = self._http_get(url, self._timeout)
        except Exception as exc:
            raise FredError(
                f"FRED request failed: {type(exc).__name__}: {redact(str(exc), self._key)}"
            ) from None
        if status != 200:
            message = body[:300]
            try:
                message = json.loads(body).get("error_message", message)
            except (ValueError, AttributeError):
                pass
            raise FredError(f"FRED returned HTTP {status}: {redact(str(message), self._key)}")
        try:
            return json.loads(body)
        except ValueError:
            raise FredError("FRED returned an unexpected response shape") from None

    def search(self, text: str, *, limit: int = 50) -> list[dict]:
        """Series whose title matches ``text``, most popular first. Each row keeps only
        ``id, title, frequency_short, seasonal_adjustment_short, observation_end``."""
        data = self._get_json(SEARCH_URL, {"search_text": text, "file_type": "json", "limit": limit,
                                           "order_by": "popularity", "sort_order": "desc"})
        try:
            return [{k: item.get(k) for k in ("id", "title", "frequency_short",
                                              "seasonal_adjustment_short", "observation_end")}
                    for item in data["seriess"]]
        except (KeyError, TypeError, AttributeError):
            raise FredError("FRED returned an unexpected response shape") from None

    def observations(
        self,
        series_id: str,
        *,
        observation_start: date | None = None,
        observation_end: date | None = None,
        as_of: date | None = None,
    ) -> dict:
        """Fetch a series. Returns ``{"series_id", "as_of", "observations": [{"date",
        "value"}, ...]}`` with ``value`` ``None`` where FRED has no reading.

        With ``as_of`` the request is pinned to that date's vintage and also capped at it.
        """
        params = {"series_id": series_id, "file_type": "json"}
        if observation_start is not None:
            params["observation_start"] = observation_start.isoformat()
        end = observation_end or as_of
        if end is not None:
            params["observation_end"] = end.isoformat()
        if as_of is not None:
            params["realtime_start"] = as_of.isoformat()
            params["realtime_end"] = as_of.isoformat()
        url = f"{API_URL}?{urllib.parse.urlencode(params)}&api_key={urllib.parse.quote(self._key)}"

        try:
            status, body = self._http_get(url, self._timeout)
        except Exception as exc:
            # `from None`: the original exception can embed the request URL (and the key).
            raise FredError(
                f"FRED request failed: {type(exc).__name__}: {redact(str(exc), self._key)}"
            ) from None

        if status != 200:
            message = body[:300]
            try:
                message = json.loads(body).get("error_message", message)
            except (ValueError, AttributeError):
                pass
            raise FredError(f"FRED returned HTTP {status}: {redact(str(message), self._key)}")

        try:
            data = json.loads(body)
            rows = [
                {"date": item["date"], "value": _to_float(item.get("value"))}
                for item in data["observations"]
            ]
        except (ValueError, KeyError, TypeError):
            raise FredError("FRED returned an unexpected response shape") from None

        return {
            "series_id": series_id,
            "as_of": as_of.isoformat() if as_of else None,
            "observations": rows,
        }


def parse_observations(payload: dict) -> list[Observation]:
    """Payload from ``FredClient.observations`` -> readings sorted oldest to newest,
    with missing values dropped."""
    out = [
        Observation(date.fromisoformat(row["date"]), float(row["value"]))
        for row in payload.get("observations", [])
        if row.get("value") is not None
    ]
    out.sort(key=lambda o: o.day)
    return out
