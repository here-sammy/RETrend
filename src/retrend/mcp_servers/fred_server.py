"""MCP server exposing FRED observations, with point-in-time (as-of) support.

Why our own instead of an existing FRED MCP server: the one we looked at is AGPL-licensed,
needs Node, and (per its README) has no real-time/vintage parameters, which backtesting
needs. This one is ~30 lines over ``retrend.fred``.

Run directly (stdio transport):  python -m retrend.mcp_servers.fred_server
Requires FRED_API_KEY in the environment. Needs the ``mcp`` package (1.x line):
    pip install -e ".[mcp]"

Never print to stdout in this module: with the stdio transport, stdout is the protocol.
"""

from datetime import date
from typing import Optional

from retrend.fred import FredClient


def _parse(value: Optional[str]) -> Optional[date]:
    return date.fromisoformat(value) if value else None


def build_server():
    from mcp.server.fastmcp import FastMCP  # mcp 1.x

    server = FastMCP("fred")

    @server.tool()
    def fred_series_observations(
        series_id: str,
        observation_start: Optional[str] = None,
        observation_end: Optional[str] = None,
        as_of: Optional[str] = None,
    ) -> dict:
        """Observations for a FRED series, as {series_id, as_of, observations: [{date, value}]}.

        Dates are YYYY-MM-DD. `value` is null where FRED has no reading. If `as_of` is
        given, returns the series as it was known on that date (point-in-time data) and
        nothing after it.
        """
        return FredClient().observations(
            series_id,
            observation_start=_parse(observation_start),
            observation_end=_parse(observation_end),
            as_of=_parse(as_of),
        )

    return server


def main() -> None:
    build_server().run()  # stdio transport by default


if __name__ == "__main__":
    main()
