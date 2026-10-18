"""Call the FRED MCP server from a worker.

Same call shape as ``FredClient.observations``, so a worker can use either. Each call
starts the server as a subprocess over stdio (about a second of overhead), which is fine
because results are cached in the raw store.

Not usable from inside an already-running event loop (e.g. a Jupyter cell), because it
uses ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import date

from retrend.fred import ENV_KEY, FredError, redact

TOOL_NAME = "fred_series_observations"


class McpFredSource:
    def __init__(self, *, python: str | None = None) -> None:
        self._python = python or sys.executable

    def __call__(
        self,
        series_id: str,
        *,
        observation_start: date | None = None,
        observation_end: date | None = None,
        as_of: date | None = None,
    ) -> dict:
        key = os.environ.get(ENV_KEY)
        if not key:
            raise FredError(f"{ENV_KEY} is not set")
        arguments = {"series_id": series_id}
        for name, value in (
            ("observation_start", observation_start),
            ("observation_end", observation_end),
            ("as_of", as_of),
        ):
            if value is not None:
                arguments[name] = value.isoformat()
        try:
            return asyncio.run(self._call(key, arguments))
        except FredError:
            raise
        except Exception as exc:
            raise FredError(
                f"MCP call failed: {type(exc).__name__}: {redact(str(exc), key)}"
            ) from None

    async def _call(self, key: str, arguments: dict) -> dict:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        # The key goes to the server through its environment, not its command line.
        params = StdioServerParameters(
            command=self._python,
            args=["-m", "retrend.mcp_servers.fred_server"],
            env={ENV_KEY: key},
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(TOOL_NAME, arguments=arguments)

        texts = [c.text for c in result.content if getattr(c, "type", None) == "text"]
        if result.isError:
            raise FredError(redact(" ".join(texts) or "MCP tool reported an error", key))
        if not texts:
            raise FredError("MCP tool returned no text content")
        return json.loads(texts[0])
