"""Structured JSON-lines logging.

Every worker input, output and tool call is logged as one JSON object per line, so any
odd conclusion can be traced back to the signal that caused it.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload, default=str, sort_keys=False)


def get_logger(
    name: str = "retrend", path: str | Path | None = None, level: int = logging.INFO
) -> logging.Logger:
    """Return a logger that writes JSON lines to ``path`` (or stderr if no path)."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = logging.FileHandler(path, encoding="utf-8")
    else:
        handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return logger


def log_event(logger: logging.Logger | None, event: str, **fields: Any) -> None:
    """Log a named event with structured fields. A ``None`` logger is a no-op."""
    if logger is not None:
        logger.info(event, extra={"fields": fields})
