"""Configuration: .env loading and the model registry (config/models.yaml)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def load_dotenv(path: str | Path = ".env", *, override: bool = False) -> list[str]:
    """Load KEY=VALUE lines from ``path`` into ``os.environ``. Returns the names set.

    Handles blank lines, ``#`` comments, an optional ``export`` prefix and quoted values.
    Existing environment variables win unless ``override`` is true. Values are never
    printed or logged. A missing file is fine.
    """
    p = Path(path)
    if not p.is_file():
        return []
    loaded: list[str] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, sep, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if not sep or not name:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if override or name not in os.environ:
            os.environ[name] = value
            loaded.append(name)
    return loaded


@dataclass(frozen=True)
class ModelConfig:
    name: str  # short label used on the command line and in reports
    spec: str  # "provider:model_id", e.g. "groq:llama-3.3-70b-versatile"
    class_path: str | None = None  # optional LangChain chat class to use for this model
    params: dict[str, Any] = field(default_factory=dict)  # passed to the chat class
    parallel: int | None = None  # questions to ask at once for this model (overrides --parallel)
    rpm: float | None = None  # most requests per minute the provider allows (free tiers are low)

    @property
    def provider(self) -> str:
        return self.spec.split(":", 1)[0]

    @property
    def model_id(self) -> str:
        return self.spec.split(":", 1)[1]


RESERVED_NAMES = {"baseline", "stub"}


def load_models(path: str | Path) -> dict[str, ModelConfig]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    models: dict[str, ModelConfig] = {}
    for name, entry in (raw.get("models") or {}).items():
        if name in RESERVED_NAMES:
            raise ValueError(f"model name {name!r} is reserved")
        spec = (entry or {}).get("spec", "")
        provider, sep, model_id = str(spec).partition(":")
        if not sep or not provider or not model_id:
            raise ValueError(f"model {name!r}: spec must look like 'provider:model_id'")
        params = entry.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError(f"model {name!r}: params must be a mapping")
        par = entry.get("parallel")
        if par is not None and (not isinstance(par, int) or isinstance(par, bool) or par < 1):
            raise ValueError(f"model {name!r}: parallel must be a whole number, 1 or more")
        rpm = entry.get("rpm")
        if rpm is not None and (not isinstance(rpm, (int, float)) or isinstance(rpm, bool) or rpm <= 0):
            raise ValueError(f"model {name!r}: rpm must be a number above 0")
        models[name] = ModelConfig(
            name=name, spec=spec, class_path=entry.get("class_path"), params=dict(params),
            parallel=par, rpm=rpm,
        )
    return models
