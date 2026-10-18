"""Model access: one common path for every model, so comparisons are fair.

Each model gets the identical prompt and is asked for a JSON object, which we parse and
validate ourselves (with a few retries that feed the validation error back). We do NOT
rely on provider-native structured output, because support varies by model and we want
format failures to be visible and comparable, not hidden.

Provider errors (rate limits, timeouts) are retried with exponential backoff. Every
successful judgment is cached in the store's ``llm_calls`` table keyed by (model and
settings, prompt, schema), so re-running a backtest never pays for the same call twice.
``JsonJudge.stats`` counts what happened, which is itself a model-comparison metric.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Protocol, TypeVar

from pydantic import BaseModel

from retrend.config import ModelConfig
from retrend.logging_utils import log_event
from retrend.store import RawStore

T = TypeVar("T", bound=BaseModel)

# Chat classes per provider prefix in models.yaml. Both are the documented LangChain
# integrations. Add more with `class_path` in models.yaml.
PROVIDER_CLASSES = {
    "ollama": "langchain_ollama.ChatOllama",  # local models, no API key
    "google_genai": "langchain_google_genai.ChatGoogleGenerativeAI",
    "groq": "langchain_groq.ChatGroq",
}

DEFAULT_STUB_RESPONSE = json.dumps(
    {
        "direction": "flat",
        "confidence": 0.5,
        "summary": "Stub model: canned response for testing the pipeline, not an analysis.",
        "caveats": ["Stub model output."],
    }
)


class ChatModel(Protocol):
    name: str
    cache_id: str  # identifies the model and its settings in the call cache

    def complete(self, messages: list[tuple[str, str]]) -> str:
        """``messages`` are (role, text) pairs with role system/user/assistant."""


class JudgeError(RuntimeError):
    def __init__(self, message: str, *, attempts: int, last_raw: str) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_raw = last_raw


@dataclass(frozen=True)
class JudgeResult:
    value: BaseModel
    raw: str
    attempts: int
    cached: bool
    model: str


# -- text helpers ----------------------------------------------------------------


def message_text(msg: Any) -> str:
    """Plain text of a LangChain message, tolerant of version differences: ``.text`` may
    be a property or a method, and ``.content`` may be a string or a list of blocks
    (reasoning blocks are skipped)."""
    if isinstance(msg, str):
        return msg
    text = getattr(msg, "text", None)
    if callable(text):
        try:
            text = text()
        except TypeError:
            text = None
    if isinstance(text, str) and text:
        return text
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type", "text") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


def strip_reasoning(text: str) -> str:
    """Drop <think>...</think> reasoning that some models (e.g. Qwen) put in the reply."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    if "<think>" in text.lower():
        return ""  # reasoning that never closed: there is no answer in this reply
    return text


def extract_json_object(text: str) -> dict:
    """First JSON object in ``text``, tolerating code fences and surrounding prose."""
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("no JSON object found in the reply")


# -- chat implementations ------------------------------------------------------------


BREAKER_FAILURES = 6  # this many failed questions with no success at all = something is wrong, stop


class PrefetchAborted(RuntimeError):
    """Every question asked so far failed: wrong model id, bad key, no quota... not worth continuing."""


RATE_LIMIT_WORDS = re.compile(r"429|rate.?limit|quota|resource.?exhausted|too many requests", re.I)
RATE_LIMIT_PAUSE = 20.0
SPEED_ONLY_PARAMS = {"keep_alive", "base_url", "num_thread", "timeout", "client_kwargs"}


class LangChainChat:
    """A chat model built from models.yaml via its LangChain integration class."""

    def __init__(self, config: ModelConfig) -> None:
        path = config.class_path or PROVIDER_CLASSES.get(config.provider)
        if path is None:
            raise ValueError(
                f"unknown provider {config.provider!r} for model {config.name!r}; "
                f"known: {sorted(PROVIDER_CLASSES)}. Set class_path in models.yaml for others."
            )
        module_name, _, class_name = path.rpartition(".")
        try:
            cls = getattr(importlib.import_module(module_name), class_name)
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                f"cannot load {path} for model {config.name!r}. Install the provider "
                f'package, e.g. pip install -e ".[llm]". ({exc})'
            ) from exc
        self.name = config.name
        # speed and plumbing settings do not change the answers, so they must not change the cache key
        answer_params = {k: v for k, v in config.params.items() if k not in SPEED_ONLY_PARAMS}
        self.cache_id = json.dumps(
            {"spec": config.spec, "class": path, "params": answer_params},
            sort_keys=True,
            default=str,
        )
        self._model = cls(model=config.model_id, **config.params)

    def complete(self, messages: list[tuple[str, str]]) -> str:
        return message_text(self._model.invoke(messages))


class StubChat:
    """Canned responses for tests and for running the pipeline without any API keys."""

    def __init__(self, responses: list[Any] | None = None, *, name: str = "stub") -> None:
        self.name = name
        self.cache_id = f"stub:{name}"
        self._responses = list(responses) if responses else [DEFAULT_STUB_RESPONSE]
        self.calls: list[list[tuple[str, str]]] = []

    def complete(self, messages: list[tuple[str, str]]) -> str:
        self.calls.append(list(messages))
        response = self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]
        if isinstance(response, Exception):
            raise response
        return response


# -- the judge ---------------------------------------------------------------------------


class JsonJudge:
    """Ask a chat model for a JSON object matching a pydantic schema."""

    def __init__(
        self,
        chat: ChatModel,
        *,
        store: RawStore | None = None,
        logger: logging.Logger | None = None,
        max_attempts: int = 3,
        base_delay: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.chat = chat
        self.store = store
        self.logger = logger
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self._sleep = sleep
        self._lock = threading.Lock()  # stats are shared by the threads of a prefetch
        self.rpm: float | None = None  # throttle: at most this many requests a minute, across all threads
        self._next_slot = 0.0
        self.parallel: int | None = None  # per-model limit from models.yaml, if any
        self.collect: list | None = None  # when a list, cache misses are recorded instead of asked
        self.stats = {
            "calls": 0,  # requests sent to the model
            "successes": 0,  # judgments returned (including cache hits)
            "failures": 0,  # judgments that exhausted every attempt
            "parse_failures": 0,  # replies that were not valid JSON for the schema
            "provider_errors": 0,  # exceptions from the provider (rate limit, timeout, auth)
            "cache_hits": 0,
        }

    @property
    def model_name(self) -> str:
        return self.chat.name

    def judge(self, system: str, user: str, schema: type[T]) -> JudgeResult:
        started = time.perf_counter()

        model_key, prompt_key = self._keys(system, user, schema)

        data = None
        cached = False
        if self.store is not None:
            stored = self.store.get_llm_response(model_key, prompt_key)
            if stored is not None:
                try:
                    data = json.loads(stored)
                    schema.model_validate(data["parsed"])
                    cached = True
                    self.stats["cache_hits"] += 1
                except (ValueError, KeyError, TypeError):
                    data = None  # unreadable or stale entry: treat as a miss
        if data is None:
            if self.collect is not None:  # planning pass: note the question, answer nothing
                self.collect.append((system, user, schema))
                raise JudgeError("collecting prompts", attempts=0, last_raw="")
            try:
                data = self._attempt_loop(system, user, schema)
            except JudgeError:
                self.stats["failures"] += 1
                raise
            if self.store is not None:
                self.store.put_llm_response(model_key, prompt_key, json.dumps(data))

        self.stats["successes"] += 1
        result = JudgeResult(
            value=schema.model_validate(data["parsed"]),
            raw=data["raw"],
            attempts=data["attempts"],
            cached=cached,
            model=self.chat.name,
        )
        log_event(
            self.logger,
            "judge_done",
            model=self.chat.name,
            attempts=result.attempts,
            cached=cached,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return result

    def _keys(self, system: str, user: str, schema: type[T]) -> tuple[str, str]:
        # The key covers the model + settings, the schema definition and the full prompt,
        # so changing any of them is a cache miss.
        model_key = f"{self.chat.name}|{hashlib.sha256(self.chat.cache_id.encode()).hexdigest()[:12]}"
        prompt_key = "\n".join([json.dumps(schema.model_json_schema(), sort_keys=True), system, user])
        return model_key, prompt_key

    def _wait_for_slot(self) -> None:
        if not self.rpm:
            return
        gap = 60.0 / self.rpm
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot)
            self._next_slot = slot + gap
        if slot > now:
            self._sleep(slot - now)

    def _bump(self, key: str) -> None:
        with self._lock:
            self.stats[key] += 1

    def uncached(self, requests: list) -> dict:
        """The distinct requests with no cached reply yet, keyed by cache key."""
        todo: dict[tuple[str, str], tuple] = {}
        for system, user, schema in requests:
            key = self._keys(system, user, schema)
            if key not in todo and (self.store is None or self.store.get_llm_response(*key) is None):
                todo[key] = (system, user, schema)
        return todo

    def prefetch(self, requests: list, *, workers: int = 4,
                 progress: Callable[[str], None] | None = None) -> dict:
        """Answer many (system, user, schema) requests at once and cache the replies.

        Only the provider calls run in threads; the cache (SQLite) is touched from this thread
        alone. A request that fails is left uncached, so the normal run retries it and falls back
        to the rule as usual. Several judges at once: see eval.prefetch.warm_many."""
        todo = self.uncached(requests)
        if not todo:
            return {"requested": 0, "done": 0, "failed": 0}
        done = failed = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(self._attempt_loop, *args): key for key, args in todo.items()}
            for fut in as_completed(futures):
                try:
                    data = fut.result()
                except Exception as exc:  # JudgeError after every attempt, or anything unexpected
                    failed += 1
                    if done == 0 and failed >= BREAKER_FAILURES:
                        for f in futures:
                            f.cancel()
                        raise PrefetchAborted(
                            f"{self.chat.name}: the first {failed} questions all failed, so the run "
                            f"was stopped. Last error: {str(exc)[:300]}") from exc
                else:
                    done += 1
                    if self.store is not None:
                        self.store.put_llm_response(*futures[fut], json.dumps(data))
                if progress and (done + failed) % 25 == 0:
                    progress(f"asked {done + failed}/{len(todo)} model questions: {done} answered, {failed} failed")
        return {"requested": len(todo), "done": done, "failed": failed}

    def _attempt_loop(self, system: str, user: str, schema: type[T]) -> dict:
        messages: list[tuple[str, str]] = [("system", system), ("user", user)]
        last_raw = ""
        last_problem = ""
        for attempt in range(1, self.max_attempts + 1):
            self._bump("calls")
            self._wait_for_slot()
            try:
                raw = self.chat.complete(messages)
            except Exception as exc:  # provider errors: rate limits, timeouts, auth, ...
                self._bump("provider_errors")
                last_problem = f"{type(exc).__name__}: {' '.join(str(exc).split())[:200]}"
                log_event(self.logger, "judge_provider_error", model=self.chat.name,
                          attempt=attempt, error=last_problem)
                if attempt < self.max_attempts:  # no pointless sleep after the last attempt
                    wait = self.base_delay * 2 ** (attempt - 1)
                    if RATE_LIMIT_WORDS.search(last_problem):  # a rate limit needs a real pause, not 2 s
                        wait = max(wait, RATE_LIMIT_PAUSE * attempt)
                    self._sleep(wait)
                continue

            last_raw = raw
            answer = strip_reasoning(raw)
            try:
                value = schema.model_validate(extract_json_object(answer))
            except ValueError as exc:  # JSON errors and pydantic ValidationError
                self._bump("parse_failures")
                last_problem = " ".join(str(exc).split())[:400]
                log_event(self.logger, "judge_retry", model=self.chat.name, attempt=attempt,
                          error=last_problem)
                messages = messages + [
                    ("assistant", answer[:1500] or "(empty reply)"),
                    (
                        "user",
                        f"That reply was not valid: {last_problem}\n"
                        "Reply again with ONLY the JSON object and nothing else.",
                    ),
                ]
                continue
            return {"raw": raw, "parsed": value.model_dump(mode="json"), "attempts": attempt}
        raise JudgeError(
            f"{self.chat.name}: no valid JSON after {self.max_attempts} attempts "
            f"(last problem: {last_problem})",
            attempts=self.max_attempts,
            last_raw=last_raw,
        )
