"""Local web server for the question UI. Standard library only, bound to 127.0.0.1.

All store and worker access happens on ONE job thread (sqlite connections are thread-bound);
request threads just hand it work and stream back the events it emits.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

from retrend.config import load_models

MAX_BODY = 8 * 1024
MAX_QUERY = 500
STATIC = Path(__file__).parent / "static"
EXAMPLES = [
    {"signal": "Overall outlook", "question": "Will home prices rise in Seattle, WA over the next year?"},
    {"signal": "Home prices and rents", "question": "How have home values and rents changed in Phoenix, AZ?"},
    {"signal": "Supply and demand", "question": "Is Austin, TX turning into a buyer's market?"},
    {"signal": "Mortgage rates", "question": "How are mortgage rates affecting housing in Boston, MA?"},
    {"signal": "School enrollment", "question": "Is school enrollment growing in Denver, CO, and what does it mean for housing?"},
    {"signal": "Local jobs", "question": "How is the job market in Atlanta, GA affecting home prices?"},
    {"signal": "Building permits", "question": "Is new construction picking up in Nashville, TN?"},
    {"signal": "News", "question": "What is the housing news saying about Miami, FL?"},
    {"signal": "ZIP code", "question": "Will home prices rise in ZIP 98101?"},
    {"signal": "Compare cities", "question": "Compare Phoenix, AZ and Denver, CO"},
]


class JobThread(threading.Thread):
    """Owns the runner. Jobs are callables taking the runner."""

    def __init__(self, make_runner: Callable[[], object], close: Callable[[], None] | None = None):
        super().__init__(daemon=True, name="retrend-jobs")
        self._make, self._close = make_runner, close
        self._q: queue.Queue = queue.Queue()
        self.runner = None
        self.ready = threading.Event()
        self.error: Exception | None = None
        self.busy = threading.Lock()

    def run(self):
        try:
            self.runner = self._make()
        except Exception as exc:  # reported on first request
            self.error = exc
        self.ready.set()
        while True:
            job = self._q.get()
            if job is None:
                break
            fn, out = job
            try:
                out.put(("ok", fn(self.runner)))
            except Exception as exc:
                out.put(("err", exc))
        if self._close:
            self._close()

    def call(self, fn, timeout: float = 30.0):
        self.ready.wait(timeout)
        if self.error:
            raise self.error
        out: queue.Queue = queue.Queue()
        self._q.put((fn, out))
        kind, val = out.get(timeout=timeout)
        if kind == "err":
            raise val
        return val

    def stop(self):
        self._q.put(None)


def make_handler(jobs: JobThread, warnings: list[str], allowed_hosts: set[str],
                 model_names: list[str], monthly_dir: Path,
                 model_labels: dict[str, str] | None = None):

    class Handler(BaseHTTPRequestHandler):
        server_version = "retrend-ui"
        protocol_version = "HTTP/1.0"  # close-delimited bodies make streaming simple

        def log_message(self, *a):  # quiet
            pass

        # -- guards -------------------------------------------------------------------
        def _host_ok(self) -> bool:
            if self.headers.get("Host", "") not in allowed_hosts:  # DNS-rebinding guard
                self._send(403, {"error": "bad host"})
                return False
            origin = self.headers.get("Origin")
            if origin and origin not in {f"http://{h}" for h in allowed_hosts}:
                self._send(403, {"error": "bad origin"})
                return False
            return True

        def _send(self, code: int, obj=None, *, body: bytes | None = None, ctype="application/json"):
            data = body if body is not None else json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith("text") else ""))
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'")
            self.end_headers()
            self.wfile.write(data)

        # -- routes -------------------------------------------------------------------
        def do_GET(self):
            if not self._host_ok():
                return
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(200, body=(STATIC / "index.html").read_bytes(), ctype="text/html")
            elif path == "/api/config":
                self._send(200, {"models": model_names, "model_labels": model_labels or {}, "examples": EXAMPLES, "warnings": warnings,
                                 "max_query": MAX_QUERY,
                                 "has_monthly": bool(self._latest_monthly())})
            elif path == "/api/history":
                try:
                    self._send(200, {"items": jobs.call(lambda r: r.history())})
                except Exception as exc:
                    self._send(500, {"error": str(exc)[:200]})
            elif path in ("/api/schools", "/api/prices"):
                from urllib.parse import parse_qs, urlparse
                q = parse_qs(urlparse(self.path).query)
                city, state = (q.get("city") or [""])[0], (q.get("state") or [""])[0]
                zip_code = (q.get("zip") or [""])[0][:10]
                kind = "schools_view" if path == "/api/schools" else "prices_view"
                try:
                    self._send(200, jobs.call(lambda r: getattr(r, kind)(city, state, zip_code), timeout=180))
                except ValueError as exc:
                    self._send(400, {"error": str(exc)[:300]})
                except Exception as exc:  # no match, missing file, download problems: say what happened
                    self._send(404, {"error": f"{type(exc).__name__}: {str(exc)[:400]}"})
            elif path == "/monthly/latest":
                f = self._latest_monthly()
                if f:
                    self._send(200, body=f.read_bytes(), ctype="text/html")
                else:
                    self._send(404, {"error": "no monthly report yet; run 'retrend monthly'"})
            else:
                self._send(404, {"error": "not found"})

        def _latest_monthly(self):
            try:
                found = sorted(monthly_dir.glob("*/report.html"))
                return found[-1] if found else None
            except OSError:
                return None

        def do_POST(self):
            if not self._host_ok():
                return
            if self.path != "/api/ask":
                return self._send(404, {"error": "not found"})
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                return self._send(415, {"error": "send JSON"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._send(400, {"error": "bad length"})
            if n <= 0 or n > MAX_BODY:
                return self._send(413, {"error": "body too large or empty"})
            try:
                body = json.loads(self.rfile.read(n))
                query = str(body["query"]).strip()
                model = str(body.get("model", "baseline"))
                judge_signals = bool(body.get("judge_signals", False))
                writer = str(body.get("writer", "template"))
            except (ValueError, KeyError, TypeError):
                return self._send(400, {"error": "expected {query, model, judge_signals}"})
            if not query or len(query) > MAX_QUERY:
                return self._send(400, {"error": f"question must be 1 to {MAX_QUERY} characters"})
            if model not in model_names or (writer != "template" and (writer not in model_names or writer == "baseline")):
                return self._send(400, {"error": "unknown model"})
            if not jobs.busy.acquire(blocking=False):
                return self._send(429, {"error": "another question is running; wait for it to finish"})
            try:
                self._stream(query, model, judge_signals, writer)
            finally:
                jobs.busy.release()

        def _stream(self, query, model, judge_signals, writer):
            events: queue.Queue = queue.Queue()
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()

            def work(runner):
                try:
                    runner.handle(query, model, judge_signals, events.put, writer)
                except Exception as exc:
                    events.put({"event": "error", "message": f"{type(exc).__name__}: {str(exc)[:300]}"})
                finally:
                    events.put(None)

            threading.Thread(target=lambda: jobs.call(work, timeout=3600), daemon=True).start()
            while True:
                ev = events.get()
                if ev is None:
                    break
                try:
                    self.wfile.write((json.dumps(ev, default=str) + "\n").encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    # the browser went away; let the job finish quietly
                    while events.get() is not None:
                        pass
                    break

    return Handler


def serve(jobs: JobThread, *, port: int, warnings: list[str], monthly_dir: Path,
          config_dir: Path, ready: Callable[[str], None] | None = None) -> ThreadingHTTPServer:
    model_names = ["baseline"]
    labels = {"baseline": "baseline (rules only, instant)"}
    try:
        loaded = load_models(Path(config_dir) / "models.yaml")
        model_names += list(loaded)
        for n, m in loaded.items():
            where = "local" if m.provider == "ollama" else f"hosted, {m.provider}"
            labels[n] = f"{n} ({where}: {m.model_id})"
    except Exception:
        pass
    allowed: set[str] = set()  # filled once the real port is known
    jobs.start()
    httpd = ThreadingHTTPServer(("127.0.0.1", port),
                                make_handler(jobs, warnings, allowed, model_names, monthly_dir, labels))
    httpd.daemon_threads = True
    real = httpd.server_address[1]
    allowed.update({f"127.0.0.1:{real}", f"localhost:{real}"})
    if ready:
        ready(f"http://127.0.0.1:{real}/")
    return httpd
