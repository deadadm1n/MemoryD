from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .runtime import MemoryRuntime


class MemoryHandler(BaseHTTPRequestHandler):
    runtime: MemoryRuntime

    def log_message(self, format: str, *args: object) -> None:
        return  # daemon consumers should own logging

    def _json(self, body: Any, status: int = 200) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _body(self) -> dict[str, Any]:
        size = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(size) or b"{}")

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        path = parsed.path
        params = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
        if path == "/health": self._json({"status": "ok", "stats": self.runtime.store.stats()}); return
        if path == "/timeline": self._json({"memories": self.runtime.timeline(limit=int(params.get("limit", 50)))}); return
        if path == "/events": self._json({"events": self.runtime.events(limit=int(params.get("limit", 50)))}); return
        if path == "/beliefs": self._json(self.runtime.beliefs()); return
        if path == "/explain":
            try:
                self._json(self.runtime.explain(subject=params.get("subject") or None, key=params.get("key") or None,
                    statement=params.get("statement") or None))
            except (ValueError, KeyError) as exc:
                self._json({"error": str(exc)}, 400)
            return
        if path == "/state": self._json({"state": self.runtime.state(subject=params.get("subject") or None,
            key=params.get("key") or None, history=params.get("history", "false").lower() == "true", at=params.get("at") or None)}); return
        if path.startswith("/memories/"):
            item = self.runtime.get(path.rsplit("/", 1)[-1])
            self._json(item or {"error": "not found"}, 200 if item else 404); return
        self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        try:
            body = self._body()
            if self.path == "/remember":
                memory = self.runtime.remember(**body)
                self._json(memory.to_dict(), HTTPStatus.CREATED); return
            if self.path == "/observe":
                content = body.pop("content")
                self._json(self.runtime.observe(content, **body), HTTPStatus.CREATED); return
            if self.path == "/recall":
                query = body.pop("query")
                self._json({"results": [r.to_dict() for r in self.runtime.recall(query, **body)]}); return
            if self.path == "/context":
                self._json(self.runtime.context(body["query"], budget=int(body.get("budget", 4000)))); return
            if self.path == "/consolidate":
                self._json(self.runtime.consolidate(limit=int(body.get("limit", 200)))); return
            if self.path == "/reflect":
                self._json(self.runtime.reflect(limit=int(body.get("limit", 200)))); return
            if self.path == "/link":
                self.runtime.link(body["from_id"], body["to_id"], body["relation"]); self._json({"status": "linked"}); return
            if self.path.startswith("/forget/"):
                self.runtime.forget(self.path.rsplit("/", 1)[-1]); self._json({"status": "forgotten"}); return
            self._json({"error": "not found"}, 404)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, 400)


def serve(runtime: MemoryRuntime, host: str = "127.0.0.1", port: int = 7319) -> None:
    handler = type("RuntimeHandler", (MemoryHandler,), {"runtime": runtime})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"memoryd listening at http://{host}:{port}")
    server.serve_forever()
