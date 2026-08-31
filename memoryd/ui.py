"""A small, dependency-free browser for inspecting a :class:`MemoryRuntime`.

This deliberately exposes no mutation endpoints.  It is intended for local
inspection of a brain while the REST, CLI, and MCP transports remain the ways
agents change it.
"""
from __future__ import annotations

from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlsplit

from .runtime import MemoryRuntime


def _text(value: object) -> str:
    """Render untrusted stored data as text, never markup."""
    return escape(str(value), quote=True)


def _link(memory_id: str, label: str | None = None) -> str:
    safe_id = quote(memory_id, safe="")
    return f'<a href="/memory/{safe_id}">{_text(label or memory_id)}</a>'


def _page(title: str, body: str) -> bytes:
    # A CSP is useful even though every interpolated field is escaped: memories
    # are user-controlled data and should never become executable UI content.
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_text(title)} · memoryd</title>
<style>body{{font:16px system-ui,sans-serif;line-height:1.45;max-width:960px;margin:2rem auto;padding:0 1rem;color:#17202a}}
a{{color:#0759a8}} code,.meta{{font-family:ui-monospace,monospace;font-size:.9em}} article{{border:1px solid #d7dde3;border-radius:7px;padding:1rem;margin:.7rem 0}}
.meta{{color:#55606b}} .empty{{color:#667}} input{{width:min(35rem,80%);padding:.5rem}} button{{padding:.5rem .8rem}} pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style>
</head><body><nav><a href="/">memoryd inspector</a> · <a href="/state">current state</a></nav><main>{body}</main></body></html>"""
    return document.encode("utf-8")


def _memory_card(memory: dict[str, Any], *, score: float | None = None, reasons: list[str] | None = None) -> str:
    meta = f"{memory.get('kind', 'memory')} · {memory.get('status', 'active')} · importance {_text(memory.get('importance', ''))}"
    if score is not None:
        meta += f" · score {score:.4f}"
    why = f"<div class=\"meta\">{_text(', '.join(reasons or []))}</div>" if reasons else ""
    return f"<article><div class=\"meta\">{_link(str(memory['id']))} · {meta}</div><p>{_text(memory.get('content', ''))}</p>{why}</article>"


class MemoryUIHandler(BaseHTTPRequestHandler):
    """Read-only HTML transport for a supplied runtime."""

    runtime: MemoryRuntime

    def log_message(self, format: str, *args: object) -> None:
        return

    def _html(self, title: str, body: str, status: int = HTTPStatus.OK) -> None:
        payload = _page(title, body)
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'self'")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        params = parse_qs(parsed.query, keep_blank_values=True)
        query = params.get("q", [""])[-1].strip()
        if parsed.path == "/":
            self._dashboard(query)
        elif parsed.path == "/state":
            self._state()
        elif parsed.path.startswith("/memory/"):
            self._detail(unquote(parsed.path[len("/memory/"):]))
        else:
            self._html("Not found", "<h1>Not found</h1>", HTTPStatus.NOT_FOUND)

    def _dashboard(self, query: str) -> None:
        search = f"""<h1>Memory inspector</h1><form method="get" action="/"><label for="q">Search memories</label><br>
<input id="q" name="q" value="{_text(query)}" maxlength="500" autofocus><button type="submit">Search</button></form>"""
        stats = self.runtime.store.stats()
        stat_line = " · ".join(f"{_text(key)}: {_text(value)}" for key, value in sorted(stats.items()))
        if query:
            results = self.runtime.recall(query, limit=20)
            cards = "".join(_memory_card(item.memory.to_dict(), score=item.score, reasons=item.reasons) for item in results)
            section = f"<h2>Results for “{_text(query)}”</h2>{cards or '<p class=empty>No active memories matched.</p>'}"
        else:
            cards = "".join(_memory_card(item) for item in self.runtime.timeline(limit=12))
            section = f"<h2>Recent timeline</h2>{cards or '<p class=empty>No active memories yet.</p>'}"
        self._html("Inspector", f"{search}<p class=meta>{stat_line}</p>{section}")

    def _state(self) -> None:
        facts = self.runtime.state()
        rows = "".join(
            f"<tr><td>{_text(fact['subject'])}</td><td>{_text(fact['state_key'])}</td><td>{_text(fact['value'])}</td><td>{_link(str(fact['memory_id']))}</td></tr>"
            for fact in facts
        )
        body = "<h1>Current state</h1>" + (f"<table><thead><tr><th>Subject</th><th>Key</th><th>Value</th><th>Source</th></tr></thead><tbody>{rows}</tbody></table>" if rows else "<p class=empty>No current state facts.</p>")
        self._html("Current state", body)

    def _detail(self, memory_id: str) -> None:
        memory = self.runtime.get(memory_id)
        if not memory:
            self._html("Not found", "<h1>Memory not found</h1>", HTTPStatus.NOT_FOUND)
            return
        attributes = ("id", "kind", "source", "status", "confidence", "importance", "strength", "created_at", "updated_at", "access_count")
        fields = "".join(f"<dt>{_text(name)}</dt><dd>{_text(memory.get(name, ''))}</dd>" for name in attributes)
        relations = "".join(
            f"<li>{_text(item['relation'])}: {_link(str(item['to_id'] if item['from_id'] == memory_id else item['from_id']), item['content'])} <span class=meta>({_text(item['kind'])})</span></li>"
            for item in memory.get("relationships", [])
        ) or "<li class=empty>No explicit provenance links.</li>"
        entity_related = "".join(
            f"<li>{_text(item['entity'])}: {_link(str(item['id']), item['content'])} <span class=meta>({_text(item['kind'])})</span></li>"
            for item in memory.get("entity_related", [])
        ) or "<li class=empty>No entity-related active memories.</li>"
        body = f"""<h1>Memory detail</h1><article><pre>{_text(memory['content'])}</pre><dl>{fields}</dl></article>
<h2>Provenance and relationships</h2><ul>{relations}</ul><h2>Related by entity</h2><ul>{entity_related}</ul>
<h2>Metadata</h2><pre>{_text(memory.get('metadata', {}))}</pre>"""
        self._html("Memory detail", body)


def serve(runtime: MemoryRuntime, host: str = "127.0.0.1", port: int = 7320) -> None:
    """Serve the inspector. Defaults deliberately restrict it to loopback."""
    handler = type("RuntimeUIHandler", (MemoryUIHandler,), {"runtime": runtime})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"memoryd inspector listening at http://{host}:{port}")
    server.serve_forever()
