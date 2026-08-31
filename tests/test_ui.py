from __future__ import annotations

import threading
from contextlib import contextmanager
from http.client import HTTPConnection

from memoryd.runtime import MemoryRuntime
from memoryd.ui import MemoryUIHandler, ThreadingHTTPServer


@contextmanager
def ui_server(runtime: MemoryRuntime):
    handler = type("TestUIHandler", (MemoryUIHandler,), {"runtime": runtime})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)


def page(port: int, path: str) -> tuple[int, str, dict[str, str]]:
    connection = HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        return response.status, response.read().decode(), dict(response.getheaders())
    finally:
        connection.close()


def test_ui_inspects_search_state_timeline_and_provenance_safely(tmp_path):
    runtime = MemoryRuntime(tmp_path / "brain.db")
    decision = runtime.remember('SQLite decision <script>alert("bad")</script>', kind="decision")
    state = runtime.remember("Memory Runtime: database = SQLite.", kind="state")
    runtime.link(state.id, decision.id, "depends_on")
    with ui_server(runtime) as port:
        status, body, headers = page(port, "/")
        assert status == 200 and "Recent timeline" in body and decision.id in body
        assert "&lt;script&gt;" in body and "<script>" not in body
        assert "Content-Security-Policy" in headers

        status, body, _ = page(port, "/?q=SQLite")
        assert status == 200 and "Results for" in body and decision.id in body
        status, body, _ = page(port, "/state")
        assert status == 200 and "Memory Runtime" in body and "database" in body
        status, body, _ = page(port, f"/memory/{state.id}")
        assert status == 200 and "Provenance and relationships" in body and "depends_on" in body
        status, _, _ = page(port, "/memory/not-real")
        assert status == 404
