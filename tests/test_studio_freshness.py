"""A Studio that keeps running after its files change says so: its pages would mix new
page scripts with old server code (2026-10-01, the session page stopped loading)."""

import os
import threading
from http.server import ThreadingHTTPServer

from test_studio_app import _app, _get

from unitwave.studio.freshness import Freshness, code_fingerprint
from unitwave.studio.server import make_handler


def _code(tmp_path):
    root = tmp_path / "code"
    (root / "static").mkdir(parents=True)
    (root / "__pycache__").mkdir()
    (root / "server.py").write_text("x = 1\n")
    (root / "static" / "app.js").write_text("let x = 1;\n")
    (root / "notes.txt").write_text("not code\n")
    (root / "__pycache__" / "server.cpython-311.pyc").write_bytes(b"\0")
    return root


def _touch(path, text):
    """Rewrite a file and move its time on, as an editor or `git pull` would."""
    mtime = path.stat().st_mtime_ns + 1_000_000_000
    path.write_text(text)
    os.utime(path, ns=(mtime, mtime))


def _serve(app, fresh):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app, fresh))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def test_the_fingerprint_follows_code_files_only(tmp_path):
    root = _code(tmp_path)
    before = code_fingerprint(root)
    assert code_fingerprint(root) == before
    _touch(root / "notes.txt", "still not code\n")
    (root / "__pycache__" / "other.pyc").write_bytes(b"\1")
    assert code_fingerprint(root) == before
    _touch(root / "static" / "app.js", "let x = 2;\n")
    assert code_fingerprint(root) != before


def test_once_the_code_changes_studio_stays_stale(tmp_path):
    root = _code(tmp_path)
    fresh = Freshness(root, check_every_s=0.0)
    assert not fresh.stale()
    _touch(root / "server.py", "x = 2\n")
    assert fresh.stale()
    _touch(root / "server.py", "x = 1\n")  # changing it back doesn't undo a restart's need
    assert fresh.stale()


def test_a_stale_studio_asks_for_a_restart_instead_of_showing_pages(tmp_path):
    root = _code(tmp_path)
    fresh = Freshness(root, check_every_s=0.0)
    server, base = _serve(_app(tmp_path), fresh)
    try:
        status, body, _ = _get(base + "/")
        assert status == 200 and "Choose data" in body
        _touch(root / "static" / "app.js", "let x = 2;\n")
        for page in ("/", "/session"):
            status, body, _ = _get(base + page)
            assert status == 503
            assert "Studio was updated: restart it" in body
            assert "Saved projects are kept" in body
        # Pages already open keep working until reloaded: their scripts match the server.
        status, _, _ = _get(base + "/static/app.js")
        assert status == 200
        status, body, _ = _get(base + "/api/session")
        assert status == 400 and "No session is open" in body
    finally:
        server.shutdown()
