"""Is the running Studio still the code on disk?

The server keeps its Python code in memory from the moment it starts, but serves the
pages and their scripts from disk on every load. After an update (a `git pull`, a new
version, an edit) a server left running would hand new page scripts to old server
code, and the page fails in ways no one can read (2026-10-01: "Cannot read properties
of undefined"). So the server remembers a fingerprint of its code files when it starts.
Once they change, it stops showing pages and asks for a restart, in plain words.
Pages already open keep working until reloaded: their scripts match the running code.
"""

import hashlib
import threading
import time
from datetime import datetime
from html import escape
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]  # the unitwave package
CODE_SUFFIXES = (".py", ".js", ".html", ".css")


def code_fingerprint(root: Path = CODE_ROOT) -> str:
    """sha256 over every code file's path, size and modification time under root
    (compiled caches excluded)."""
    digest = hashlib.sha256()
    files = sorted(
        p
        for p in Path(root).rglob("*")
        if p.suffix in CODE_SUFFIXES and "__pycache__" not in p.parts and p.is_file()
    )
    for path in files:
        st = path.stat()
        digest.update(f"{path.relative_to(root)}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
    return digest.hexdigest()


class Freshness:
    """Whether the code under root changed since this was made, checked at most every
    check_every_s seconds. Once changed it stays changed: changing a file back doesn't
    put back the code a restart would load."""

    def __init__(self, root: Path = CODE_ROOT, check_every_s: float = 1.0):
        self.root = Path(root)
        self.check_every_s = check_every_s
        self.started = code_fingerprint(self.root)
        self.started_at = datetime.now().astimezone()  # local time, for the message
        self._lock = threading.Lock()
        self._checked = time.monotonic()
        self._stale = False

    def stale(self) -> bool:
        with self._lock:
            now = time.monotonic()
            if not self._stale and now - self._checked >= self.check_every_s:
                self._checked = now
                self._stale = code_fingerprint(self.root) != self.started
            return self._stale

    def restart_page(self) -> bytes:
        """What a stale Studio shows instead of a page: plain HTML, no scripts."""
        started = self.started_at.strftime("%H:%M on %d %B %Y")
        return _RESTART.replace("{started}", escape(started)).encode()


_RESTART = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Restart Studio</title>
<style>
  :root { --bg: #f7f6f3; --ink: #0b0b0b; --ink2: #52514e; --card: #ffffff; --line: #dedcd6; }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #151515; --ink: #ecebe8; --ink2: #a9a7a1; --card: #1f1f1f; --line: #353432; }
  }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 560px; margin: 12vh auto; padding: 24px 16px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
          padding: 20px 24px; }
  h1 { font-size: 20px; margin: 0 0 12px; }
  p { margin: 0 0 10px; color: var(--ink2); }
  p.first { color: var(--ink); }
</style>
</head>
<body>
<main><div class="card">
<h1>Studio was updated: restart it</h1>
<p class="first">UnitWave Studio's files changed after this copy started, at {started}.
This page would mix the new files with the old running copy, so it isn't shown.</p>
<p>To restart: stop Studio (Ctrl-C in the window where it runs, or close that window),
then start it again the way you started it, and reload this page.</p>
<p>Saved projects are kept. A session that is open now is not: if you want its view,
save its project from a page that is still open before restarting.</p>
</div></main>
</body>
</html>
"""
