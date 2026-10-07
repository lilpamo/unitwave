"""Session sets: a named list of sessions chosen on the homepage, for later (step 12).

A set is plain JSON (`<name>.unitwave-set.json` under `data_root/sets/`; sets saved
before the rename end in `.ndset.json`, still open and are never overwritten):
- the eids, sorted;
- the manifest version they were chosen under;
- the session and trial filters used;
- a sha256 over all of that, so an edited file is refused rather than trusted.

Reopening a set warns if the manifest version changed, since the same filters could
now select different sessions. The session view still opens one session at a time.
"""

import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

SET_SUFFIX = ".unitwave-set.json"
OLD_SET_SUFFIXES = (".ndset.json",)  # read, never written
SET_SUFFIXES = (SET_SUFFIX, *OLD_SET_SUFFIXES)
SET_VERSION = 1
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,79}$")


def _hash(content: dict) -> str:
    text = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _content(raw: dict) -> dict:
    return {
        k: raw[k]
        for k in ("version", "name", "eids", "manifest_version", "session_filter", "trial_filter")
    }


def save_set(
    directory: str | os.PathLike,
    name: str,
    eids,
    manifest_version: int,
    session_filter: dict,
    trial_filter: dict,
) -> Path:
    """Write a set atomically; the name is a plain file name (letters, digits, _ . - space)."""
    if not isinstance(name, str) or not _NAME.match(name) or ".." in name:
        raise ValueError(
            "a set name is 1-80 letters, digits, spaces, _ . or -, starting with a letter or digit"
        )
    eids = sorted({str(e) for e in eids})
    if not eids:
        raise ValueError("a set needs at least one session")
    content = {
        "version": SET_VERSION,
        "name": name,
        "eids": eids,
        "manifest_version": int(manifest_version),
        "session_filter": session_filter,
        "trial_filter": trial_filter,
    }
    record = {**content, "hash": _hash(content), "saved_at": datetime.now(UTC).isoformat()}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}{SET_SUFFIX}"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=1))
    tmp.replace(path)
    return path


def read_set(path: str | os.PathLike, manifest_version: int) -> tuple[dict, list[str]]:
    """(set, warnings). Refuses a file whose hash doesn't match its content."""
    raw = json.loads(Path(path).read_text())
    if raw.get("version") != SET_VERSION:
        raise ValueError(f"{Path(path).name}: set version {raw.get('version')} is not supported")
    if _hash(_content(raw)) != raw.get("hash"):
        raise ValueError(f"{Path(path).name}: the hash doesn't match; the file was edited")
    warnings = []
    if raw["manifest_version"] != manifest_version:
        warnings.append(
            f"this set was chosen under manifest version {raw['manifest_version']}, now "
            f"{manifest_version}: the same filters may select different sessions"
        )
    return raw, warnings


def list_sets(directory: str | os.PathLike) -> list[dict]:
    """Saved sets of either ending, newest first: name, file, number of sessions, when
    saved."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    rows = []
    for path in sorted({p for suffix in SET_SUFFIXES for p in directory.glob(f"*{suffix}")}):
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        rows.append(
            {
                "name": raw.get("name", _stem(path.name)),
                "file": path.name,
                "n_sessions": len(raw.get("eids", [])),
                "saved_at": raw.get("saved_at"),
                "mtime": path.stat().st_mtime,
            }
        )
    return sorted(rows, key=lambda r: r["mtime"], reverse=True)


def set_path(directory: str | os.PathLike, name: str) -> Path:
    """A saved set by its name or file name alone; never a path. A bare name prefers
    the new ending."""
    if not isinstance(name, str) or not _NAME.match(name) or ".." in name:
        raise ValueError("open a set by its name")
    candidates = [name] if _stem(name) else [f"{name}{suffix}" for suffix in SET_SUFFIXES]
    for file in candidates:
        if (Path(directory) / file).is_file():
            return Path(directory) / file
    raise ValueError(f"no set named {name!r}")


def _stem(file: str) -> str | None:
    """A set file's name without its ending, for either ending; None otherwise."""
    for suffix in SET_SUFFIXES:
        if file.endswith(suffix) and len(file) > len(suffix):
            return file[: -len(suffix)]
    return None
