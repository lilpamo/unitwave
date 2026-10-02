"""Homepage entry points besides the BWM catalog: Phy folders and recent projects.

**Phy folders** are opened only from under a configured root (configs/catalog.yaml,
`phy_root`). A path is resolved with every symlink followed and must stay inside the
root, so `../` and symlinks pointing out are refused alike. A folder pairs with the
`events.csv` beside its `params.py`. Missing files get plain-language refusals.

**Recent projects** are the `*.unitwave.json` files in `data_root/projects` (and
`*.ndstudio.json`, from before the rename), newest
first. They are opened by name only, never by a path.
"""

import json
import os
from pathlib import Path

from unitwave.studio.project import SUFFIXES, project_stem

EVENTS_NAME = "events.csv"
_MAX_CHOICES = 200


def _inside(root: Path, path: Path) -> Path:
    """path resolved (symlinks followed); refused unless it stays inside root."""
    real_root = root.resolve()
    real = path.resolve()
    if real != real_root and real_root not in real.parents:
        raise ValueError(f"{path} is outside the Phy folder root {root}")
    return real


def resolve_phy_folder(root: str | os.PathLike, user_path: str) -> tuple[Path, Path]:
    """(folder, events CSV) for a path under the root, or a plain-language refusal."""
    root = Path(root).expanduser()
    if not root.is_dir():
        raise ValueError(f"the Phy folder root {root} does not exist; set phy_root in configs")
    candidate = Path(user_path).expanduser()
    folder = _inside(root, candidate if candidate.is_absolute() else root / candidate)
    if not folder.exists():
        raise ValueError(f"{user_path} does not exist under {root}")
    if not folder.is_dir():
        raise ValueError(f"{user_path} is a file, not a folder")
    if not (folder / "spike_times.npy").exists():
        raise ValueError(f"{user_path} has no spike_times.npy: not a Kilosort/Phy output folder")
    if not (folder / "params.py").exists():
        raise ValueError(f"{user_path} has no params.py, so its sampling rate is unknown")
    events = folder / EVENTS_NAME
    if not events.exists():
        raise ValueError(
            f"{user_path} has no {EVENTS_NAME} beside params.py; Studio needs the trial events"
        )
    return folder, _inside(root, events)


def complete_phy_path(root: str | os.PathLike, prefix: str) -> list[dict]:
    """Folders under the root that continue `prefix` ("mouse1/" lists mouse1's folders).

    Each: {path relative to the root, phy: has spike_times.npy, events: has events.csv}.
    Symlinks leading out of the root are never offered.
    """
    root = Path(root).expanduser()
    if not root.is_dir():
        return []
    parent_text, _, stem = prefix.rpartition("/")
    parent = _inside(root, root / parent_text) if parent_text else root.resolve()
    if not parent.is_dir():
        return []
    out = []
    for child in sorted(parent.iterdir()):
        if not child.name.startswith(stem) or child.name.startswith("."):
            continue
        try:
            real = _inside(root, child)
        except ValueError:
            continue  # a symlink out of the root
        if not real.is_dir():
            continue
        rel = f"{parent_text}/{child.name}" if parent_text else child.name
        out.append(
            {
                "path": rel,
                "phy": (real / "spike_times.npy").exists(),
                "events": (real / EVENTS_NAME).exists(),
            }
        )
        if len(out) >= _MAX_CHOICES:
            break
    return out


def recent_projects(directory: str | os.PathLike) -> list[dict]:
    """Project files of either ending, newest first: name, file, its data source, when
    it was last saved."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    rows = []
    for path in sorted({p for suffix in SUFFIXES for p in directory.glob(f"*{suffix}")}):
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        source = {k: v for k, v in raw.get("source", {}).items() if v is not None}
        source.pop("release", None)
        source.pop("task_sha256", None)
        rows.append(
            {
                "name": project_stem(path),
                "file": path.name,
                "source": source,
                "mtime": path.stat().st_mtime,
            }
        )
    return sorted(rows, key=lambda r: r["mtime"], reverse=True)


def project_path(directory: str | os.PathLike, name: str) -> Path:
    """A project in the directory by its name or its file name alone; never a path. A
    bare name prefers the new ending."""
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise ValueError("open a recent project by its name")
    candidates = [name] if project_stem(name) else [f"{name}{suffix}" for suffix in SUFFIXES]
    for file in candidates:
        if (Path(directory) / file).is_file():
            return Path(directory) / file
    raise ValueError(f"no project named {name!r}")
