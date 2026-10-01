"""Allen CCF 2017 structure meshes, downloaded once per structure and cached.

Each structure's surface is one OBJ file on the Allen Institute's download server,
named by its Allen structure id (997 is the whole brain). Vertices are CCF µm in
(ap, dv, ml) order. Files are cached in `<root>/ccf_2017_meshes/`, where root is
`data_root/atlas`, never the repo. A download is written to a temporary file and
renamed into place, so an interrupted one never looks cached.
"""

import os
import urllib.request
from collections.abc import Callable
from http.client import HTTPException
from pathlib import Path

MESH_URL = (
    "https://download.alleninstitute.org/informatics-archive/current-release/"
    "mouse_ccf/annotation/ccf_2017/structure_meshes/{}.obj"
)
_TIMEOUT_S = 60
_ATTEMPTS = 2


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=_TIMEOUT_S) as response:
        return response.read()


def mesh_path(
    structure_id: int, root: str | os.PathLike, fetch: Callable[[str], bytes] = _fetch
) -> Path:
    """The cached OBJ file for an Allen structure id, downloading it the first time."""
    if isinstance(structure_id, bool) or not isinstance(structure_id, int) or structure_id < 0:
        raise ValueError(f"not an Allen structure id: {structure_id!r}")
    path = Path(root) / "ccf_2017_meshes" / f"{structure_id}.obj"
    if path.exists():
        return path
    # The Allen server sometimes stalls or cuts a download short (S1: 2 of ~20 on one
    # day), so a failed download is tried once more before giving up.
    for attempt in range(_ATTEMPTS):
        try:
            data = fetch(MESH_URL.format(structure_id))
            break
        except (OSError, HTTPException) as e:  # timeouts, refusals, HTTP errors, cut short
            if attempt == _ATTEMPTS - 1:
                raise OSError(
                    f"could not download the mesh for structure {structure_id} from the "
                    f"Allen Institute ({type(e).__name__}: {e}); it is tried again next time"
                ) from e
    head = data[:4096].decode("ascii", errors="replace")
    if not any(line.startswith(("v ", "#", "o ")) for line in head.splitlines()):
        raise ValueError(f"the download for structure {structure_id} is not an OBJ mesh")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".obj.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return path
