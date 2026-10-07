"""Where units are: brain regions at a chosen level, and positions in Allen CCF space.

Regions come from iblatlas's copy of the Allen CCF 2017 structure ontology:
- the three levels are its Allen, Beryl and Cosmos mappings;
- names, colours and parents come from it too.

At Beryl and Cosmos, fibre tracts and a few nuclei map to `root`. Those units keep
`root` as their region, and region_tree counts them as root's direct units rather
than dropping them. Unit QC keeps using the Allen acronym whatever level is shown.

IBL positions (x = ML right +, y = AP anterior +, z = DV dorsal +, metres from
bregma) convert to CCF µm with iblatlas's bregma landmark. The conversion is
iblatlas AllenAtlas's with no scaling; tests/test_atlas.py checks the two agree.
"""

from functools import lru_cache

import numpy as np
from iblatlas.atlas import ALLEN_CCF_LANDMARKS_MLAPDV_UM
from iblatlas.regions import BrainRegions

LEVELS = ("Allen", "Beryl", "Cosmos")
ROOT = "root"


@lru_cache(maxsize=1)
def _ontology() -> tuple[BrainRegions, dict]:
    """(BrainRegions, acronym -> {id, name, colour, parent}) for the unlateralised ids."""
    br = BrainRegions()
    by_id = {int(i): k for k, i in enumerate(br.id) if i >= 0}
    table = {}
    for i, k in by_id.items():
        parent = br.parent[k]
        parent_k = by_id.get(int(parent)) if np.isfinite(parent) else None
        table[str(br.acronym[k])] = {
            "id": i,
            "name": str(br.name[k]),
            "colour": "#{:02x}{:02x}{:02x}".format(*(int(c) for c in br.rgb[k])),
            "parent": None if parent_k is None else str(br.acronym[parent_k]),
        }
    return br, table


def _check(acronyms) -> np.ndarray:
    """(n_units,) object array of acronyms, with any missing value (None, NaN) as None."""
    acronyms = np.array(
        [a if isinstance(a, str) else None for a in np.asarray(acronyms, dtype=object)],
        dtype=object,
    )
    assert acronyms.ndim == 1, f"acronyms must be (n_units,), got {acronyms.shape}"
    known = _ontology()[1]
    unknown = sorted({a for a in acronyms if a is not None and a not in known})
    if unknown:
        raise ValueError(f"not Allen CCF 2017 acronyms: {unknown}")
    return acronyms


def region_at_level(acronyms, level: str) -> np.ndarray:
    """(n_units,) each unit's region at `level`, from its Allen acronym; None stays None."""
    if level not in LEVELS:
        raise ValueError(f"unknown region level {level!r}; choose one of {list(LEVELS)}")
    acronyms = _check(acronyms)
    out = acronyms.copy()
    present = np.array([a is not None for a in acronyms], dtype=bool)
    if present.any():
        br = _ontology()[0]
        out[present] = br.acronym2acronym(acronyms[present].astype(str), mapping=level)
    return out


def region_info(acronyms) -> dict[str, dict]:
    """acronym -> {id, name, colour ('#rrggbb'), parent} for each distinct acronym."""
    table = _ontology()[1]
    return {a: dict(table[a]) for a in _check(list(acronyms)) if a is not None}


def _ancestry(acronym: str) -> list[str]:
    """[acronym, its parent, ..., root]."""
    table, chain = _ontology()[1], [acronym]
    while table[chain[-1]]["parent"] is not None:
        chain.append(table[chain[-1]]["parent"])
    return chain


def region_tree(regions) -> list[dict]:
    """Every region on a path from root to a unit's region, parents before children.

    regions: (n_units,) acronyms at one level. Each node: acronym, name, colour, parent,
    n_direct (units whose region is this node) and n_total (n_direct plus descendants').
    """
    regions = _check(regions)
    direct, total, depth = {}, {}, {}
    for region in regions:
        if region is None:
            continue
        direct[region] = direct.get(region, 0) + 1
        chain = _ancestry(region)
        for d, node in enumerate(reversed(chain)):
            total[node] = total.get(node, 0) + 1
            depth[node] = d
    info = region_info(list(total))
    order = sorted(total, key=lambda a: (_ancestry(a)[::-1], a))
    return [
        {
            "acronym": a,
            "name": info[a]["name"],
            "colour": info[a]["colour"],
            "parent": info[a]["parent"],
            "depth": depth[a],
            "n_direct": direct.get(a, 0),
            "n_total": total[a],
        }
        for a in order
    ]


def units_in_node(regions, node: str) -> np.ndarray:
    """(n_units,) bool: the unit's region is `node` or one of its descendants."""
    regions = _check(regions)
    _check([node])
    return np.array([r is not None and node in _ancestry(r) for r in regions], dtype=bool)


def ccf_um(xyz: np.ndarray) -> np.ndarray:
    """(n, 3) IBL xyz in metres -> (n, 3) CCF µm in (ap, dv, ml) order, the meshes' order."""
    xyz = np.asarray(xyz, np.float64)
    assert xyz.ndim == 2 and xyz.shape[1] == 3, f"xyz must be (n, 3), got {xyz.shape}"
    ml0, ap0, dv0 = ALLEN_CCF_LANDMARKS_MLAPDV_UM["bregma"]
    ml = ml0 + xyz[:, 0] * 1e6
    ap = ap0 - xyz[:, 1] * 1e6
    dv = dv0 - xyz[:, 2] * 1e6
    return np.column_stack([ap, dv, ml])


def probe_track(sites: np.ndarray) -> np.ndarray:
    """(2, 3) endpoints of the straight line through (n_sites, 3) positions, spanning them."""
    sites = np.asarray(sites, np.float64)
    assert sites.ndim == 2 and sites.shape[1] == 3 and len(sites) >= 2
    centre = sites.mean(axis=0)
    direction = np.linalg.svd(sites - centre)[2][0]  # first principal axis
    t = (sites - centre) @ direction
    return centre + np.outer([t.min(), t.max()], direction)


def depth_runs(depths: np.ndarray, regions) -> list[dict]:
    """Runs of depth-neighbouring units in the same region, shallow-numbered first.

    depths: (n_units,) µm along the probe; regions: (n_units,) acronyms, None skipped.
    Each run spans its own units' depths only: where units were recorded, not where
    region boundaries are.
    """
    depths = np.asarray(depths, np.float64)
    regions = _check(regions)
    assert depths.shape == regions.shape
    keep = np.array([r is not None for r in regions]) & np.isfinite(depths)
    order = np.argsort(depths[keep], kind="stable")
    runs = []
    for d, r in zip(depths[keep][order], regions[keep][order]):
        if runs and runs[-1]["region"] == r:
            runs[-1]["bottom_um"] = float(d)
            runs[-1]["n_units"] += 1
        else:
            runs.append({"region": r, "top_um": float(d), "bottom_um": float(d), "n_units": 1})
    return runs
