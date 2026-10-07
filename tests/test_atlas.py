import numpy as np
import pytest

from unitwave.analysis.atlas import (
    LEVELS,
    ccf_um,
    depth_runs,
    probe_track,
    region_at_level,
    region_info,
    region_tree,
    units_in_node,
)
from unitwave.data.load import load_data_config

UNITS = np.array(["CA1", "DG-mo", "ml", "VISam5", None], dtype=object)


def test_levels_remap_as_the_allen_hierarchy_says():
    assert LEVELS == ("Allen", "Beryl", "Cosmos")
    assert region_at_level(UNITS, "Allen").tolist() == ["CA1", "DG-mo", "ml", "VISam5", None]
    assert region_at_level(UNITS, "Beryl").tolist() == ["CA1", "DG", "root", "VISam", None]
    assert region_at_level(UNITS, "Cosmos").tolist() == ["HPF", "HPF", "root", "Isocortex", None]
    with pytest.raises(ValueError, match="level"):
        region_at_level(UNITS, "Swanson")
    with pytest.raises(ValueError, match="XYZ"):
        region_at_level(np.array(["XYZ"], dtype=object), "Beryl")


def test_region_info_has_allen_names_and_colours():
    info = region_info(["CA1", "root"])
    assert info["CA1"]["name"] == "Field CA1"
    assert info["CA1"]["colour"] == "#7ed04b"  # Allen's colour_hex_triplet for CA1
    assert info["CA1"]["id"] == 382


def test_region_tree_counts_units_through_the_hierarchy():
    # Beryl level: two CA1, one DG, one in a fibre tract (root).
    tree = region_tree(np.array(["CA1", "CA1", "DG", "root"], dtype=object))
    nodes = {n["acronym"]: n for n in tree}
    assert nodes["root"]["parent"] is None
    assert nodes["root"]["n_total"] == 4 and nodes["root"]["n_direct"] == 1
    assert nodes["CA1"]["n_total"] == nodes["CA1"]["n_direct"] == 2
    assert nodes["HIP"]["n_total"] == 3 and nodes["HIP"]["n_direct"] == 0
    for n in tree:
        children = [c for c in tree if c["parent"] == n["acronym"]]
        assert n["n_total"] == n["n_direct"] + sum(c["n_total"] for c in children)
    # Only regions on the path from root to a unit's region are in the tree.
    assert "Isocortex" not in nodes


def test_units_in_node_includes_descendants():
    regions = np.array(["CA1", "CA1", "DG", "root", None], dtype=object)
    assert units_in_node(regions, "HIP").tolist() == [True, True, True, False, False]
    assert units_in_node(regions, "root").tolist() == [True, True, True, True, False]
    assert units_in_node(regions, "CA1").tolist() == [True, True, False, False, False]


def test_ccf_um_converts_ibl_coordinates_by_hand():
    # Bregma is at CCF (ML 5739, AP 5400, DV 332) µm. IBL x = ML (right +), y = AP
    # (anterior +), z = DV (dorsal +), in metres; CCF AP grows posterior, DV ventral.
    xyz = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.0, 0.001, 0.0], [0.0, 0.0, -0.001]])
    np.testing.assert_allclose(
        ccf_um(xyz),  # (ap, dv, ml), the Allen meshes' vertex order
        [[5400, 332, 5739], [5400, 332, 6739], [4400, 332, 5739], [5400, 1332, 5739]],
        atol=1e-6,
    )


def test_probe_track_spans_its_sites_along_their_line():
    sites = np.array([[0, 0, 0], [0, 0, 10], [0, 0, 30], [0, 0, 20]], float)
    track = probe_track(sites)
    assert track.shape == (2, 3)
    np.testing.assert_allclose(sorted(track[:, 2]), [0, 30], atol=1e-9)
    np.testing.assert_allclose(track[:, :2], 0, atol=1e-9)


def test_ccf_um_matches_iblatlas_allen_atlas():
    from iblatlas.atlas import AllenAtlas

    # mock=True builds iblatlas's coordinate system with empty volumes: no download.
    atlas = AllenAtlas(res_um=25, mock=True)
    # Inside the atlas volume: ML ±4 mm, AP -7..+3 mm, DV -6..0 mm from bregma.
    xyz = np.random.default_rng(0).uniform([-0.004, -0.007, -0.006], [0.004, 0.003, 0], (50, 3))
    np.testing.assert_allclose(ccf_um(xyz), atlas.xyz2ccf(xyz, ccf_order="apdvml"), atol=1e-6)


ATLAS_DIR = load_data_config().data_root / "atlas"


@pytest.mark.skipif(
    not (ATLAS_DIR / "annotation_25.nrrd").exists(), reason="Allen volumes not downloaded"
)
def test_every_d23a44ef_unit_lands_in_its_own_region():
    from iblatlas.atlas import AllenAtlas

    from unitwave.data.load import load_session

    atlas = AllenAtlas(res_um=25, hist_path=ATLAS_DIR / "average_template_25.nrrd")
    units = load_session("d23a44ef-1402-4ed7-97f5-47e9a7a504d9", "bwm").units
    ap, dv, ml = np.round(ccf_um(units[["x", "y", "z"]].to_numpy()) / 25).astype(int).T
    found = atlas.regions.acronym[atlas.label[ap, ml, dv]]  # volumes are (ap, ml, dv)
    assert (found == units["acronym"].to_numpy()).all()


def test_nan_is_missing_like_none():
    regions = np.array(["CA1", np.nan], dtype=object)
    assert region_at_level(regions, "Beryl").tolist() == ["CA1", None]
    assert units_in_node(regions, "root").tolist() == [True, False]


def test_depth_runs_group_neighbouring_units_by_region():
    # Units sorted by depth: 20, 40 in CA1; 60 in DG; 80 in CA1 again; one unit missing.
    depths = np.array([60, 20, 80, 40, 100.0])
    regions = np.array(["DG", "CA1", "CA1", "CA1", None], dtype=object)
    assert depth_runs(depths, regions) == [
        {"region": "CA1", "top_um": 20.0, "bottom_um": 40.0, "n_units": 2},
        {"region": "DG", "top_um": 60.0, "bottom_um": 60.0, "n_units": 1},
        {"region": "CA1", "top_um": 80.0, "bottom_um": 80.0, "n_units": 1},
    ]
