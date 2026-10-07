import pandas as pd
import pytest

from unitwave.analysis.catalog import (
    SessionFilter,
    filter_sessions,
    load_catalog,
    region_counts,
    summary,
)
from unitwave.data.manifest import Manifest, manifest_versions


def _manifest() -> Manifest:
    """Three sessions. s1: two probes, CA1 + DG-mo; s2: one probe, PO; s3: one probe, CA1."""
    sessions = pd.DataFrame(
        {
            "eid": ["s1", "s2", "s3"],
            "subject": ["mA", "mB", "mA"],
            "lab": ["labX", "labY", "labX"],
            "date": ["2020-01-01", "2020-06-01", "2021-01-01"],
            "session_number": [1, 1, 1],
            "n_probes": [2, 1, 1],
            "n_good_units": [30, 8, 12],
            "n_trials": [500, 400, 300],
            "n_included_trials": [400, 250, 280],
            "regions": [["CA1", "DG"], ["PO"], ["CA1"]],
            "modalities": [["pose_left", "wheel"], ["wheel"], ["motion_energy_left", "wheel"]],
        }
    )
    insertions = pd.DataFrame(
        {
            "pid": ["p1", "p2", "p3", "p4"],
            "eid": ["s1", "s1", "s2", "s3"],
            "probe_name": ["probe00", "probe01", "probe00", "probe00"],
            # Tip and top of each probe, IBL xyz in metres, as the real manifest has.
            **{f"{end}_{a}": [0.0, 0.0, 0.0, 0.0] for end in ("tip", "top") for a in "xyz"},
        }
    )
    region_units = pd.DataFrame(
        {
            "eid": ["s1", "s1", "s2", "s3"],
            "pid": ["p1", "p2", "p3", "p4"],
            "acronym": ["CA1", "DG-mo", "PO", "CA1"],
            "n_good_units": [20, 10, 8, 12],
        }
    )
    return Manifest(sessions, insertions, manifest_versions(), region_units)


def _eids(frame) -> list[str]:
    return list(frame["eid"])


def test_no_filter_keeps_every_session():
    assert _eids(filter_sessions(_manifest(), SessionFilter())) == ["s1", "s2", "s3"]


@pytest.mark.parametrize(
    ("f", "expected"),
    [
        (SessionFilter(labs=("labX",)), ["s1", "s3"]),
        (SessionFilter(subjects=("mB",)), ["s2"]),
        (SessionFilter(date_from="2020-03-01"), ["s2", "s3"]),
        (SessionFilter(date_to="2020-12-31"), ["s1", "s2"]),
        (SessionFilter(min_good_units=10), ["s1", "s3"]),
        (SessionFilter(min_included_trials=260), ["s1", "s3"]),
        (SessionFilter(n_probes=(2,)), ["s1"]),
        (SessionFilter(modalities=("wheel", "pose_left")), ["s1"]),
    ],
)
def test_each_session_filter(f, expected):
    assert _eids(filter_sessions(_manifest(), f)) == expected


def test_region_filter_counts_descendants_and_applies_the_minimum():
    # HPF contains CA1 and DG (which contains DG-mo): s1 has 20 + 10, s3 has 12, s2 none.
    hpf = filter_sessions(_manifest(), SessionFilter(region="HPF"))
    assert _eids(hpf) == ["s1", "s3"]
    assert hpf["region_units"].tolist() == [30, 12]
    dg = filter_sessions(_manifest(), SessionFilter(region="DG"))
    assert _eids(dg) == ["s1"] and dg["region_units"].tolist() == [10]
    at_least = filter_sessions(_manifest(), SessionFilter(region="HPF", min_region_units=15))
    assert _eids(at_least) == ["s1"]


def test_summary_counts_sessions_probes_and_units():
    m = _manifest()
    counts = summary(m, filter_sessions(m, SessionFilter(labs=("labX",))))
    assert counts == {"n_sessions": 2, "n_probes": 3, "n_units": 42, "n_region_units": None}
    counts = summary(m, filter_sessions(m, SessionFilter(region="HPF")))
    assert counts["n_region_units"] == 42


def test_region_counts_build_the_tree_over_matching_sessions():
    m = _manifest()
    tree = {n["acronym"]: n for n in region_counts(m, ["s1", "s2"], "Beryl")}
    assert tree["root"]["n_total"] == 38  # 20 + 10 + 8
    assert tree["HPF"]["n_total"] == 30 and tree["DG"]["n_total"] == 10
    assert "DG-mo" not in tree  # shown at the Beryl level


def test_load_catalog_builds_once_then_reads_the_written_copy(tmp_path):
    built = []

    def build():
        built.append(1)
        return _manifest()

    first = load_catalog(tmp_path / "derived", build=build)
    second = load_catalog(tmp_path / "derived", build=build)
    assert len(built) == 1
    assert (tmp_path / "derived" / f"manifest-v{manifest_versions()['manifest_version']}").is_dir()
    pd.testing.assert_frame_equal(first.region_units, second.region_units)
