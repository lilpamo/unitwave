"""Homepage part (b): probe lines, session sets, Phy folders under a root, recent projects."""

import json
import os
import time

import numpy as np
import pandas as pd
import pytest
from phy_folder import write_phy_folder
from test_catalog import _manifest

from unitwave.analysis.atlas import ccf_um
from unitwave.analysis.catalog import probe_lines
from unitwave.data.manifest import manifest_versions
from unitwave.studio.entry import (
    complete_phy_path,
    project_path,
    recent_projects,
    resolve_phy_folder,
)
from unitwave.studio.sets import SET_SUFFIX, list_sets, read_set, save_set, set_path

EVENTS = pd.DataFrame({"intervals_0": [0.5], "intervals_1": [2.9], "stimOn_times": [0.6]})


def _with_ends(m):
    tips = {
        "p1": (0.001, -0.002, -0.003),
        "p2": (0.0, 0.0, -0.001),
        "p3": (-0.002, -0.004, -0.0035),
        "p4": (0.0005, -0.001, -0.002),
    }
    ins = m.insertions.copy()
    for end, shift in (("tip", 0.0), ("top", 0.002)):
        for i, axis in enumerate("xyz"):
            ins[f"{end}_{axis}"] = [
                tips[p][i] + (shift if axis == "z" else 0.0) for p in ins["pid"]
            ]
    return m.__class__(m.sessions, ins, m.provenance, m.region_units)


def test_probe_lines_use_the_same_conversion_as_unit_positions():
    m = _with_ends(_manifest())
    lines = probe_lines(m, ["s1", "s3"])
    assert list(lines["pid"]) == ["p1", "p2", "p4"]
    ins = m.insertions.set_index("pid").loc[list(lines["pid"])]
    np.testing.assert_array_equal(
        np.vstack(lines["tip"]), ccf_um(ins[["tip_x", "tip_y", "tip_z"]].to_numpy())
    )
    np.testing.assert_array_equal(
        np.vstack(lines["top"]), ccf_um(ins[["top_x", "top_y", "top_z"]].to_numpy())
    )
    assert set(lines.columns) >= {"eid", "probe_name", "lab", "subject", "date"}


# ---------- session sets ----------


def test_a_session_set_round_trips(tmp_path):
    path = save_set(
        tmp_path,
        "hippocampus",
        ["s3", "s1"],
        manifest_versions()["manifest_version"],
        {"region": "HPF"},
        {"bwm_include": True},
    )
    assert path.name == f"hippocampus{SET_SUFFIX}"
    loaded, warnings = read_set(path, manifest_versions()["manifest_version"])
    assert loaded["eids"] == ["s1", "s3"] and loaded["session_filter"] == {"region": "HPF"}
    assert warnings == []
    assert [s["name"] for s in list_sets(tmp_path)] == ["hippocampus"]


def test_a_changed_manifest_version_warns_and_a_tampered_set_is_refused(tmp_path):
    path = save_set(tmp_path, "old", ["s1"], 1, {}, {})
    _, warnings = read_set(path, 2)
    assert any("manifest version 1" in w and "now 2" in w for w in warnings)
    raw = json.loads(path.read_text())
    raw["eids"] = ["s1", "s2"]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="hash"):
        read_set(path, 1)


@pytest.mark.parametrize("bad", ["../escape", "a/b", "", "x" * 200])
def test_set_names_are_plain(tmp_path, bad):
    with pytest.raises(ValueError, match="name"):
        save_set(tmp_path, bad, ["s1"], 2, {}, {})


# ---------- Phy folders under a root ----------


def _phy(root, name, events=True):
    folder = write_phy_folder(root / name, [30, 60], [1, 1])
    if events:
        EVENTS.to_csv(folder / "events.csv", index=False)
    return folder


def test_a_phy_folder_inside_the_root_opens_with_its_events(tmp_path):
    root = tmp_path / "phy"
    folder = _phy(root, "mouse1/probe00")
    found, events = resolve_phy_folder(root, "mouse1/probe00")
    assert found == folder.resolve() and events == (folder / "events.csv").resolve()


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("../outside", "outside the Phy folder root"),
        ("mouse1/../../outside", "outside the Phy folder root"),
        ("/etc", "outside the Phy folder root"),
        ("escape_link", "outside the Phy folder root"),
        ("not_phy", "no spike_times.npy"),
        ("no_events", "no events.csv"),
        ("missing", "does not exist"),
    ],
)
def test_phy_paths_are_refused_in_plain_language(tmp_path, path, message):
    root = tmp_path / "phy"
    _phy(root, "mouse1/probe00")
    _phy(tmp_path, "outside")
    (root / "not_phy").mkdir()
    _phy(root, "no_events", events=False)
    os.symlink(tmp_path / "outside", root / "escape_link")
    with pytest.raises(ValueError, match=message):
        resolve_phy_folder(root, path)


def test_completion_lists_folders_under_the_root_and_marks_phy_ones(tmp_path):
    root = tmp_path / "phy"
    _phy(root, "mouse1/probe00")
    _phy(root, "mouse2/probe00", events=False)
    _phy(tmp_path, "outside")
    os.symlink(tmp_path / "outside", root / "escape_link")
    top = complete_phy_path(root, "")
    assert [c["path"] for c in top] == ["mouse1", "mouse2"]  # the symlink out is not offered
    deeper = complete_phy_path(root, "mouse1/")
    # recording: whether it holds a recording.yaml (step 14a; tests/test_recording.py)
    assert deeper == [{"path": "mouse1/probe00", "phy": True, "events": True, "recording": False}]
    assert complete_phy_path(root, "mouse2/")[0]["events"] is False
    with pytest.raises(ValueError, match="outside"):
        complete_phy_path(root, "../")


# ---------- recent projects ----------


def test_recent_projects_are_the_folders_files_newest_first(tmp_path):
    projects = tmp_path / "projects"
    projects.mkdir()
    for i, name in enumerate(["a", "b", "c"]):
        path = projects / f"{name}.ndstudio.json"
        path.write_text(json.dumps({"source": {"kind": "ibl", "eid": f"e{name}"}, "view": {}}))
        os.utime(path, (time.time() - 100 + i * 10, time.time() - 100 + i * 10))
    (projects / "notes.txt").write_text("not a project")
    listed = recent_projects(projects)
    assert [p["name"] for p in listed] == ["c", "b", "a"]
    assert listed[0]["source"] == {"kind": "ibl", "eid": "ec"}
    assert recent_projects(tmp_path / "none") == []


# ---------- file endings from before the rename (docs/DECISIONS.md) ----------


def test_an_old_ending_set_opens_and_saving_writes_the_new_ending(tmp_path):
    new = save_set(tmp_path, "cortex", ["s1"], 2, {}, {})
    assert new.name == "cortex.unitwave-set.json"
    old = new.rename(tmp_path / "cortex.ndset.json")
    assert [(s["name"], s["file"]) for s in list_sets(tmp_path)] == [
        ("cortex", "cortex.ndset.json")
    ]
    loaded, _ = read_set(set_path(tmp_path, "cortex"), 2)
    assert loaded["eids"] == ["s1"]
    again = save_set(tmp_path, "cortex", ["s1", "s2"], 2, {}, {})
    assert again.name == "cortex.unitwave-set.json" and old.exists()  # the old file is kept
    files = sorted(s["file"] for s in list_sets(tmp_path))
    assert files == ["cortex.ndset.json", "cortex.unitwave-set.json"]
    # By name the new file wins; by file name either opens.
    assert set_path(tmp_path, "cortex") == again
    assert set_path(tmp_path, "cortex.ndset.json") == old


def test_recent_projects_list_both_endings_and_open_by_either(tmp_path):
    projects = tmp_path / "projects"
    projects.mkdir()
    for file in ("old.ndstudio.json", "new.unitwave.json"):
        (projects / file).write_text(json.dumps({"source": {"kind": "ibl", "eid": "e"}}))
    listed = sorted((p["name"], p["file"]) for p in recent_projects(projects))
    assert listed == [("new", "new.unitwave.json"), ("old", "old.ndstudio.json")]
    assert project_path(projects, "old") == projects / "old.ndstudio.json"
    assert project_path(projects, "old.ndstudio.json") == projects / "old.ndstudio.json"
    assert project_path(projects, "new") == projects / "new.unitwave.json"
    with pytest.raises(ValueError, match="by its name"):
        project_path(projects, "../old.ndstudio.json")
