"""Step 5: NWB files in Studio, from the homepage to every view. Hand-built files are
test inputs only; the Richards file is DANDI 000017's, used only when downloaded."""

import dataclasses
import json
from pathlib import Path

import pytest
import yaml
from test_nwb_intake import LAYOUT, RICHARDS, _write

from unitwave.data.load import load_data_config
from unitwave.studio.project import (
    DEFAULT_VIEW,
    Source,
    load_source,
    make_project,
    open_project,
    save_project,
)
from unitwave.studio.server import App, Studio


def _source(tmp_path) -> Source:
    path = _write(tmp_path / "f.nwb")
    layout = tmp_path / "layout.yaml"
    layout.write_text(yaml.safe_dump(LAYOUT))
    return Source(kind="nwb", file=str(path), layout=str(layout), task="steinmetz")


def test_an_nwb_file_opens_with_its_layouts_rule_and_report(tmp_path):
    source = _source(tmp_path)
    session, qc = load_source(source)
    studio = Studio(session, qc, tmp_path / "atlas", source)
    s = studio.session_json({})
    assert s["source"]["kind"] == "nwb" and s["source"]["label"].endswith("Test layout")
    report = s["source"]["report"]
    assert "sampling period" in report["behaviour"]["wheel"]["refused"]
    assert s["qc"]["rule"] == "NWB quality (phy_annotations)"
    assert s["qc"]["config"] == "configs/qc_nwb.yaml"
    rows = {u["id"]: u for u in studio.units_json({"all": "1"})["units"]}
    assert rows["ProbeA_0"]["label"] == "good" and rows["ProbeA_0"]["qc_passed"] is True
    assert rows["ProbeA_1"]["label"] == "MUA"
    assert rows["ProbeA_1"]["qc_reason"] == "phy_annotations 1 (MUA) < 2"
    assert s["decoding"]["available"] is False and "an NWB file lacks" in s["decoding"]["why"]
    q = studio.quality_json({"unit": "ProbeA_0"})
    assert q["ibl_missing"].startswith("only for IBL sessions")
    assert q["waveform_missing"] == "waveforms aren't read from NWB files"
    assert studio.geometry_json({})["missing"].startswith("the test's coordinates")


def test_an_nwb_project_records_the_file_and_its_layout(tmp_path):
    source = _source(tmp_path)
    session, qc = load_source(source)
    path = tmp_path / "p.unitwave.json"
    save_project(make_project(source, session, qc, {**DEFAULT_VIEW, "event": "stim_on"}), path)
    saved = json.loads(path.read_text())
    assert saved["files"] == {"f.nwb": saved["files"]["f.nwb"]}
    assert saved["source"]["layout_label"] == "Test layout" and "layout_sha256" in saved["source"]
    assert open_project(path)[3] == []
    assert Source.from_record(saved["source"]) == source
    layout_file = Path(source.layout)
    layout = yaml.safe_load(layout_file.read_text())
    layout["label"] = "Test layout, edited"
    layout_file.write_text(yaml.safe_dump(layout))
    assert any("NWB layout" in w and "changed" in w for w in open_project(path)[3])


def test_the_homepage_opens_nwb_files_only_from_under_its_root(tmp_path):
    from test_catalog import _manifest

    (tmp_path / "root" / "sub").mkdir(parents=True)
    _write(tmp_path / "root" / "sub" / "f.nwb")
    (tmp_path / "root" / "notes.txt").write_text("not NWB")
    layout = tmp_path / "layout.yaml"
    layout.write_text(yaml.safe_dump(LAYOUT))
    app = App(load_data_config(), manifest=_manifest())
    app.catalog_cfg = dataclasses.replace(app.catalog_cfg, nwb_root=str(tmp_path / "root"))
    assert app.nwb_complete({"prefix": "sub/"})["choices"][0]["path"] == "sub/f.nwb"
    assert [c["path"] for c in app.nwb_complete({"prefix": ""})["choices"]] == ["sub"]
    opened = app.open({"kind": "nwb", "path": "sub/f.nwb", "layout": str(layout)})
    assert opened["url"] == "/session" and app.studio.task.name == "steinmetz"
    assert app.studio.view["trials"] == {"included": False, "exclude_nogo": False, "outcomes": []}
    with pytest.raises(ValueError, match="outside the NWB file root"):
        app.open({"kind": "nwb", "path": "../f.nwb"})
    with pytest.raises(ValueError, match="not an .nwb file"):
        app.open({"kind": "nwb", "path": "notes.txt"})
    assert any(t["name"] == "steinmetz_2019" for t in app.home_json({})["options"]["layouts"])


@pytest.mark.skipif(not RICHARDS.exists(), reason="the Steinmetz Richards file isn't downloaded")
def test_the_richards_session_in_studio(tmp_path):
    source = Source(kind="nwb", file=str(RICHARDS), layout="steinmetz_2019", task="steinmetz")
    session, qc = load_source(source)
    studio = Studio(session, qc, load_data_config().data_root / "atlas", source)
    s = studio.session_json({})
    assert s["n_units_total"] == 778 and s["qc"]["rule"] == "NWB quality (phy_annotations)"
    assert s["events"] == {
        "stim_on": "Stimulus onset",
        "go_cue": "Go cue",
        "response": "Response",
        "feedback_reward": "Feedback: reward",
        "feedback_error": "Feedback: error",
    }
    assert s["comparisons"] == {
        "choice": ["right (-1)", "left (+1)"],
        "outcome": ["error", "reward"],
    }
    assert s["movement"]["first_movement"] is False and s["movement"]["stimulus"] is None
    assert "units.acronym" not in s["missing"] and studio.has_regions and not studio.has_positions
    q = {"event": "stim_on", "t0": "-0.5", "t1": "1.0", "bin": "0.02", "theme": "light"}
    unit = studio.units_json({**q})["units"][0]["id"]
    d = studio.unit_data({**q, "unit": unit, "split": "choice"})
    assert [g.name for g in d["groups"]] == ["right (-1)", "no-go (0)", "left (+1)"]
    sel = studio.selectivity_json({**q, "split": "choice"})
    assert sel["null"] == "choice permuted within contrasts"
