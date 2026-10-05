import json

import numpy as np
import pandas as pd
from phy_folder import write_phy_folder

from unitwave.analysis.events import event_times
from unitwave.analysis.psth import alternate_halves, population_psth, psth
from unitwave.analysis.responsiveness import ResponseConfig
from unitwave.studio.export import export_view
from unitwave.studio.project import DEFAULT_VIEW, Source, load_source
from unitwave.studio.server import Studio

SAMPLES = [30, 60, 90, 150, 30000, 45000, 60000, 90000]
CLUSTERS = [3, 7, 3, 7, 3, 11, 9, 7]
EVENTS = pd.DataFrame(
    {
        "intervals_0": [0.5, 1.5, 2.0, 2.5],
        "intervals_1": [1.4, 1.9, 2.4, 2.9],
        "stimOn_times": [0.6, 1.6, 2.1, 2.6],
    }
)
VIEW = {
    **DEFAULT_VIEW,
    "event": "stim_on",
    "t0": -0.5,
    "t1": 0.5,
    "bin": 0.1,
    "all": True,
    "unit": "imec0_3",
}


def _studio(tmp_path) -> Studio:
    folder = write_phy_folder(tmp_path / "imec0", SAMPLES, CLUSTERS, ks_label={3: "good"})
    EVENTS.to_csv(tmp_path / "events.csv", index=False)
    source = Source(kind="phy", folder=str(folder), events=str(tmp_path / "events.csv"))
    session, qc = load_source(source)
    return Studio(session, qc, tmp_path / "atlas", source=source)


def test_export_writes_vector_figures_sidecars_and_a_manifest(tmp_path):
    out = export_view(_studio(tmp_path), VIEW, tmp_path / "runs")
    assert out.parent == tmp_path / "runs" and out.name.endswith("_studio")
    names = {p.name for p in out.iterdir()}
    assert names == {
        "unit.svg",
        "unit.pdf",
        "unit.json",
        "population.svg",
        "population.pdf",
        "population.json",
        "quality.svg",
        "quality.pdf",
        "quality.json",
        "manifest.json",
    }
    assert (out / "unit.pdf").read_bytes()[:5] == b"%PDF-"
    assert "<svg" in (out / "unit.svg").read_text()
    assert "UnitWave Studio" in (out / "unit.svg").read_text()  # the figure says what made it
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["app"] == "UnitWave Studio"
    assert manifest["view"] == VIEW
    assert manifest["project"]["source"]["kind"] == "phy"
    assert set(manifest["project"]["files"]) >= {"spike_times.npy", "events.csv"}
    assert manifest["git"]["sha"] and "dirty" in manifest["git"]
    assert manifest["seed"] is None  # nothing random is drawn
    assert set(manifest["files"]) == names - {"manifest.json"}


def test_sidecars_equal_the_engine_output(tmp_path):
    studio = _studio(tmp_path)
    out = export_view(studio, VIEW, tmp_path / "runs")
    session = studio.session
    events = event_times(session.trials, "stim_on")

    unit = json.loads((out / "unit.json").read_text())
    expected = psth(session.spikes["imec0_3"], events, (-0.5, 0.5), 0.1)
    np.testing.assert_allclose(unit["psth"]["mean"], expected.mean)
    np.testing.assert_allclose(unit["psth"]["bin_centers"], expected.bin_centers)
    assert unit["psth"]["n_trials"] == 4 and unit["caption"].startswith("imec0_3 · ")

    pop = json.loads((out / "population.json").read_text())
    _, show = alternate_halves(events)
    rows = population_psth(session.spikes, pop["units"], show, (-0.5, 0.5), 0.1)
    np.testing.assert_allclose(pop["rates_hz"], rows)
    assert pop["n_show_trials"] == 2 and pop["n_sort_trials"] == 2


def test_a_responsiveness_result_is_exported_with_its_config(tmp_path):
    studio = _studio(tmp_path)
    studio.response_cfg = ResponseConfig((-0.2, 0.0), (0.0, 0.3), 0.001, 0.5, 0.05, 0.5, 1.0, 1.0)
    studio.test_json({"event": "stim_on", "all": "1"})
    out = export_view(studio, VIEW, tmp_path / "runs")
    table = pd.read_csv(out / "responsiveness.csv")
    assert list(table.columns)[:4] == ["unit_id", "statistic_hz", "p", "q"]
    assert len(table) == 4
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["responsiveness"]["min_shift_s"] == 0.5
