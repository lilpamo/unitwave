from urllib.parse import unquote

import pandas as pd
import pytest
from phy_folder import write_phy_folder

from unitwave.analysis.responsiveness import ResponseConfig
from unitwave.data.backends.phy import load_session_phy
from unitwave.qc.phy import PhyUnitQC
from unitwave.studio.server import Studio, _static

SAMPLES = [30, 60, 90, 150, 30000, 45000, 60000, 90000]
CLUSTERS = [3, 7, 3, 7, 3, 11, 9, 7]
EVENTS = pd.DataFrame(
    {"intervals_0": [0.5, 1.5], "intervals_1": [1.4, 2.9], "stimOn_times": [0.6, 1.6]}
)
PLOT = {"event": "stim_on", "t0": "-0.5", "t1": "0.5", "bin": "0.1", "all": "1"}


def _studio(tmp_path) -> Studio:
    folder = write_phy_folder(
        tmp_path / "imec0", SAMPLES, CLUSTERS, ks_label={3: "good", 9: "good"}
    )
    EVENTS.to_csv(tmp_path / "events.csv", index=False)
    session = load_session_phy(folder, tmp_path / "events.csv")
    qc = PhyUnitQC(("good",), 0.1, refractory_contamination=0.1, refractory_alpha=0.1)
    return Studio(session, qc, tmp_path / "atlas")


def test_static_files_never_escape_the_static_folder():
    assert _static("/static/app.js") is not None
    assert _static("/static/vendor/three/LICENSE") is not None
    assert _static("/static/../server.py") is None
    assert _static("/static/../../data/load.py") is None
    assert _static("/static/nope.js") is None


def test_without_regions_or_positions_the_api_says_why(tmp_path):
    studio = _studio(tmp_path)
    session = studio.session_json({})
    assert session["missing"]["units.acronym"] == "Phy folders have no brain region"
    units = studio.units_json({"all": "1"})
    assert units["level"] is None and units["tree"] == []
    assert [u["region_level"] for u in units["units"]] == [None] * 4
    assert studio.geometry_json({}) == {"missing": "Phy folders have no 3-D position"}
    probe = studio.probe_json({"unit": "imec0_3", "all": "1"})
    assert probe["probe"] == "imec0" and probe["runs"] == []


def test_population_headers_map_heatmap_rows_to_units(tmp_path):
    png, headers = _studio(tmp_path).population_png(PLOT)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert sorted(headers["X-Rows"].split(",")) == ["imec0_11", "imec0_3", "imec0_7", "imec0_9"]
    left, top, right, bottom = map(float, headers["X-Box"].split(","))
    assert 0 <= left < right <= 1 and 0 <= top < bottom <= 1
    assert unquote(headers["X-Caption"]) == (
        "4 units (all regions) · probe imec0 · Stimulus onset · sorted by peak time on odd trials "
        "(n = 1), "
        "showing even trials (n = 1)"
    )


def test_unit_caption_has_no_region_when_there_is_none(tmp_path):
    _, headers = _studio(tmp_path).unit_png({**PLOT, "unit": "imec0_3"})
    assert unquote(headers["X-Caption"]) == "imec0_3 · Stimulus onset · n = 2 trials"


def test_responsiveness_runs_once_per_event_and_feeds_the_table(tmp_path):
    studio = _studio(tmp_path)
    # The hand-built session is 3 s long: shrink the null's minimum shift to fit it.
    studio.response_cfg = ResponseConfig((-0.2, 0.0), (0.0, 0.3), 0.001, 0.5, 0.05)
    q = {"event": "stim_on", "all": "1"}
    with pytest.raises(ValueError, match="run the responsiveness test"):
        studio.units_json({**q, "responsive": "1"})
    summary = studio.test_json(q)
    assert summary["n_tests"] == 4 and summary["n_trials"] == 2
    rows = studio.units_json(q)["units"]
    assert {r["resp"] for r in rows} <= {"up", "down", "no"}
    assert all(0 < r["resp_p"] <= 1 for r in rows)
    assert studio.test_json(q) == summary  # cached, not rerun


def test_posts_only_from_this_page_and_saving_writes_the_project(tmp_path):
    import json
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from unitwave.studio.project import DEFAULT_VIEW, Source
    from unitwave.studio.server import make_handler

    studio = _studio(tmp_path)
    studio.source = Source(
        kind="phy", folder=str(tmp_path / "imec0"), events=str(tmp_path / "events.csv")
    )
    studio.project_path = tmp_path / "p.unitwave.json"
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(studio))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/api/project"
    body = json.dumps({**DEFAULT_VIEW, "event": "stim_on"}).encode()

    def post(headers):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            return urllib.request.urlopen(request).status
        except urllib.error.HTTPError as e:
            return e.code

    try:
        assert post({"Content-Type": "text/plain"}) == 415
        assert post({"Content-Type": "application/json", "Origin": "http://evil.test"}) == 403
        assert not studio.project_path.exists()
        assert post({"Content-Type": "application/json"}) == 200
        assert json.loads(studio.project_path.read_text())["view"]["event"] == "stim_on"
    finally:
        server.shutdown()


def _two_probe_studio(tmp_path) -> Studio:
    """The hand-built session with clusters 9 and 11 relabelled as a second probe."""
    import dataclasses

    studio = _studio(tmp_path)
    session = studio.session
    units = session.units.assign(
        probe_name=["imec0", "imec0", "imec1", "imec1"]  # imec0_3, _7, _9, _11
    )
    session = dataclasses.replace(session, units=units)
    two = Studio(session, studio.qc, tmp_path / "atlas")
    two.response_cfg = ResponseConfig((-0.2, 0.0), (0.0, 0.3), 0.001, 0.5, 0.05)
    return two


def test_the_probe_filter_applies_to_table_and_population(tmp_path):
    studio = _two_probe_studio(tmp_path)
    assert studio.session_json({})["probes"] == ["imec0", "imec1"]
    rows = studio.units_json({"all": "1", "probe": "imec1"})["units"]
    assert [r["id"] for r in rows] == ["imec0_9", "imec0_11"]
    _, headers = studio.population_png({**PLOT, "probe": "imec1"})
    assert sorted(headers["X-Rows"].split(",")) == ["imec0_11", "imec0_9"]
    assert "· probe imec1 ·" in unquote(headers["X-Caption"])
    _, headers = studio.population_png(PLOT)
    assert "· probes imec0, imec1 ·" in unquote(headers["X-Caption"])
    data = studio.population_data(PLOT)
    assert data["probes"] == [studio.units.at[u, "probe"] for u in data["units"]]
    with pytest.raises(ValueError, match="no probe"):
        studio.units_json({"probe": "imec9"})


def test_responsiveness_is_corrected_over_the_units_tested_on_a_probe(tmp_path):
    studio = _two_probe_studio(tmp_path)
    q = {"event": "stim_on", "all": "1"}
    one = studio.test_json({**q, "probe": "imec1"})
    assert one["n_tests"] == 2 and one["probes"] == ["imec1"]
    # The all-probe result is a separate test over 4 units, not reused.
    with pytest.raises(ValueError, match="run the responsiveness test"):
        studio.units_json({**q, "responsive": "1"})
    both = studio.test_json(q)
    assert both["n_tests"] == 4 and both["probes"] == ["imec0", "imec1"]
    tested = studio._tested({**q, "probe": "imec1"})
    assert list(tested.index) == ["imec0_9", "imec0_11"] and (tested["n_tests"] == 2).all()


def test_the_3d_view_follows_the_probe_filter_on_real_data(tmp_path):
    from unitwave.data.load import load_data_config, load_session
    from unitwave.qc.units import load_qc_config

    eid = "d23a44ef-1402-4ed7-97f5-47e9a7a504d9"
    try:
        session = load_session(eid, "bwm")
    except (OSError, ValueError) as e:
        pytest.skip(f"d23a44ef not available: {e}")
    studio = Studio(session, load_qc_config(), load_data_config().data_root / "atlas")
    assert studio.probes == ["probe00", "probe01"]
    both = studio.geometry_json({})
    one = studio.geometry_json({"probe": "probe00"})
    assert [t["probe"] for t in both["tracks"]] == ["probe00", "probe01"]
    assert [t["probe"] for t in one["tracks"]] == ["probe00"]
    assert {u["id"].split("_")[0] for u in one["units"]} == {"probe00"}
    assert len(one["units"]) < len(both["units"])


def test_conditions_offered_follow_the_events_file(tmp_path):
    studio = _studio(tmp_path)  # events: intervals and stimOn only
    assert studio.session_json({})["conditions"] == {}
    with pytest.raises(ValueError, match="choose a condition"):
        studio.tuning_png({**PLOT, "unit": "imec0_3"})


def test_split_tuning_and_selectivity_on_real_data(tmp_path):
    from unitwave.data.load import load_data_config, load_session
    from unitwave.qc.units import load_qc_config
    from unitwave.studio.export import export_view
    from unitwave.studio.project import DEFAULT_VIEW, Source

    eid = "d23a44ef-1402-4ed7-97f5-47e9a7a504d9"
    try:
        session = load_session(eid, "bwm")
    except (OSError, ValueError) as e:
        pytest.skip(f"d23a44ef not available: {e}")
    source = Source(kind="ibl", eid=eid, backend="bwm")
    studio = Studio(session, load_qc_config(), load_data_config().data_root / "atlas", source)
    q = {
        "event": "stim_on",
        "t0": "-0.5",
        "t1": "1.0",
        "bin": "0.02",
        "split": "choice",
        "unit": "probe00_3",
        "theme": "light",
    }
    assert set(studio.session_json({})["conditions"]) == {
        "side",
        "contrast",
        "choice",
        "outcome",
        "block",
        "reaction_time",
    }
    groups = studio.unit_data(q)["groups"]
    assert [g.name for g in groups] == ["right (-1)", "left (+1)"]
    assert sum(g.psth.n_trials for g in groups) == 410
    assert "split by choice" in unquote(studio.unit_png(q)[1]["X-Caption"])
    curve = studio.tuning_data({**q, "split": "contrast"})["curve"]
    assert curve["n"].sum() == 410 and len(curve) == 9
    summary = studio.selectivity_json(q)
    assert summary["n_tests"] == 390 and summary["null"] == "choice permuted within signed contrast"
    rows = studio.units_json(q)["units"]
    assert all(0 <= r["sel_auroc"] <= 1 for r in rows)
    assert studio.units_json(q)["selectivity"]["n_selective"] == summary["n_selective"]
    view = {**DEFAULT_VIEW, "event": "stim_on", "split": "choice", "unit": "probe00_3"}
    out = export_view(studio, view, tmp_path / "runs")
    names = {p.name for p in out.iterdir()}
    assert {"tuning.svg", "tuning.json", "selectivity.csv"} <= names


def test_phy_sessions_say_why_there_are_no_movement_controls(tmp_path):
    studio = _studio(tmp_path)  # events: intervals and stimOn only; Phy has no wheel
    movement = studio.session_json({})["movement"]
    assert movement["wheel"] is False and movement["first_movement"] is False
    assert movement["wheel_missing"] == "Phy import reads spikes and events only"
    with pytest.raises(ValueError, match="No wheel: Phy import reads spikes and events only"):
        studio.wheel_png(PLOT)
    with pytest.raises(ValueError, match="firstMovement_times"):
        studio.locking_json({"all": "1"})
    studio.response_cfg = ResponseConfig((-0.2, 0.0), (0.0, 0.3), 0.001, 0.5, 0.05)
    with pytest.raises(ValueError, match="firstMovement_times"):
        studio.test_json({"event": "stim_on", "all": "1", "movement_free": "1"})


def test_movement_controls_on_real_data(tmp_path):
    import json

    from unitwave.analysis.movement import MovementConfig
    from unitwave.data.load import load_data_config, load_session
    from unitwave.qc.units import load_qc_config
    from unitwave.studio.export import export_view
    from unitwave.studio.project import DEFAULT_VIEW, Source

    eid = "d23a44ef-1402-4ed7-97f5-47e9a7a504d9"
    try:
        session = load_session(eid, "bwm")
    except (OSError, ValueError) as e:
        pytest.skip(f"d23a44ef not available: {e}")
    source = Source(kind="ibl", eid=eid, backend="bwm")
    studio = Studio(session, load_qc_config(), load_data_config().data_root / "atlas", source)
    studio.movement_cfg = MovementConfig((-0.2, 0.0), (0.0, 0.2), 200, 0)  # fewer, for speed
    q = {"event": "stim_on", "t0": "-0.5", "t1": "1.0", "bin": "0.02", "theme": "light"}
    assert studio.session_json({})["movement"]["wheel"] is True
    png, headers = studio.wheel_png(q)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert unquote(headers["X-Caption"]).startswith("Wheel speed · Stimulus onset · n = 410 trials")
    # Movement-free responsiveness is a separate result from the all-trials one.
    free = studio.test_json({**q, "probe": "probe01", "movement_free": "1"})
    assert free["movement_free"] is True and free["n_trials"] == 87
    with pytest.raises(ValueError, match="run the responsiveness test"):
        studio.units_json({**q, "probe": "probe01", "responsive": "1"})
    with pytest.raises(ValueError, match="stimulus onset only"):
        studio.test_json({**q, "event": "feedback", "movement_free": "1"})
    locking = studio.locking_json({**q, "probe": "probe01"})
    assert locking["n_trials"] == 410 and locking["n_null"] == 200
    assert locking["null"] == "reaction times permuted within signed contrast"
    rows = studio.units_json({**q, "probe": "probe01"})["units"]
    assert len(rows) == locking["n_tests"]
    assert sum(r["locked"] for r in rows) == locking["n_locked"]
    view = {**DEFAULT_VIEW, "probe": "probe01", "movement_free": True, "unit": rows[0]["id"]}
    out = export_view(studio, view, tmp_path / "runs")
    names = {p.name for p in out.iterdir()}
    assert {"wheel.svg", "wheel.json", "movement_locking.csv", "responsiveness.csv"} <= names
    assert json.loads((out / "wheel.json").read_text())["unit"] == "rad/s"
    assert pd.read_csv(out / "responsiveness.csv")["n_trials"].iloc[0] == 87


def test_the_trial_view_on_a_phy_folder(tmp_path):
    studio = _studio(tmp_path)  # 2 trials; events: intervals and stimOn only; no wheel
    q = {**PLOT, "trial": "0", "unit": "imec0_3"}
    d = studio.trial_json(q)
    assert d["header"]["trial"] == 0 and d["header"]["passes_filter"] is True
    assert d["not_recorded"] == [] and "goCue_times" in d["absent_events"]
    assert d["wheel_missing"] == "Phy import reads spikes and events only"
    assert (d["place"], d["n"], d["previous"], d["next"]) == (1, 2, None, 1)
    png, headers = studio.trial_png(q)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert sorted(headers["X-Rows"].split(",")) == ["imec0_11", "imec0_3", "imec0_7", "imec0_9"]
    caption = unquote(headers["X-Caption"])
    assert caption.startswith("Single trial, descriptive (no test) · trial 0 (0-based) · 4 units")
    with pytest.raises(ValueError, match="whole numbers"):
        studio.trial_json({**q, "trial": "null"})
    with pytest.raises(ValueError, match="no trial 5: this session has trials 0 to 1"):
        studio.trial_json({**q, "trial": "5"})
    # The unit raster says which trial each of its rows is, for opening it here.
    _, headers = studio.unit_png({**PLOT, "unit": "imec0_3"})
    assert headers["X-Trials"] == "0,1" and len(headers["X-Box"].split(",")) == 4


def test_the_trial_view_on_real_data(tmp_path):
    import json

    from unitwave.data.load import load_data_config, load_session
    from unitwave.qc.units import load_qc_config
    from unitwave.studio.export import export_view
    from unitwave.studio.project import DEFAULT_VIEW, Source

    eid = "d23a44ef-1402-4ed7-97f5-47e9a7a504d9"
    try:
        session = load_session(eid, "bwm")
    except (OSError, ValueError) as e:
        pytest.skip(f"d23a44ef not available: {e}")
    source = Source(kind="ibl", eid=eid, backend="bwm")
    studio = Studio(session, load_qc_config(), load_data_config().data_root / "atlas", source)
    tf = json.dumps({"bwm_include": True})
    q = {**PLOT, "all": "0", "tf": tf, "trial": "0", "unit": "probe00_3"}
    d = studio.trial_json(q)
    assert d["header"]["bwm_include"] is True and d["next"] == 2  # trial 1 fails bwm_include
    assert studio.trial_json({**q, "trial_all": "1"})["next"] == 1
    one = studio.trial_json({**q, "trial": "1"})
    assert one["header"]["passes_filter"] is False and one["place"] is None
    _, headers = studio.trial_png({**q, "trial_align": "stimOn_times", "trial_n": "3"})
    assert len(headers["X-Rows"].split(",")) == 390
    _, headers = studio.unit_png({**q, "t0": "-0.5", "t1": "1.0", "bin": "0.02"})
    assert len(headers["X-Trials"].split(",")) == 290 and "1" not in headers["X-Trials"].split(",")
    view = {**DEFAULT_VIEW, "unit": "probe00_3", "trials": {"bwm_include": True}, "trial": 12}
    out = export_view(studio, view, tmp_path / "runs")
    assert {"quality.svg", "quality.pdf", "quality.json"} <= {p.name for p in out.iterdir()}
    sidecar = json.loads((out / "trial.json").read_text())
    assert len(sidecar["rows"]) == 390 and sidecar["window"]["trial"] == 12
    assert {e["label"] for e in sidecar["events"]} >= {"stimulus on", "feedback: reward"}
    assert json.loads((out / "unit.json").read_text())["trials"]["n_kept"] == 290


def test_the_quality_panel_on_a_phy_folder(tmp_path):
    from unitwave.studio.project import Source

    studio = _studio(tmp_path)  # no templates in the folder
    studio.source = Source(
        kind="phy", folder=str(tmp_path / "imec0"), events=str(tmp_path / "events.csv")
    )
    d = studio.quality_json({"unit": "imec0_3"})
    assert d["n_spikes"] == 3 and d["qc"]["config"] == "configs/qc_phy.yaml"
    assert d["refractory"]["used_by_qc"] is True and d["refractory"]["passed"] is False
    assert d["waveform"] is None
    assert d["waveform_missing"].startswith("this Phy folder has no templates.npy")
    assert d["ibl_criteria"] is None and d["ibl_missing"].startswith("not for Phy folders")
    png, headers = studio.quality_png({"unit": "imec0_3", "theme": "dark"})
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert unquote(headers["X-Caption"]).startswith("imec0_3 · quality, descriptive")
    with pytest.raises(ValueError, match="no unit 'nope'"):
        studio.quality_json({"unit": "nope"})


def test_the_quality_panel_on_real_data():
    from unitwave.data.cluster_files import ibl_session_folder
    from unitwave.data.load import load_data_config, load_session
    from unitwave.data.manifest import MANIFEST_VERSION, read_manifest
    from unitwave.qc.units import load_qc_config
    from unitwave.studio.project import Source

    eid = "d23a44ef-1402-4ed7-97f5-47e9a7a504d9"
    data = load_data_config()
    try:
        session = load_session(eid, "bwm")
        sessions = read_manifest(data.derived_root / f"manifest-v{MANIFEST_VERSION}").sessions
    except (OSError, ValueError) as e:
        pytest.skip(f"d23a44ef or the manifest not available: {e}")
    alf = ibl_session_folder(data.one_cache_root, sessions, eid)
    if alf is None:
        pytest.skip("d23a44ef is not in the local ONE cache")
    source = Source(kind="ibl", eid=eid, backend="bwm")
    studio = Studio(session, load_qc_config(), data.data_root / "atlas", source, ibl_alf=alf)
    d = studio.quality_json({"unit": "probe00_3"})
    assert d["waveform"] == {"unit": "µV", "source": "clusters.waveforms.npy (IBL)"}
    assert d["ibl_criteria"]["label"] == session.units.at["probe00_3", "label"]
    assert d["refractory"]["used_by_qc"] is False and d["refractory"]["contamination"] == 0.1
    assert d["qc"]["passed"] is True and d["qc"]["reasons"] == []
    other = studio.quality_json({"unit": studio.units.index[studio.units["probe"] == "probe01"][0]})
    # The cache has probe01's metrics but not its waveforms: the panel names the file.
    assert other["waveform"] is None and "clusters.waveforms.npy" in other["waveform_missing"]
    assert other["ibl_criteria"] is not None


def test_pairs_and_connections_on_a_phy_folder(tmp_path):
    studio = _studio(tmp_path)  # 4 units; no templates, so no site positions
    q = {**PLOT, "unit": "imec0_3", "partner": "imec0_7"}
    d = studio.pair_json(q)
    assert d["close"] is None and d["close_why"].startswith("no site position")
    assert d["tests"] == []  # not run yet
    assert unquote(studio.pair_png(q)[1]["X-Caption"]).startswith("imec0_3 → imec0_7")
    with pytest.raises(ValueError, match="other than the selected unit"):
        studio.pair_json({**q, "partner": "imec0_3"})
    summary = studio.connections_json(q)
    assert summary["n_units"] == 4 and summary["n_pairs"] == 6 and summary["n_tests"] == 12
    assert summary["null"] == "interval jitter, 5 ms windows, exact"
    assert len(studio.pair_json(q)["tests"]) == 2  # both directions, from the cached run
    rows = studio.units_json(q)["units"]
    assert all(r["conn_out"] == 0 and r["conn_in"] == 0 and r["conn_to"] == [] for r in rows)
    # Another unit set is another test: its pairs were not corrected together.
    assert studio.units_json({**q, "all": "0"})["connections"] is None


def test_a_real_monosynaptic_peak_shows_in_the_corrected_correlogram():
    from unitwave.data.load import load_data_config, load_session
    from unitwave.qc.units import load_qc_config

    try:
        session = load_session("d23a44ef-1402-4ed7-97f5-47e9a7a504d9", "bwm")
    except (OSError, ValueError) as e:
        pytest.skip(f"d23a44ef not available: {e}")
    studio = Studio(session, load_qc_config(), load_data_config().data_root / "atlas")
    d = studio.pair_data({"unit": "probe00_446", "partner": "probe00_468"})
    excess = d["observed"] - d["expected"]
    peak = d["lags"][excess.argmax()]
    assert 0.0005 <= peak <= 0.002  # CA1 probe00_446 -> probe00_468, about +1 ms
    assert d["close"] is False


LABEL_WORDS = ("responsive", "selective", "connected", "locked", "significant", "tuned")


def test_trajectories_refuse_plainly_with_too_few_trials(tmp_path):
    studio = _studio(tmp_path)  # 2 trials
    with pytest.raises(ValueError, match="no condition has at least 5 trials in each half"):
        studio.trajectory_json(PLOT)


def test_trajectories_on_real_data_are_descriptive(tmp_path):
    import json

    from unitwave.data.load import load_data_config, load_session
    from unitwave.qc.units import load_qc_config
    from unitwave.studio.export import export_view
    from unitwave.studio.project import DEFAULT_VIEW, Source

    eid = "d23a44ef-1402-4ed7-97f5-47e9a7a504d9"
    try:
        session = load_session(eid, "bwm")
    except (OSError, ValueError) as e:
        pytest.skip(f"d23a44ef not available: {e}")
    source = Source(kind="ibl", eid=eid, backend="bwm")
    studio = Studio(session, load_qc_config(), load_data_config().data_root / "atlas", source)
    tf = json.dumps({"bwm_include": True, "exclude_nogo": True})
    q = {**PLOT, "all": "0", "t0": "-0.5", "t1": "1.0", "bin": "0.02", "split": "choice", "tf": tf}
    d = studio.trajectory_json(q)
    assert d["names"] == ["right (-1)", "left (+1)"] and d["axis_names"] == ["pc_1", "pc_2", "pc_3"]
    assert d["n_fit"] == [108, 38] and d["n_show"] == [107, 37]
    assert "descriptive (no test)" in d["caption"]
    # S4: both trial counts, the fitted and the shown half, are in the caption.
    assert "fit on 146 trials (1st, 3rd, ...) and shown on the other 144" in d["caption"]
    # The page shows no labels: nothing it is sent names a unit or a trajectory as one.
    assert not any(w in json.dumps(d).lower() for w in LABEL_WORDS)
    for dims in ("2", "3"):
        png, _ = studio.trajectory_png({**q, "traj_dims": dims})
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
    view = {**DEFAULT_VIEW, "pop_view": "trajectories", "split": "choice", "trials": json.loads(tf)}
    out = export_view(studio, view, tmp_path / "runs")
    sidecar = json.loads((out / "trajectories.json").read_text())
    assert len(sidecar["units"]) == 390 and len(sidecar["components"]) == 390
    assert {"trajectories.svg", "trajectories.pdf"} <= {p.name for p in out.iterdir()}


def _wait(studio, until, seconds=5.0):
    import time

    end = time.time() + seconds
    while time.time() < end:
        status = studio.decode_status({})
        if until(status):
            return status
        time.sleep(0.02)
    raise AssertionError(f"timed out; last status {status}")


def test_decoding_refuses_a_phy_folder_plainly(tmp_path):
    studio = _studio(tmp_path)
    info = studio.session_json({})["decoding"]
    assert info["available"] is False and "Brain Wide Map" in info["why"]
    assert [t["id"] for t in info["targets"]] == [
        "choice",
        "stimulus_side",
        "block",
        "movement_state",
    ]
    with pytest.raises(ValueError, match="Brain Wide Map"):
        studio.decode_start({"target": "choice", "query": {"all": "1"}}, manifest=None)
    assert studio.decode_status({})["state"] == "idle"


def test_a_decoding_job_runs_in_the_background_with_progress(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace

    from unitwave.studio import server
    from unitwave.studio.project import Source

    studio = _studio(tmp_path)
    studio.source = Source(kind="ibl", eid="e", backend="bwm")  # past the BWM gate
    release, calls = threading.Event(), {}

    def fake_decode(eid, target, *, unit_ids, cfg, manifest, load, progress):
        calls.update(eid=eid, target=target, unit_ids=list(unit_ids), manifest=manifest)
        progress("fitting", 3, 10)
        release.wait(5)
        return SimpleNamespace(summary=lambda: {"target": target, "run": "r1"})

    monkeypatch.setattr(server, "decode", fake_decode)
    with pytest.raises(ValueError, match="not a Studio decoding target"):
        studio.decode_start({"target": "wheel_velocity", "query": {}}, manifest="m")
    started = studio.decode_start({"target": "choice", "query": {"all": "1"}}, manifest="m")
    assert started["state"] == "running" and started["target"] == "choice"
    status = _wait(studio, lambda s: s.get("done") == 3)
    assert status["total"] == 10 and status["stage"] == "fitting" and status["elapsed_s"] >= 0
    with pytest.raises(ValueError, match="already running"):
        studio.decode_start({"target": "block", "query": {}}, manifest="m")
    release.set()
    done = _wait(studio, lambda s: s["state"] == "done")
    assert done["summary"] == {"target": "choice", "run": "r1"}
    # The shown units at the moment of the request were decoded.
    assert calls["unit_ids"] == list(studio._select({"all": "1"})) and calls["manifest"] == "m"


def test_decoding_refuses_units_chosen_by_responsiveness(tmp_path, monkeypatch):
    # Responsiveness is tested on every trial, the decoder's test trials included, so
    # units chosen by it would let test data pick the units: refused, not run.
    from unitwave.studio import server
    from unitwave.studio.project import Source

    studio = _studio(tmp_path)
    studio.source = Source(kind="ibl", eid="e", backend="bwm")
    monkeypatch.setattr(server, "decode", lambda *a, **k: pytest.fail("decode ran"))
    with pytest.raises(ValueError, match="Responsive only"):
        studio.decode_start({"target": "choice", "query": {"responsive": "1"}}, manifest="m")
    assert studio.decode_status({})["state"] == "idle"


def test_a_failed_decoding_job_says_why(tmp_path, monkeypatch):
    from unitwave.studio import server
    from unitwave.studio.project import Source

    studio = _studio(tmp_path)
    studio.source = Source(kind="ibl", eid="e", backend="bwm")

    def refuse(*args, **kwargs):
        raise ValueError("e: 2 biased blocks; leave-one-block-out needs 4")

    monkeypatch.setattr(server, "decode", refuse)
    studio.decode_start({"target": "block", "query": {"all": "1"}}, manifest="m")
    status = _wait(studio, lambda s: s["state"] == "error")
    assert "leave-one-block-out needs 4" in status["error"]
