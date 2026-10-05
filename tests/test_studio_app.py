"""The homepage app: no session at start, opening and switching sessions, trial filters."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pandas as pd
import pytest
from phy_folder import write_phy_folder
from test_catalog import _manifest

from unitwave.analysis.responsiveness import ResponseConfig
from unitwave.data.backends.phy import load_session_phy
from unitwave.data.load import load_data_config
from unitwave.qc.phy import PhyUnitQC
from unitwave.studio.server import App, Studio, build_app, make_handler

SAMPLES = [30, 60, 90, 150, 30000, 45000, 60000, 90000]
CLUSTERS = [3, 7, 3, 7, 3, 11, 9, 7]
EVENTS = pd.DataFrame(
    {
        "intervals_0": [0.5, 1.2, 1.9, 2.4],
        "intervals_1": [1.1, 1.8, 2.3, 2.9],
        "stimOn_times": [0.6, 1.3, 2.0, 2.5],
        "feedbackType": [1.0, -1.0, 1.0, 1.0],
    }
)
QC = PhyUnitQC(("good",), 0.1, refractory_contamination=0.1, refractory_alpha=0.1)


def _session(tmp_path, name="imec0"):
    folder = write_phy_folder(tmp_path / name, SAMPLES, CLUSTERS, ks_label={3: "good"})
    EVENTS.to_csv(tmp_path / f"{name}.csv", index=False)
    return load_session_phy(folder, tmp_path / f"{name}.csv"), folder


def _app(tmp_path, **kwargs) -> App:
    session, _ = _session(tmp_path)
    return App(
        load_data_config(), manifest=_manifest(), loader=lambda source: (session, QC), **kwargs
    )


def _serve(app):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _get(url):
    try:
        with urllib.request.urlopen(url) as r:
            return r.status, r.read().decode(), r.geturl()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(), url


def test_the_server_starts_with_no_session_and_serves_the_homepage(tmp_path):
    app = _app(tmp_path)
    assert app.studio is None
    server, base = _serve(app)
    try:
        status, body, _ = _get(base + "/api/session")
        assert status == 400 and "No session is open" in body
        status, body, final = _get(base + "/session")
        assert final.endswith("/") and "Choose data" in body  # redirected to the homepage
        status, body, _ = _get(base + "/")
        assert status == 200 and "Choose data" in body
    finally:
        server.shutdown()


def test_the_old_flags_still_skip_the_homepage(tmp_path):
    _, folder = _session(tmp_path)
    app, path = build_app(["--phy", str(folder), "--events", str(tmp_path / "imec0.csv")])
    assert path == "/session" and app.studio is not None
    app, path = build_app([])
    assert path == "/" and app.studio is None


def test_opening_is_a_guarded_post_and_switching_resets_state(tmp_path):
    app = _app(tmp_path)
    server, base = _serve(app)
    body = json.dumps({"kind": "ibl", "eid": "s1", "trials": {"outcomes": [1.0]}}).encode()

    def post(headers):
        request = urllib.request.Request(base + "/api/open", data=body, headers=headers)
        try:
            return urllib.request.urlopen(request).status
        except urllib.error.HTTPError as e:
            return e.code

    try:
        assert post({"Content-Type": "text/plain"}) == 415
        assert post({"Content-Type": "application/json", "Origin": "http://evil.test"}) == 403
        assert app.studio is None
        assert post({"Content-Type": "application/json"}) == 200
        first = app.studio
        assert first.view["trials"] == {
            "bwm_include": False,
            "exclude_nogo": False,
            "contrasts": [],
            "blocks": [],
            "outcomes": [1.0],
        }
        first.response_cfg = ResponseConfig(
            (-0.2, 0.0), (0.0, 0.3), 0.001, 0.5, 0.05, 0.5, 1.0, 1.0
        )
        first.test_json({"event": "stim_on", "all": "1"})
        assert first._tests
        assert post({"Content-Type": "application/json"}) == 200
        assert app.studio is not first and not app.studio._tests and not app.studio._selectivity
    finally:
        server.shutdown()


def test_a_result_under_one_trial_filter_is_never_shown_under_another(tmp_path):
    session, _ = _session(tmp_path)
    studio = Studio(session, QC, tmp_path / "atlas")
    studio.response_cfg = ResponseConfig((-0.2, 0.0), (0.0, 0.3), 0.001, 0.5, 0.05, 0.5, 1.0, 1.0)
    rewards = {"event": "stim_on", "all": "1", "tf": json.dumps({"outcomes": [1.0]})}
    everything = {"event": "stim_on", "all": "1"}
    summary = studio.test_json(rewards)
    assert summary["n_trials"] == 3  # the error trial is filtered out
    assert studio.units_json(rewards)["test"] is not None
    assert studio.units_json(everything)["test"] is None
    assert all("resp" not in r for r in studio.units_json(everything)["units"])
    with pytest.raises(ValueError, match="run the responsiveness test"):
        studio.units_json({**everything, "responsive": "1"})
    trials = studio.units_json(rewards)["trials"]
    assert trials["n_kept"] == 3 and trials["excluded"] == {"outcome not selected": 1}


def test_home_json_filters_counts_and_marks_cached_sessions(tmp_path):
    app = _app(tmp_path)
    home = app.home_json({"f": json.dumps({"labs": ["labX"], "region": "HPF"})})
    assert home["summary"] == {"n_sessions": 2, "n_probes": 3, "n_units": 42, "n_region_units": 42}
    assert [r["eid"] for r in home["sessions"]] == ["s1", "s3"]
    assert all(r["cached"] is False for r in home["sessions"])
    assert home["options"]["labs"] == ["labX", "labY"]
    assert "not Studio's QC" in home["notes"]["units"]
    by_units = app.home_json({"f": "{}", "sort": "n_good_units", "desc": "1"})
    assert [r["eid"] for r in by_units["sessions"]] == ["s1", "s3", "s2"]


def test_the_default_trial_filter_keeps_only_what_the_session_supports(tmp_path):
    app = _app(tmp_path)  # the Phy events have no bwm_include or choice column
    app.open({"kind": "ibl", "eid": "s1"})
    assert app.studio.view["trials"]["bwm_include"] is False
    assert app.studio.view["trials"]["exclude_nogo"] is False


def test_opening_a_phy_folder_goes_through_the_root(tmp_path):
    import dataclasses

    from unitwave.analysis.catalog import load_catalog_config

    root = tmp_path / "phy"
    folder = write_phy_folder(root / "m1" / "probe00", SAMPLES, CLUSTERS, ks_label={3: "good"})
    EVENTS.to_csv(folder / "events.csv", index=False)
    app = App(load_data_config(), manifest=_manifest())
    app.catalog_cfg = dataclasses.replace(load_catalog_config(), phy_root=str(root))
    assert app.phy_complete({"prefix": "m1/"})["choices"][0]["phy"] is True
    with pytest.raises(ValueError, match="outside the Phy folder root"):
        app.open({"kind": "phy", "path": "../elsewhere"})
    app.open({"kind": "phy", "path": "m1/probe00"})
    assert app.studio.source.folder == str(folder.resolve())


def test_a_recent_project_opens_by_name_only(tmp_path):
    from unitwave.studio.project import DEFAULT_VIEW, Source, make_project, save_project

    folder = write_phy_folder(tmp_path / "imec0", SAMPLES, CLUSTERS, ks_label={3: "good"})
    EVENTS.to_csv(tmp_path / "events.csv", index=False)
    source = Source(kind="phy", folder=str(folder), events=str(tmp_path / "events.csv"))
    session = load_session_phy(folder, tmp_path / "events.csv")
    data = dataclasses_replace_root(load_data_config(), tmp_path)
    save_project(
        make_project(source, session, QC, {**DEFAULT_VIEW, "event": "stim_on"}),
        data.data_root / "projects" / "mine.unitwave.json",
    )
    app = App(data, manifest=_manifest())
    assert [p["name"] for p in app.projects({})["projects"]] == ["mine"]
    with pytest.raises(ValueError, match="by its name"):
        app.open({"kind": "project", "name": "../mine"})
    app.open({"kind": "project", "name": "mine"})
    assert app.studio.view["event"] == "stim_on"


def dataclasses_replace_root(data, root):
    import dataclasses

    return dataclasses.replace(data, data_root=root)
