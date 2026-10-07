"""Environment variables after the rename: UNITWAVE_* names, NEURODECODER_* still read."""

import pytest

from unitwave import env as env_module
from unitwave.data.load import load_data_config
from unitwave.env import env


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    for name in ("UNITWAVE_DATA_ROOT", "NEURODECODER_DATA_ROOT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(env_module, "_warned", set())


def test_the_new_name_is_read_without_a_warning(monkeypatch, capsys):
    monkeypatch.setenv("UNITWAVE_DATA_ROOT", "/new")
    assert env("DATA_ROOT") == "/new"
    assert capsys.readouterr().err == ""


def test_the_old_name_still_works_with_one_warning_line(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("NEURODECODER_DATA_ROOT", str(tmp_path))
    assert load_data_config().data_root == tmp_path
    assert env("DATA_ROOT") == str(tmp_path)  # a second read does not warn again
    err = capsys.readouterr().err
    assert err == "warning: NEURODECODER_DATA_ROOT is deprecated; use UNITWAVE_DATA_ROOT\n"


def test_when_both_are_set_the_new_one_wins(monkeypatch, capsys):
    monkeypatch.setenv("UNITWAVE_DATA_ROOT", "/new")
    monkeypatch.setenv("NEURODECODER_DATA_ROOT", "/old")
    assert env("DATA_ROOT") == "/new"
    assert "NEURODECODER_DATA_ROOT is ignored" in capsys.readouterr().err


def test_unset_gives_the_default():
    assert env("DATA_ROOT", "~/data/neurodecoder") == "~/data/neurodecoder"
    assert env("NETWORK_TESTS") is None
