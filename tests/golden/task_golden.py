"""Golden record for step 3 (user-defined tasks): everything the page computes on
d23a44ef, recorded with the code before task definitions, and checked after.

    .venv/bin/python tests/golden/task_golden.py    # writes d23a44ef_task_golden.*

It goes through the page's own methods (Studio.*_data / *_json), so captions are
covered as well as numbers. Real data, kept in tests as a fixture: never shown in the
app.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from unitwave.analysis.conditions import (
    TrialFilter,
    apply_trial_filter,
    available_conditions,
    condition,
)
from unitwave.data.load import load_data_config, load_session
from unitwave.qc.units import load_qc_config
from unitwave.studio.project import Source
from unitwave.studio.server import Studio

EID = "d23a44ef-1402-4ed7-97f5-47e9a7a504d9"
HERE = Path(__file__).resolve().parent
OUT = HERE / "d23a44ef_task_golden"
TF = {"bwm_include": True, "exclude_nogo": True}
BASE = {
    "t0": "-0.5",
    "t1": "1.0",
    "bin": "0.02",
    "baseline": "0",
    "b0": "-0.2",
    "b1": "0",
    "all": "0",
    "level": "Beryl",
    "tf": json.dumps(TF),
    "theme": "light",
}


def studio() -> Studio:
    session = load_session(EID, "bwm")
    return Studio(
        session,
        load_qc_config(),
        load_data_config().data_root / "atlas",
        Source(kind="ibl", eid=EID, backend="bwm"),
    )


def _table(prefix: str, table: pd.DataFrame, arrays: dict, texts: dict) -> None:
    texts[f"{prefix}/index"] = [str(i) for i in table.index]
    for column in table.columns:
        values = table[column]
        if values.dtype == object or pd.api.types.is_string_dtype(values):
            texts[f"{prefix}/{column}"] = [str(v) for v in values]
        else:
            arrays[f"{prefix}/{column}"] = values.to_numpy(np.float64)


def collect(s: Studio) -> tuple[dict, dict]:
    """(arrays, texts): every number and caption the page shows on d23a44ef."""
    arrays, texts = {}, {}
    session = s.session_json({**BASE, "event": "stim_on"})
    for key in ("events", "conditions", "comparisons", "trial_filters"):
        texts[f"session/{key}"] = json.dumps(session.get(key), sort_keys=True, default=str)
    units = list(s.units.index[s.units["qc_passed"].astype(bool)][:4])
    texts["units"] = units
    for event in session["events"]:
        q = {**BASE, "event": event}
        for unit in units:
            d = s.unit_data({**q, "unit": unit})
            texts[f"unit/{event}/{unit}/caption"] = d["caption"]
            arrays[f"unit/{event}/{unit}/raster_trials"] = np.asarray(d["raster_trials"], float)
            g = d["groups"][0]
            arrays[f"unit/{event}/{unit}/mean"] = np.asarray(g.psth.mean, float)
            arrays[f"unit/{event}/{unit}/sem"] = np.asarray(g.psth.sem, float)
        pop = s.population_data(q)
        texts[f"pop/{event}/caption"] = pop["caption"]
        texts[f"pop/{event}/units"] = pop["units"]
        arrays[f"pop/{event}/rates"] = np.asarray(pop["rates_hz"], float)
    for split in session["conditions"]:
        q = {**BASE, "event": "stim_on", "split": split}
        for unit in units[:2]:
            d = s.unit_data({**q, "unit": unit})
            texts[f"split/{split}/{unit}/caption"] = d["caption"]
            texts[f"split/{split}/{unit}/names"] = [g.name for g in d["groups"]]
            for i, g in enumerate(d["groups"]):
                arrays[f"split/{split}/{unit}/{i}/mean"] = np.asarray(g.psth.mean, float)
            t = s.tuning_data({**q, "unit": unit})
            texts[f"tuning/{split}/{unit}/caption"] = t["caption"]
            _table(f"tuning/{split}/{unit}", t["curve"], arrays, texts)
    trials = s.session.trials
    for name in available_conditions(trials):
        c = condition(trials, name)
        arrays[f"condition/{name}/values"] = c.values
        texts[f"condition/{name}/names"] = list(c.names)
        texts[f"condition/{name}/excluded"] = f"{c.n_excluded} {c.excluded}"
    sel = apply_trial_filter(trials, TrialFilter.from_dict(TF))
    arrays["filter/mask"] = sel.mask.astype(float)
    texts["filter/excluded"] = json.dumps(sel.excluded, sort_keys=True)
    for free in ("0", "1"):
        q = {**BASE, "event": "stim_on", "movement_free": free}
        texts[f"responsive/{free}/summary"] = json.dumps(s.test_json(q), sort_keys=True)
        _table(f"responsive/{free}", s._tests[s._response_key(q)], arrays, texts)
    for split in session["comparisons"]:
        q = {**BASE, "event": "stim_on", "split": split}
        texts[f"selectivity/{split}/summary"] = json.dumps(s.selectivity_json(q), sort_keys=True)
        _table(f"selectivity/{split}", s._selectivity[s._selectivity_key(q)], arrays, texts)
    q = {**BASE, "event": "stim_on"}
    texts["locking/summary"] = json.dumps(s.locking_json(q), sort_keys=True)
    _table("locking", s._locking[s._locking_key(q)], arrays, texts)
    for trial in ("12", "236"):
        tq = {**BASE, "event": "stim_on", "unit": units[0], "trial": trial}
        tq.update(trial_align="stimOn_times", trial_n="3")
        texts[f"trial/{trial}/json"] = json.dumps(s.trial_json(tq), sort_keys=True, default=str)
        texts[f"trial/{trial}/caption"] = s.trial_data(tq)["caption"]
    return arrays, texts


def main() -> None:
    arrays, texts = collect(studio())
    np.savez_compressed(OUT.with_suffix(".npz"), **arrays)
    OUT.with_suffix(".json").write_text(json.dumps(texts, indent=1, sort_keys=True))
    print(f"{len(arrays)} arrays, {len(texts)} texts")


if __name__ == "__main__":
    main()
