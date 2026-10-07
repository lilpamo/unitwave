"""How far a source's own unit QC and spike-time QC agree, logged under runs/<run_id>/.

    python -m unitwave.cli.qc_agreement EID [EID ...]
    python -m unitwave.cli.qc_agreement --set "S1 sessions"

For IBL sessions: IBL's label rule (configs/qc.yaml) against the spike-time rule
(configs/qc_spikes.yaml, qc.spike_times). Descriptive: it decides nothing. Each session
is read from the local cache only; one that isn't there is left out, with why, never
downloaded. Each run writes:
- manifest.json: the git SHA, both configs with their sha256, the sessions used and
  those left out with why;
- agreement.json: per session and in total, the units passing both rules, either one,
  or neither, the share agreeing, and the criteria behind each disagreement;
- units.csv: every unit's two verdicts with their reasons, rate and presence ratio.
"""

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from unitwave.analysis.units import unit_table
from unitwave.cli.evaluate import REPO, _git, _jsonable, _sha256
from unitwave.data.cache import SessionCache
from unitwave.data.load import key_parts, load_data_config, load_session
from unitwave.data.manifest import MANIFEST_VERSION
from unitwave.qc.spike_times import DEFAULT_CONFIG as SPIKE_QC_CONFIG
from unitwave.qc.spike_times import agreement, load_spike_qc_config
from unitwave.qc.units import DEFAULT_CONFIG as QC_CONFIG
from unitwave.qc.units import load_qc_config
from unitwave.studio.sets import read_set, set_path

CONFIG_FILES = {"qc": Path(QC_CONFIG), "spike_qc": Path(SPIKE_QC_CONFIG)}
_COLUMNS = [
    "qc_passed",
    "qc_reason",
    "spike_qc_passed",
    "spike_qc_reason",
    "firing_rate_hz",
    "presence_ratio",
    "region",
]


def load_cached(eid: str):
    """An IBL session from the local cache; refused, never downloaded, if absent."""
    data = load_data_config()
    if not SessionCache(data.cache_root).path(key_parts(eid, "bwm")).exists():
        raise ValueError("not in the local cache (open it in Studio first)")
    return load_session(eid, "bwm")


def run(eids, *, runs_dir: Path = REPO / "runs", load=load_cached, progress=print) -> Path:
    """Compare both rules on every session; the run's folder."""
    qc, spike_qc = load_qc_config(), load_spike_qc_config()
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}_qc_agreement"
    out = Path(runs_dir) / run_id
    out.mkdir(parents=True)
    manifest = {
        "run_id": run_id,
        "status": "running",
        "created": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "git": _git(),
        "eids": list(eids),
        "configs": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
                "content": yaml.safe_load(path.read_text()),
            }
            for name, path in CONFIG_FILES.items()
        },
        "versions": {
            "python": platform.python_version(),
            **{m: sys.modules[m].__version__ for m in ("numpy", "pandas", "scipy")},
        },
        "sessions_used": [],
        "sessions_failed": {},
    }

    def save(name, value):
        (out / name).write_text(json.dumps(_jsonable(value), indent=1))

    save("manifest.json", manifest)
    tables, per_session = [], {}
    for i, eid in enumerate(eids, 1):
        try:
            table = unit_table(load(eid), qc, spike_qc)
        except (ValueError, KeyError, OSError) as e:  # left out, and counted with why
            manifest["sessions_failed"][eid] = str(e)
            progress(f"{i}/{len(eids)} {eid[:8]}: left out ({e})")
            continue
        per_session[eid] = agreement(table)
        tables.append(table[_COLUMNS].assign(eid=eid).rename_axis("unit_id").reset_index())
        manifest["sessions_used"].append(eid)
        a = per_session[eid]
        progress(
            f"{i}/{len(eids)} {eid[:8]}: {a['n_units']} units, both {a['both']}, label only "
            f"{a['source_only']}, spikes only {a['spikes_only']}, neither {a['neither']}"
        )
        save("manifest.json", manifest)
    if not tables:
        manifest["status"] = "failed: no session could be read"
        save("manifest.json", manifest)
        raise ValueError("no session could be read")
    units = pd.concat(tables, ignore_index=True)[["eid", "unit_id", *_COLUMNS]]
    units.to_csv(out / "units.csv", index=False)
    save("agreement.json", {"sessions": per_session, "total": agreement(units)})
    manifest["status"] = "complete"
    save("manifest.json", manifest)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("eids", nargs="*", help="IBL session ids")
    ap.add_argument("--set", default="", help="a saved session set's name, instead of ids")
    ap.add_argument("--runs-dir", type=Path, default=REPO / "runs")
    args = ap.parse_args(argv)
    eids = list(args.eids)
    if args.set:
        sets_dir = load_data_config().data_root / "sets"
        eids += read_set(set_path(sets_dir, args.set), MANIFEST_VERSION)[0]["eids"]
    if not eids:
        ap.error("give session ids or --set")
    print(f"wrote {run(eids, runs_dir=args.runs_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
