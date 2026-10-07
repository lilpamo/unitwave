"""Write a held-out plan, before anything runs on its sessions (step 9b).  [§5]

    python -m unitwave.cli.plan --name "held out" --task steinmetz \
        --step choice_beyond_stimulus/selective --step regions_differ/summary \
        --trial-filter '{"included": true}' --nwb A.nwb B.nwb ... --layout steinmetz_2019
    python -m unitwave.cli.plan --name "held out IBL" --task ibl \
        --step stimulus_beyond_movement/respond --eid EID [EID ...]

The plan names the sessions (an NWB file or Phy folder with its files' sha256), the
recipe steps to run on them and the trial filter, and is saved with a hash and time
under <data_root>/plans/. It is never edited. The first run of a planned step on a
planned session after that is confirmatory; every other result is exploratory
(studio.plans).
"""

import argparse
import json
from pathlib import Path

from unitwave.data.load import load_data_config
from unitwave.studio.plans import read_plan, save_plan, source_identity
from unitwave.studio.project import Source


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--task", required=True, help="the task definition, e.g. ibl or steinmetz")
    ap.add_argument("--step", action="append", required=True, help="recipe/step, repeatable")
    ap.add_argument("--trial-filter", default="{}", help="JSON, e.g. '{\"included\": true}'")
    ap.add_argument("--eid", nargs="+", help="IBL sessions")
    ap.add_argument("--nwb", type=Path, nargs="+", help="NWB files")
    ap.add_argument("--layout", default="", help="with --nwb: their layout")
    ap.add_argument(
        "--phy", type=Path, nargs="+", help="Phy folders, each with events.csv beside params.py"
    )
    ap.add_argument("--plans-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    if sum(x is not None for x in (args.eid, args.nwb, args.phy)) != 1:
        ap.error("name the sessions with one of --eid, --nwb or --phy")
    if args.eid:
        sources = [Source(kind="ibl", eid=e, backend="bwm", task=args.task) for e in args.eid]
    elif args.nwb:
        if not args.layout:
            ap.error("--nwb needs --layout")
        sources = [
            Source(kind="nwb", file=str(f.resolve()), layout=args.layout, task=args.task)
            for f in args.nwb
        ]
    else:
        sources = [
            Source(
                kind="phy",
                folder=str(f.resolve()),
                events=str((f / "events.csv").resolve()),
                task=args.task,
            )
            for f in args.phy
        ]
    for s in sources:
        for path in (s.file, s.folder, s.events):
            if path and not Path(path).exists():
                raise SystemExit(f"no such file or folder: {path}")
    sessions = [dict(zip(("eid", "sha256"), source_identity(s))) for s in sources]
    directory = args.plans_dir or load_data_config().data_root / "plans"
    path = save_plan(
        directory, args.name, args.task, sessions, args.step, json.loads(args.trial_filter)
    )
    plan = read_plan(path)
    print(f"wrote {path}\nsaved at {plan['saved_at']} · sha256 {plan['hash']}")
    print(f"{len(sessions)} sessions · steps: {', '.join(plan['steps'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
