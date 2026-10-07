"""Build the release-wide task-period firing-rate table (see qc/task_rates.py).

    python -m unitwave.cli.build_task_rates [--workers N]
"""

import argparse
import time

from unitwave.data.load import load_data_config
from unitwave.qc.task_rates import build_task_rates


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=6, help="processes decoding shards")
    args = parser.parse_args(argv)
    config = load_data_config()
    start = time.time()
    path = build_task_rates(config.bwm_ephys_root, config.derived_root, workers=args.workers)
    print(f"wrote {path} in {time.time() - start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
