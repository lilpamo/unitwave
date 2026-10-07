"""Environment variables by their UnitWave names, e.g. env("DATA_ROOT") reads
UNITWAVE_DATA_ROOT.

The NEURODECODER_* names from before the rename (docs/DECISIONS.md, "Rename:
UnitWave Studio") still work, with a one-line deprecation warning on stderr, once per
variable. When both are set, the UNITWAVE_* value wins and a line says the old one is
ignored.
"""

import os
import sys

PREFIX = "UNITWAVE_"
OLD_PREFIX = "NEURODECODER_"
_warned: set[str] = set()


def _warn_once(name: str, message: str) -> None:
    if name not in _warned:
        _warned.add(name)
        print(f"warning: {message}", file=sys.stderr)


def env(name: str, default: str | None = None) -> str | None:
    """The value of UNITWAVE_<name>, else the deprecated NEURODECODER_<name>, else default."""
    new, old = PREFIX + name, OLD_PREFIX + name
    if new in os.environ:
        if old in os.environ:
            _warn_once(old, f"{old} is ignored because {new} is set")
        return os.environ[new]
    if old in os.environ:
        _warn_once(old, f"{old} is deprecated; use {new}")
        return os.environ[old]
    return default
