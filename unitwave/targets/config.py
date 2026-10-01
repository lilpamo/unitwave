"""The tunable part of the target definitions: trial-level decoding windows.  [R6]

Everything else about the targets is fixed in code and recorded in docs/DECISIONS.md.
TARGETS_VERSION is bumped by hand whenever a target's output changes for the same
input; the fingerprint hashes it with the windows, and every target records it.
"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from unitwave.data.session import TRIAL_TIME_FIELDS

TARGETS_VERSION = 1
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "targets.yaml"
WINDOW_TARGETS = ("block", "choice", "stimulus_side")


@dataclass(frozen=True)
class TrialWindow:
    """[anchor + start_s, anchor + stop_s], with anchor a trials event time column."""

    anchor: str
    start_s: float
    stop_s: float

    def __post_init__(self) -> None:
        if self.anchor not in TRIAL_TIME_FIELDS:
            raise ValueError(f"anchor {self.anchor!r} is not one of {TRIAL_TIME_FIELDS}")
        if not self.start_s < self.stop_s:
            raise ValueError(f"start_s {self.start_s} must be before stop_s {self.stop_s}")

    def n_bins(self, bin_ms: int) -> int:
        n = (self.stop_s - self.start_s) * 1000 / bin_ms
        if abs(n - round(n)) > 1e-9:
            raise ValueError(
                f"a {self.stop_s - self.start_s:g} s window is not a whole number of "
                f"{bin_ms} ms bins"
            )
        return round(n)


@dataclass(frozen=True)
class TargetConfig:
    windows: dict  # target name -> TrialWindow

    def __post_init__(self) -> None:
        if set(self.windows) != set(WINDOW_TARGETS):
            raise ValueError(f"windows {sorted(self.windows)}, expected {list(WINDOW_TARGETS)}")
        if not all(isinstance(w, TrialWindow) for w in self.windows.values()):
            raise ValueError("windows must be TrialWindow objects")

    def fingerprint(self) -> str:
        payload = {
            "targets_version": TARGETS_VERSION,
            "windows": {name: asdict(w) for name, w in self.windows.items()},
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def load_target_config(path: str | os.PathLike = DEFAULT_CONFIG) -> TargetConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if set(raw) != {"windows"}:
        raise ValueError(f"{path}: keys {sorted(raw)}, expected ['windows']")
    windows = {}
    for name, spec in raw["windows"].items():
        if set(spec) != {"anchor", "start_s", "stop_s"}:
            raise ValueError(f"{path}: window {name} has keys {sorted(spec)}")
        windows[name] = TrialWindow(spec["anchor"], float(spec["start_s"]), float(spec["stop_s"]))
    return TargetConfig(windows)
