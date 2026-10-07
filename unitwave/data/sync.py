"""Clock sync for a Phy folder's events (step 13a; docs/DECISIONS.md, "Step 13a").

A Phy folder's spike times are on the probe's clock; a lab's task events are usually on
another (the NIDQ board's, or a behaviour computer's). The same sync pulse train is
recorded on both, so pairing its edges gives the map between the clocks.

- **Pulses (`read_pulses`):** the rising edges of the sync train, in seconds on one
  clock, from
  - a CatGT edge file (`*.xd_*.txt` and the like): one time per line;
  - IBL's `_spikeglx_sync.times*.npy`, with its `.channels` and `.polarities` beside it:
    the rising fronts of the sync channel. The channel is the file's only one, or the
    `syncNiChan` of a `.nidq.meta` beside it, or named; otherwise refused.
- **The fit (`fit_clock`, the user's choices, 2026-10-05):**
  - a line from the events' clock to the probe's (offset plus drift), least squares
    over the pulses paired in order;
  - refused when the counts differ (which pulse is which can't be guessed for a regular
    square wave), and when any pulse is further than `tolerance_ms` from the line;
  - reported: offset, drift (ppm), largest and RMS residual.
- **Mapping (`ClockFit.to_probe`):** events' times to the probe's clock. Missing times
  stay missing; times more than `max_outside_s` outside the pulses are refused.
  `ClockFit.to_events` is the inverse, for a recording whose probes' spikes all move
  onto the shared events clock (step 14a).

On d23a44ef this reproduces IBL's own probe00 alignment within 0.1 ms
(tests/test_sync.py).
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "sync.yaml"
_KEYS = {"tolerance_ms", "max_outside_s"}
_IBL = re.compile(r"^_spikeglx_sync\.times(\..+)?\.npy$")


@dataclass(frozen=True)
class SyncConfig:
    tolerance_ms: float
    max_outside_s: float


def load_sync_config(path: str | os.PathLike = DEFAULT_CONFIG) -> SyncConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown, missing = sorted(set(raw) - _KEYS), sorted(_KEYS - set(raw))
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {unknown}, missing keys {missing}")
    return SyncConfig(float(raw["tolerance_ms"]), float(raw["max_outside_s"]))


def _ni_sync_channel(folder: Path) -> int | None:
    """syncNiChan from a .nidq.meta beside IBL's sync files, if exactly one is there."""
    metas = sorted(folder.glob("*.nidq.meta"))
    if len(metas) != 1:
        return None
    for line in metas[0].read_text().splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "syncNiChan":
            return int(value)
    return None


def read_pulses(path: str | os.PathLike, channel: int | None = None) -> np.ndarray:
    """(n_pulses,) rising edges of the sync train, seconds, increasing."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"no sync pulse file {path}")
    match = _IBL.match(path.name)
    if match:
        suffix = match.group(1) or ""
        times = np.load(path)
        chans = np.load(path.with_name(f"_spikeglx_sync.channels{suffix}.npy"))
        pols = np.load(path.with_name(f"_spikeglx_sync.polarities{suffix}.npy"))
        if not times.shape == chans.shape == pols.shape:
            raise ValueError(f"{path.name}: times, channels and polarities differ in length")
        found = sorted(int(c) for c in np.unique(chans))
        if channel is None:
            channel = found[0] if len(found) == 1 else _ni_sync_channel(path.parent)
        if channel is None:
            raise ValueError(
                f"{path.name} holds channels {found}: name the sync channel (or keep the "
                ".nidq.meta beside it, whose syncNiChan names it)"
            )
        if channel not in found:
            raise ValueError(f"{path.name} has no channel {channel}; it holds {found}")
        pulses = times[(chans == channel) & (pols > 0)].astype(np.float64)
    else:
        try:
            pulses = np.loadtxt(path, dtype=np.float64, ndmin=1)
        except ValueError as e:
            raise ValueError(f"{path.name}: expected one time in seconds per line ({e})") from None
    if pulses.ndim != 1 or pulses.size < 2 or not np.isfinite(pulses).all():
        raise ValueError(f"{path.name}: needs at least 2 finite pulse times")
    if np.any(np.diff(pulses) <= 0):
        raise ValueError(f"{path.name}: pulse times are not increasing")
    return pulses


@dataclass(frozen=True)
class ClockFit:
    """probe time = offset_s + (1 + drift_ppm / 1e6) * events time, fitted on paired pulses."""

    offset_s: float
    drift_ppm: float
    max_residual_ms: float
    rms_residual_ms: float
    n_pulses: int
    span: tuple[float, float]  # first and last pulse, events' clock
    max_outside_s: float

    def to_probe(self, times) -> np.ndarray:
        """(n,) events' times on the probe's clock; NaN stays NaN. Refuses times more
        than max_outside_s outside the pulses."""
        t = np.asarray(times, np.float64)
        outside = np.isfinite(t) & (
            (t < self.span[0] - self.max_outside_s) | (t > self.span[1] + self.max_outside_s)
        )
        if outside.any():
            raise ValueError(
                f"{int(outside.sum())} event times lie more than {self.max_outside_s:g} s outside "
                f"the sync pulses ({self.span[0]:.1f}–{self.span[1]:.1f} s on the events' clock), "
                "where the clock fit isn't checked"
            )
        return self.offset_s + (1.0 + self.drift_ppm * 1e-6) * t

    def to_events(self, times) -> tuple[np.ndarray, np.ndarray]:
        """Probe-clock times on the events' clock (the inverse line), and (n,) which lie
        within max_outside_s of the pulses: (mapped, inside). The caller decides what
        to do with the rest (a recording drops and counts spikes beyond the pulses)."""
        t = np.asarray(times, np.float64)
        mapped = (t - self.offset_s) / (1.0 + self.drift_ppm * 1e-6)
        inside = (mapped >= self.span[0] - self.max_outside_s) & (
            mapped <= self.span[1] + self.max_outside_s
        )
        return mapped, inside

    def describe(self) -> str:
        return (
            f"{self.n_pulses} sync pulses · offset {self.offset_s * 1000:.3f} ms · drift "
            f"{self.drift_ppm:.1f} ppm · largest residual {self.max_residual_ms:.3f} ms "
            f"(RMS {self.rms_residual_ms:.3f} ms)"
        )


def fit_clock(probe_pulses, events_pulses, cfg: SyncConfig) -> ClockFit:
    """The line from the events' clock to the probe's, over pulses paired in order.

    probe_pulses, events_pulses: (n_pulses,) the same pulses on each clock."""
    probe = np.asarray(probe_pulses, np.float64)
    events = np.asarray(events_pulses, np.float64)
    assert probe.ndim == events.ndim == 1
    if probe.size != events.size:
        raise ValueError(
            f"{probe.size} pulses on the probe's clock and {events.size} on the events' clock: "
            "which pulse is which can't be guessed. Trim both files to the same pulses (or "
            "re-extract them) and try again"
        )
    slope, offset = np.polyfit(events, probe, 1)
    residual = probe - (offset + slope * events)
    fit = ClockFit(
        offset_s=float(offset),
        drift_ppm=float((slope - 1.0) * 1e6),
        max_residual_ms=float(np.abs(residual).max() * 1000),
        rms_residual_ms=float(np.sqrt(np.mean(residual**2)) * 1000),
        n_pulses=int(probe.size),
        span=(float(events.min()), float(events.max())),
        max_outside_s=cfg.max_outside_s,
    )
    if fit.max_residual_ms > cfg.tolerance_ms:
        raise ValueError(
            f"the clocks don't follow a line: largest residual {fit.max_residual_ms:.3f} ms, "
            f"above the {cfg.tolerance_ms:g} ms tolerance (configs/sync.yaml). A pulse may be "
            "missing from one file, or the clock drifted unevenly"
        )
    return fit
