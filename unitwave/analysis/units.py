"""The unit table Studio shows: one row per unit, with the repo's own unit QC verdict,
and the spike-time verdict beside it (qc.spike_times)."""

import numpy as np
import pandas as pd

from unitwave.data.session import Session
from unitwave.qc.phy import REFRACTORY, PhyUnitQC, phy_unit_qc, refractory_passes
from unitwave.qc.spike_times import (
    PRESENCE,
    SpikeQC,
    load_spike_qc_config,
    spike_time_metrics,
    spike_unit_qc,
)
from unitwave.qc.units import TASK_RATE, UnitQC, unit_qc


def unit_table(
    session: Session, qc: UnitQC | PhyUnitQC | SpikeQC, spike_qc: SpikeQC | None = None
) -> pd.DataFrame:
    """(n_units, 11): probe, region, depth_um, lateral_um, firing_rate_hz (task period),
    label, qc_passed, qc_reason, presence_ratio, spike_qc_passed, spike_qc_reason.

    Indexed by unit_id. qc is the source's rule: IBL's numeric QC label with UnitQC
    (configs/qc.yaml), the Phy group with PhyUnitQC (configs/qc_phy.yaml), or spike
    times only with SpikeQC (configs/qc_spikes.yaml), for a source without labels;
    `label` is then NaN. spike_qc: the spike-time rule shown beside it (default: the
    signed-off config; qc itself when qc is a SpikeQC). A field the session lacks is
    NaN, never filled.
    """
    spike_qc = qc if isinstance(qc, SpikeQC) else spike_qc or load_spike_qc_config()
    metrics = spike_time_metrics(session, spike_qc)
    units = session.units.assign(**{c: metrics[c] for c in metrics})
    spikes_verdict = spike_unit_qc(metrics, spike_qc)
    if isinstance(qc, SpikeQC):
        verdict, label = spikes_verdict, pd.Series(np.nan, units.index)
    elif isinstance(qc, PhyUnitQC):
        if (qc.refractory_contamination, qc.refractory_alpha) != (
            spike_qc.refractory_contamination,
            spike_qc.refractory_alpha,
        ):
            # The Phy rule's own thresholds differ from the spike-time rule's: recompute.
            units = units.assign(**{REFRACTORY: refractory_passes(session, qc)})
        verdict, label = phy_unit_qc(units, qc), units["phy_group"]
    else:
        verdict, label = unit_qc(units, qc), units["label"]
    table = pd.DataFrame(
        {
            "probe": units["probe_name"],
            "region": units.get("acronym", pd.Series(np.nan, units.index, dtype=object)),
            "depth_um": units.get("depths", pd.Series(np.nan, units.index)),
            "lateral_um": units.get("lateral_um", pd.Series(np.nan, units.index)),
            "firing_rate_hz": units[TASK_RATE],
            "label": label,
            "qc_passed": verdict["passed"],
            "qc_reason": verdict["reason"],
            PRESENCE: units[PRESENCE],
            "spike_qc_passed": spikes_verdict["passed"],
            "spike_qc_reason": spikes_verdict["reason"],
        },
        index=units.index,
    )
    assert len(table) == session.n_units
    return table
