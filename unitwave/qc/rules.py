"""Apply a session's own unit QC rule, whichever it is (step 8a).

Each source has its rule: IBL's label (qc.units), the Phy group (qc.phy), spike times
only (qc.spike_times) or an NWB layout's quality columns (qc.nwb). Preprocessing bins
only the units that pass the session's rule. IBL's rule takes exactly the path it always
took (qc.units.apply_unit_qc), so IBL's preprocessing and its fingerprint are unchanged
(tests/test_decoding_tasks.py).
"""

from unitwave.data.session import Capabilities, Session
from unitwave.qc.units import TASK_RATE, UnitQC, apply_unit_qc, task_firing_rates


def apply_qc(session: Session, qc) -> Session:
    """The same session with only the units that pass `qc` (and their spikes); its units
    table gains the columns the rule read."""
    if isinstance(qc, UnitQC):
        return apply_unit_qc(session, qc)
    # Imported here: the IBL path above needs none of them.
    from unitwave.qc.nwb import NwbUnitQC, nwb_unit_qc
    from unitwave.qc.phy import REFRACTORY, PhyUnitQC, phy_unit_qc, refractory_passes
    from unitwave.qc.spike_times import SpikeQC, spike_time_metrics, spike_unit_qc

    if isinstance(qc, SpikeQC):
        metrics = spike_time_metrics(session, qc)
        units = session.units.assign(**{c: metrics[c] for c in metrics})
        verdict = spike_unit_qc(metrics, qc)
    elif isinstance(qc, PhyUnitQC):
        units = session.units.assign(
            **{TASK_RATE: task_firing_rates(session), REFRACTORY: refractory_passes(session, qc)}
        )
        verdict = phy_unit_qc(units, qc)
    elif isinstance(qc, NwbUnitQC):
        units = session.units.assign(**{TASK_RATE: task_firing_rates(session)})
        verdict = nwb_unit_qc(units, qc)
    else:
        raise TypeError(f"unknown unit QC rule {type(qc).__name__}")
    keep = verdict.index[verdict["passed"].to_numpy()]
    if len(keep) == 0:
        raise ValueError(f"{session.eid}: no units pass QC")
    return Session(
        eid=session.eid,
        time_bounds=session.time_bounds,
        spikes={u: session.spikes[u] for u in keep},
        units=units.loc[keep],
        trials=session.trials,
        behaviour=dict(session.behaviour),
        available=Capabilities(
            present=session.available.present, missing=dict(session.available.missing)
        ),
    )
