<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="unitwave/studio/static/brand/unitwave-navbar-dark.svg">
    <img src="unitwave/studio/static/brand/unitwave-navbar-light.svg" alt="UnitWave Studio: neural data, from signals to insight" width="420">
  </picture>
</p>

# UnitWave Studio

A local app for analysis after spike sorting. Load sorted Neuropixels units with
their task events, browse them by probe and brain region, and run event-aligned,
tuning, movement, quality and pair analyses, with strict statistics.

- **Data:** IBL Brain Wide Map sessions, NWB files, or a Kilosort/Phy folder plus
  a CSV of trial events.
- **Statistics:** every label ("responsive", "selective", "movement-locked",
  "putative connection") is tested against a null, corrected for the number of
  units tested, and shown with its trial counts. Plots without a label make no
  claim.
- **Local:** it runs on your machine, in your browser. Data never leaves it.

## Launch

Python 3.11, from the repository folder:

```
python3.11 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python -m unitwave.studio.server
```

Then open http://127.0.0.1:8765/ and choose data on the homepage. To skip the
homepage:

```
.venv/bin/python -m unitwave.studio.server --eid <IBL session id>
.venv/bin/python -m unitwave.studio.server --phy <Phy folder> --events events.csv
.venv/bin/python -m unitwave.studio.server --project <name>.unitwave.json
```

Data lives under `data_root` in `configs/data.yaml` (by default
`~/data/neurodecoder`, the folder's name from before the rename); the
`UNITWAVE_DATA_ROOT` environment variable overrides it. The old name
`NEURODECODER_DATA_ROOT` still works, with a deprecation warning.

## Documents

- [CLAUDE.md](CLAUDE.md): the project's rules.
- [docs/proposals/studio_next_steps.md](docs/proposals/studio_next_steps.md): what
  is built and what is planned.
- [docs/DECISIONS.md](docs/DECISIONS.md): every decision, with the evidence.
- [docs/NEGATIVE_RESULTS.md](docs/NEGATIVE_RESULTS.md): what was tried and failed,
  including the parked decoding project this repository started as.
