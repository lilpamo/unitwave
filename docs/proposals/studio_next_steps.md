# UnitWave Studio: next steps

Proposal, 2026-09-30. Follows the prototype on `studio-prototype`
(`docs/DECISIONS.md`, "Direction change: Neurodecoder Studio"). Each step keeps
the prototype's rule: the UI computes nothing itself. It loads through `data/`
and `qc/`, gets every number from `unitwave/analysis/` and draws through
`viz/`.

## Current sequence (2026-10-01, second): other datasets and tasks

The user's plan from here (`docs/DECISIONS.md`, "Next: other datasets and tasks"):
UnitWave on any task, proven dataset by dataset rather than through an abstract
universal format. It replaces every step of the S1–S8 sequence below that hadn't
started:
- **Replaced steps:** S6 (unit browsing) moves to "plan only", S7 becomes step 3,
  and S8 becomes step 9.
- **Kept:** S3–S5 (decoding, trajectories, region summaries) were built against
  IBL's targets and are kept (the user, 2026-10-01). Step 8 rewrites them against
  task definitions.

**After each step, and rules for every step:** as for S1–S8 below, plus:
- **Ask first** before changing `preprocess/`, `splits/` or any signed-off QC
  default (R6).
- **No synthetic data shown as real:** test fixtures stay in tests.
- **Before downloading any dataset:** list the files and sizes and wait for the
  user's OK.

### Step 0. Plan (docs only) — built (2026-10-01)

This section, and the DECISIONS.md entry.

### Step 1. Robustness pass on varied IBL sessions — built as S1 (2026-10-01)

### Step 2. Prior-art audit (docs only) — built as S2 (2026-10-01); one addition pending

Pending: a column in `docs/PRIOR_ART.md` §F's matrix, "works with tasks and
datasets beyond one lab".

### Step 3. User-defined tasks — planned

Today, events and conditions are hard-coded to IBL's trial columns.
- **A task definition file** per task, YAML in `configs/tasks/`. It declares:
  - **events:** the time columns, with labels;
  - **conditions:** columns, type (categorical, ordinal or continuous), levels
    and labels, and the derivations needed (e.g. IBL's signed contrast from two
    columns, with its NaN convention);
  - **comparisons:** which two-level comparisons are offered;
  - **nulls:** the null for each comparison, as a permutation with its strata
    declared. Pseudo-sessions apply only where a block generator is known, today
    only IBL;
  - **trial-structure variables:** what the no-spikes baseline (null_trialstruct)
    may use, for decoding later;
  - **behaviour:** the time series it uses, mapped from the file's own names;
  - **trial filters:** the filters it offers.
- **IBL becomes one built-in definition.** `analysis/events.py`, `conditions.py`,
  `tuning.py`, `movement.py` and `trial_view.py` read definitions, with no
  behaviour change.
- **Choosing a definition:** Phy sessions and NWB files choose one when opened. A
  definition whose columns are missing is refused in plain language, naming the
  columns.
- **Names:** the UI shows the task's own event and condition names everywhere,
  including captions, exports and the project file.
- **Tests first:**
  - every existing test passes unchanged with the IBL definition;
  - d23a44ef gives identical numbers before and after: PSTHs, split PSTHs, tuning,
    responsiveness, selectivity, movement locking. The comparison is reported;
  - a hand-built non-IBL trials table with its own definition works end to end;
  - malformed definitions are refused.

### Step 4. QC from spike times, for any source — planned

Some sources have no QC labels (e.g. Steinmetz 2019 has no quality metrics).
- **A source-independent rule, from spike times only:**
  - **Metrics:** task-period firing rate, the existing refractory-violation metric
    and presence ratio, with thresholds in `configs/`.
  - **Amplitudes:** spike amplitudes aren't stored, so amplitude-based metrics are
    declared unavailable.
- **Sources with their own labels keep them:** the IBL label, Phy group or NWB
  quality columns. Both verdicts are shown side by side.
- **The default rule per source goes in `configs/`.** IBL's signed-off default is
  not changed. The report says how far the IBL label and spike-time QC agree on the
  robustness-pass sessions, for the user to decide.
- **Tests:**
  - each metric against a hand-computed spike train;
  - a source with no labels uses the spike-time rule;
  - agreement reporting.

### Step 5. General NWB intake, proven on Steinmetz et al. 2019 (DANDI 000017) — planned

- **Read a declared subset of NWB:** the units table (spike times, and quality
  columns if present), the trials table, behaviour time series, and unit or
  electrode locations if present.
- **The capability report (`nwb/probe.py`):** each file states what it supports.
  Anything outside the subset is refused with the reason, never guessed.
- **A Steinmetz task definition.**
- **Regions:**
  - **Names:** from the file's own location names.
  - **3D view and levels:** the 3D view and the Allen/Beryl/Cosmos levels are
    enabled only when the names or coordinates are Allen CCF, and disabled with the
    reason otherwise.
- **The proof:** one Steinmetz session end to end, from homepage to open; every
  view works or refuses with a stated reason, with screenshots of each.
- **Tests first:**
  - a small NWB file written with pynwb inside the test, mapped field by field;
  - the capability report on it;
  - refusals for an unsupported layout.

### Step 6. Allen Brain Observatory Visual Coding (Neuropixels) — planned

A passive task: stimulus presentations instead of decision trials, and many
stimulus conditions (e.g. orientation, spatial frequency).
- **Find the source:** verify which DANDI dandiset or source holds it, and propose
  one session with sizes. Wait for the user's OK before downloading.
- **Its task definition,** with stimulus presentations playing the role of trials.
- **Tuning:** curves over many levels. An orientation-selectivity style summary
  only if it carries a null and correction.
- **No choice:** everything without a choice must work, or be refused with its
  reason (selectivity on choice, movement controls).
- **The proof:** as in step 5, every view, with screenshots.

### Step 7. A dataset without the Allen mouse atlas — planned

- **Choose the dataset:** propose 2–3 candidates (e.g. rat or monkey recordings on
  DANDI with a trials table), with sizes and why each is a good test. Wait for the
  user's choice.
- **The proof:** everything works without an atlas, and atlas-only features are
  disabled with the reason.

### Step 8. Decoding, trajectories and region summaries, for any task — planned

The earlier specs, written against task definitions from the start. Each is its own
step, in this order, proven on IBL plus at least one other dataset:
- **Decoding:**
  - the six-row contract;
  - the no-spikes baseline uses the task definition's trial-structure variables;
  - splits come from the registry, never random time points;
  - balanced accuracy and AUROC.
- **Population trajectories:** PCA fit on odd trials and projected on even ones,
  with PC names only (R5).
- **Region summaries:** only for datasets with regions. Sessions are the unit of
  inference, with FDR across regions.

### Step 9. Guided workflow recipes (no AI) — planned

As specified in S8 below, with three changes:
- **Each recipe declares the task features it needs** (events, condition types).
  It is offered only where the task has them, with the reason shown otherwise.
- **Wording first:** three recipes' wording is drafted for the user's approval
  before the UI is built.
- **An analysis log** in the project file.

### Plan only, don't build yet

What each needs is listed under "Not to build yet" below:
- **(a)** Phy clock sync and channel locations;
- **(b)** the installer;
- **(c)** an AI layer over recipes, bound by CLAUDE.md §6;
- **(d)** unit browsing extras.

## Earlier sequence (2026-10-01): S1–S8 — S1–S5 built, S6–S8 replaced

The user's plan from here. The steps are named S1–S8 so they don't clash with the
numbered plan steps below; each says which plan step it covers.

**After each step:**
- run the full test suite, and preview UI steps in the browser pane with
  screenshots;
- mark the step built here, with any differences from the plan, and record it in
  DECISIONS.md;
- stage, report and stop.

On "commit it, go on", commit and start the next step. A step never starts with
the previous one uncommitted.

**Rules for every step:**
- Tests first where the contract is clear. Array shapes are in docstrings and
  asserted.
- About 400 new lines at most before running something.
- The UI computes nothing: numbers come from `analysis/`, figures from `viz/`.
- Every label is a claim, with a null and a multiple-testing correction (R4, §5).
  Descriptive views say they make no claim.
- Missing data is excluded and counted, never filled in.
- Every new dependency gets a DECISIONS.md line.
- A step needing changes to `preprocess/` or `splits/` stops and asks first (§10).

### S1. Robustness pass (no new features) — built (2026-10-01)

Built (`docs/DECISIONS.md`, "S1 robustness pass"). Eight sessions, every view, no
failures left. Differences from the plan:
- **Not available in the release:** a session without wheel (all 459 have it),
  and two shanks of one Neuropixels 2.0 probe (one shank per session at most).
  3a3ea015 covers a single NP2.0 shank.
- **Found and fixed, each with a test from the real case:**
  - the connection test never finished on fast units: over 30 minutes for one
    region of 6a601cc5, now 37 s in the page;
  - one failed mesh download hid every region mesh, and a download cut short
    left the page an empty response;
  - the heatmap's unit axis had fractional ticks with 2 units.
- **Over 2 s on the largest session (dd4da095), not changed:**
  - movement locking 22.2 s;
  - responsiveness 12.5 s, and 11.6 s on movement-free trials;
  - export 8.5 s.
- **Browser checks:** the browser found the mesh bugs, which the scripted
  requests did not.

**The plan:**

- **Sessions:** about 8 BWM sessions from the catalog that differ from d23a44ef:
  - one probe only;
  - a Neuropixels 2.0 multi-shank probe, if the release has one;
  - missing pose or wheel, and missing events;
  - a cortical-only session and a deep-structure-only session;
  - the largest and smallest unit counts.
- **Exercise every view and test,** each session opened through the homepage:
  - the unit table, rasters and PSTHs, split PSTHs and tuning;
  - responsiveness, selectivity and movement controls;
  - the trial view, unit quality and correlograms;
  - the 3D view and probe strip;
  - project save and reopen, and export.
- **Fixes:** each bug gets a test first, built from the real case that exposed it.
- **Report:** a table of session, what was tried, what failed and the fix. Each
  view is timed on the largest session, listing anything slower than 2 s.

### S2. Prior-art audit (docs only) — built (2026-10-01)

Built: `docs/PRIOR_ART.md` §F, appended after the existing text, which is
unchanged. Differences from the plan:
- **Checked from sources, not run:** repositories, docs, source files, and PyPI
  and GitHub metadata. Each claim is marked [V] or [A].
- **Newer tools found:**
  - NeuroPyGuiN (2026), a desktop app from sorting to histology with PSTHs and
    connection tests;
  - NeuroPyxels, its statistics library;
  - Power Pixels, Spyglass and nwbwidgets, noted but not audited.
- **Urchin was archived on 2026-08-04.**
- **Finding:** none of the GUIs checked corrects its labels across the units or
  pairs tested. CellExplorer corrects across lag bins only. NeuroExplorer and
  NeuroPyxels/NeuroPyGuiN don't correct. Elephant's SPADE uses BH, but it is a
  library.

**The plan:**

- **Where:** a section "F. Post-sorting analysis apps" in `docs/PRIOR_ART.md`.
- **Verification:** each tool is checked against its current docs or repo and
  marked [V] verified or [A] assumed.
- **Tools:**
  - CellExplorer, NeuroExplorer, IBL's public data viewer, Neurosift;
  - Phy, the SpikeInterface GUI, sortingview;
  - Pynapple (and any GUI it now has), Elephant, brainrender, Urchin;
  - any newer tool found.
- **For each:** what it does, data formats, statistical tests with nulls and
  multiple-testing correction, atlas and 3D support, reproducibility features,
  licence and language.
- **Then:** a comparison matrix, and an honest novelty assessment for UnitWave
  Studio, like the existing sections.

### S3. Decoding in Studio (plan step 10) — built (2026-10-01)

Built (`docs/DECISIONS.md`, "S3 decoding in Studio"). Differences from the plan,
the first three signed off in chat:
- **Verdicts are within-session.** The model is ranked among 100 shifted-target
  refits (and the pseudo-sessions for block). A paired bootstrap over test trials
  compares it with null_trialstruct and baseline_ridge, with BH across the
  comparisons. The contract's across-session test needs 5 sessions.
- **Stimulus side is a new target:** 0–100 ms after stimulus onset, with 0%
  contrast excluded and counted. Its trial-structure null includes the block
  prior, which predicts the side well.
- **IBL sessions only.** Phy folders need a single-session split in `splits/`.
- **baseline_rrr is not run;** the page says why. ceiling_within and
  baseline_ridge are the model itself, and are marked so.
- **Run times on one session:** choice 2–3 min, stimulus side 1–2 min, movement
  state about 45–60 min, block 8 min. Runs go in the background with progress.

**The plan:**

Single-session decoding in the GUI, with the parked decoding code (`splits/`,
`evaluation/`, `models/baselines/`). The modules touched are stated before
starting.
- **Targets:** choice, stimulus side, block and movement state, from the
  filtered units (probe, region node, QC).
- **The §5 decoding contract:** every row that applies to one session is
  reported, and the page says which rows don't apply and why (`baseline_rrr` is
  multi-session).
- **Block:** uses the leave-one-block-out split and the pseudo-session null.
- **Splits:** splits come from the registry (R1), never random time points (R2).
  Normalisation is fit on training data only (R3).
- **Plain verdicts:** the page says plainly when a result does not beat
  `null_trialstruct` ("not decoding") or `baseline_ridge`.
- **Class imbalance:** balanced accuracy and AUROC, never raw accuracy alone.
- **Runs:** logged to `runs/` with the §7 manifest. Long runs show progress and
  don't block the other views.
- **Test first:** the GUI path gives the same table as the CLI path on the same
  session and config.

### S4. Population trajectories (plan step 9) — built (2026-10-01)

The engine, the cross-validation and both tests are already in place: plan step 9
below.
- **Built as specified:**
  - PCA on condition-split PSTHs of the filtered units;
  - normalisation and components fit on odd trials, with even trials projected;
  - 2D and 3D views with the event marked;
  - tests for planted low-dimensional structure and for a fit that never touches
    the displayed half.
- **Finished in S4:**
  - Each component's share of the shown trials' variance is on its axis, as
    `pc_1 (58%)`; the legend says what the percentage is. Names stay `pc_k` (R5).
  - The caption gives both trial counts ("fit on 146 trials (1st, 3rd, ...) and
    shown on the other 144"), and the legend still gives them per condition.
- **Found on the way:** the legend's title was drawn black, unreadable on the dark
  page; it now uses the theme's ink.

### S5. Across-session region summaries (plan step 12) — built (2026-10-01)

Built (`docs/DECISIONS.md`, "S5 region summaries"). Differences from the plan:
- **The null:** region labels permuted within each session, computed exactly, as
  the sum of each session's hypergeometric count; no seed. A mixed model wasn't
  used.
- **Labels:** responsive, selective or movement-locked, the same tests Studio
  runs, on the set's own trial filter.
- **Minimum:** 5 sessions per region, signed off in chat. With 8 CA1-rich
  sessions, only CA1 reached it.
- **Homepage:** the card shows finished runs and gives the command for a chosen
  set. Runs aren't started from the page, because they are slow.

**The plan:**

The engine goes in `analysis/summary.py`, with a CLI entry point because it is
slow. The input is a session set from the homepage.
- **Per region,** at the chosen level (Beryl by default): the fraction of
  responsive, selective or movement-locked units.
- **Never only the pooled fraction (§5):** the per-session distribution is always
  shown, with n sessions and n units per region.
- **The null for region-level claims:** sessions are the unit of inference. Either
  region labels permuted across units within each session, or a mixed model with
  session as a random effect; DECISIONS.md records which.
- **Correction:** FDR across regions.
- **Minimum:** regions below a minimum number of sessions (in `configs/`) are
  refused, with the reason shown.
- **A Swanson flatmap,** from iblatlas.
- **Results:** written to `runs/<run_id>/` with a manifest (git SHA, config hashes,
  the session set and its hash). The app reads them.
- **Tests first:**
  - aggregation checked against a hand-built two-session example;
  - the per-session distribution is present;
  - a region below the minimum is refused;
  - the region null stays calibrated when units are pooled unevenly across
    sessions;
  - re-running gives identical results.

### S6. Unit browsing (plan step 11) — replaced: its extras are plan only, (d) below

- **Search:** units by id or region.
- **Keyboard:** up/down arrows move through the unit table and update every view.
- **Compare:** pin a unit, and compare two units side by side.
- **Tags and notes** per unit, saved in the project file. They are user input,
  not results, and the project module's docstring says so.
- **Tests first:**
  - search;
  - keyboard order follows the table's sort;
  - a project round trip keeps pins and tags;
  - switching session clears pins.

### S7. User-defined tasks — replaced by step 3 of the current sequence

Today, events and conditions are hard-coded to IBL's trial columns, so other labs
must rename their columns to IBL's.
- **A task definition file** per task, YAML in `configs/tasks/`. It declares:
  - events: the trials-table columns holding times, with labels;
  - conditions: columns, type (categorical, ordinal or continuous), levels and
    labels;
  - which two-level comparisons are offered;
  - the null for each comparison: a permutation, with strata declared in the file.
    The pseudo-session null exists only for IBL's block generator.
- **IBL becomes one built-in definition.** `analysis/events.py` and
  `analysis/conditions.py` are refactored to read definitions, with no behaviour
  change.
- **Choosing a definition:** a Phy session, or an NWB file with a trials table,
  picks its task definition when opened. A definition whose columns are missing is
  refused, naming the columns.
- **Tests first:**
  - with the IBL definition, every existing test passes unchanged, and d23a44ef
    gives identical numbers before and after (PSTHs, responsiveness, selectivity);
  - a hand-built non-IBL trials table with its own definition works end to end;
  - malformed and missing-column definitions are refused in plain language.

### S8. Guided workflow recipes (no AI) — replaced by step 9 of the current sequence

A recipe is a sequence of analyses that answers one scientific question. Recipes
are files in the repo (YAML plus Markdown), one per question.
- **Each step in a recipe states:**
  - the question it answers;
  - why this analysis answers it, in plain language;
  - which analysis and parameters;
  - the null and correction it uses;
  - its prerequisites;
  - how to read the result, including what it would NOT show.
- **Three recipes for the IBL task.** Their wording is drafted for the user's
  approval before the UI is built:
  - "Which units respond to the stimulus beyond the movement that follows it?"
    (responsiveness → movement controls → late-movement-only test);
  - "Does activity carry choice information beyond the stimulus?" (split PSTHs →
    conditional selectivity → decoding vs `null_trialstruct`);
  - "How do regions differ in their responses?" (region summaries across a
    session set).
- **The UI:** a Recipes panel. Each step shows its rationale and a "Run analysis"
  button that calls the same engine as the manual views. Results show the null,
  `n_tests` and trial counts.
- **An analysis log,** designed in DECISIONS.md:
  - the project file records every test run in the project (what, when,
    `n_tests`), and the app shows the running total;
  - results are marked exploratory unless they ran on a held-out set of sessions
    named before they were run.
- **Tests first:**
  - a recipe runs the same functions with the same parameters as the manual
    views, with identical numbers;
  - the analysis log counts every test;
  - a malformed recipe is refused.

### Not to build yet: what each would need

- **(a) Phy clock sync and channel-location import (plan step 13):**
  - a sync source shared by the probe and the behaviour (a TTL or sync channel, or
    IBL-style sync files);
  - a fitted clock mapping between them, with its residuals reported and a
    refusal above a tolerance;
  - channel locations from a histology or alignment file, giving regions per site
    and so per unit;
  - tests on a hand-built drifting clock, and on a known alignment.
- **(b) The installer (plan step 14), and safe use by colleagues:**
  - one command that installs a pinned environment and the app, with three.js
    already in the repo and the Allen meshes fetched once;
  - a launcher that serves on 127.0.0.1 only and opens the browser;
  - a first-run data-folder setup;
  - documentation that colleagues each run their own copy, rather than anyone
    exposing the server: it has no authentication, by design.
- **(c) An AI layer over recipes, bound by CLAUDE.md §6:**
  - it reads result files only, and may propose and explain analyses and write
    prose;
  - every number it writes traces to a result file, enforced by a test like
    `tests/test_agent_no_invention.py`;
  - it never runs a test the analysis log doesn't count;
  - **a decision first:** a hosted model would send results off the machine, which
    the local-only rule in CLAUDE.md §2 forbids unless a decision records otherwise
    (a local model, or explicit consent per use).
- **(d) Unit browsing extras (from S6):**
  - search units by id or region;
  - up/down arrows move through the unit table and update every view;
  - pin a unit, and compare two units side by side;
  - tags and notes per unit, saved in the project file, which are user input, not
    results;
  - tests: search, keyboard order following the table's sort, a project round trip
    keeping pins and tags, and switching session clearing pins.

## 1. Import Kilosort / Phy folders — done (2026-09-30)

Built as planned, with these differences (`docs/DECISIONS.md`, "Phy import and
Phy QC"):
- **Code:** `data/backends/phy.py` (`load_session_phy`), `qc/phy.py` and
  `configs/qc_phy.yaml`. The group is Phy's `good`/`mua`/`noise` label, kept as
  `phy_group`, with `group_file` recording where it came from.
- **Group:** read from `cluster_group.tsv`, else `cluster_KSLabel.tsv`, else
  missing. `cluster_info.tsv` is not read.
- **Depth:** the y position of the peak channel of the cluster's most-used
  template, not an amplitude-weighted position. It is declared missing without
  the template files.
- **Events CSV:** canonical trial column names, and it must include
  `intervals_0` and `intervals_1`. Events outside the span of the recorded
  spikes are refused.
- **Phy QC:** group in `[good]` and a task-period rate of at least 0.1 Hz. IBL's
  sliding refractory-period test was added later (see step 3).
- **Opening data:** through server flags (`--phy FOLDER --events CSV`). The
  folder-picker question is still open.

## 2. Atlas and 3D view — built (2026-09-30)

Built as planned below (`docs/DECISIONS.md`, "Atlas and 3D view, built"), with no
chart library. Two things differ from the plan:
- At Beryl, `root` also holds units that IBL labelled only with a coarse parent
  region (MY, CB, TH), not just fibre tracts.
- The probe strip's region runs span recorded units, so they are not histological
  boundaries.


**What:** where each unit is, at the region level the user picks, in a 3D brain
and along the probe. Plus a visual redesign of the app.

**Region level** (`analysis/atlas.py`):
- A selector for Allen, Beryl or Cosmos. **Default: Beryl.**
- Names, colours and hierarchy come from `iblatlas.regions.BrainRegions`, already
  a dependency: `acronym2acronym(..., mapping=...)`, `rgb`, and parents and
  descendants.
- Remapping is shown, never hidden. At Beryl and Cosmos, fibre tracts and some
  nuclei map to `root` (in d23a44ef, `ml` becomes `root`). Those units are
  labelled "no Beryl region" and counted, not dropped.
- Unit QC keeps using the Allen acronym, so QC doesn't change with the display
  level (R6).

**Hierarchical region filter:**
- A tree built from the iblatlas hierarchy, trimmed to the regions this session
  has units in, with unit counts per node.
- Selecting a node includes its descendants. The heatmap title names the node
  and its count.

**3D brain** (`studio/static/`, three.js):
- The whole-brain outline plus the meshes of the regions present, in their Allen
  colours.
- Each probe track, fitted to its channel or unit positions.
- The selected unit's site as a marker. Clicking a site selects that unit.
- **Coordinates:** IBL `x, y, z` (metres, from bregma) are converted to CCF µm
  in `analysis/atlas.py` with iblatlas's own landmarks, not with constants
  copied into the code. The page receives CCF coordinates and computes none.
- **Meshes:** the Allen CCF 2017 structure meshes (`.obj`, one per structure id)
  from the Allen Institute's download server. They are downloaded once, on
  first use, into `data_root/atlas/ccf_2017_meshes/`, never into the repo, and
  their sizes are listed before downloading.
- **three.js:** kept in the repo as a local file, with no CDN, so the app works
  offline. It is loaded as an ES module with no build step.

**2D probe strip:**
- The probe's channel map (`lateral_um`, `axial_um`, or `channel_positions.npy`
  for Phy), with each unit at its site, coloured by region at the chosen level.
- Region boundaries are marked along the shank. The selected unit is
  highlighted, and clicking a unit selects it.

**Phy data:** a Phy folder has no brain position, so the 3D view and region
filter are disabled with that reason. The probe strip still works from
`channel_positions.npy`. Importing channel locations (e.g. IBL's alignment
output) is a later step.

**Visual redesign:**
- **Colours:** one set of colour tokens, in light and dark, and region colours
  from Allen.
- **Layout:** a left rail (data source, QC toggle, level selector, region tree),
  the unit table, the plots, and a 3D/probe panel.
- **Plots:** restyled to match the rest of the app.
- **Readability:** trial counts and exclusions always visible.

**Chart library, a decision at the start of this step:** the PNG plots can't do
hover or zoom.
- **Recommended:** keep matplotlib, and make heatmap rows clickable through a
  row→unit map, with no chart library.
- **If hover and zoom on rasters are wanted:** uPlot, which is small, canvas
  based and MIT licensed. It's preferred over ECharts, which is large, and
  Plotly, which is excluded by CLAUDE.md §8.
- Whichever is chosen gets its own DECISIONS.md entry, like three.js.

**Tests first:**
- **Remapping:** Allen → Beryl → Cosmos checked against hand-picked acronyms
  (`CA1`→`CA1`→`HPF`, `DG-mo`→`DG`→`HPF`, `ml`→`root`).
- **Tree:** unit counts per node sum correctly over descendants.
- **Coordinates:** for d23a44ef, each unit's CCF position lies inside its
  region's voxels in the Allen annotation volume, for all units but a
  documented, small number. This needs iblatlas's annotation volume, a one-time
  download.
- **Track fit:** a straight probe gives a track through its sites.

## 3. Responsiveness against a shuffle null — built (2026-09-30)

Built (`docs/DECISIONS.md`, "Responsiveness against a shift null"), with one change
to the plan below: instead of 1,000 seeded random shifts, **every** circular shift
on a 5 ms grid is evaluated by FFT, so the test is deterministic and p is not
floored near 0.001. The refractory-period metric for Phy QC was added afterwards: a
port of ibllib's MIT `slidingRP_viol` (`docs/DECISIONS.md`, "Phy QC gains IBL's sliding
refractory-period test").


**What:** per unit and event, a p-value for "the rate in a response window
differs from the rate in a baseline window".

**Statistic:** mean over trials of (response rate − baseline rate).

**Null:** circularly shift each trial's event time by a random offset, keeping
trial structure and the spike train's autocorrelation. This is R4's
`null_shuffle` spirit, not the evaluation module's code.
- n shifts and the minimum shift go in `configs/analysis.yaml`.
- The seed is explicit (R7).
- p = (1 + #null ≥ observed) / (1 + n).

**Correction:** Benjamini–Hochberg across all units tested in one call, with the
number of tests shown next to every "responsive" label.

**Cross-validated heatmap sorting:** find peak times on odd trials and display
even trials. This fixes the prototype's known circularity.

**Tests first:**
- **Hand-computed case:** a unit that fires exactly 1 spike after every event
  has p = 1/(n+1).
- **Calibration:** Poisson spike trains with no event locking. p is uniform (KS
  test), and BH's false discovery rate stays at or below α over many seeds. This
  is test input, never shown as data.
- **Seed:** a fixed seed reproduces p exactly.

**UI:** a "responsive" column in the unit table, with p and q values. The
population heatmap gains a "responsive only" filter.

**Phy QC:** a refractory-period metric was added (IBL's sliding RP test, MIT port).
Phy QC now passes 161 units on d23a44ef probe00, down from 200; IBL's label passes
114, and 101 pass both.

## 4. Project file and figure export — built (2026-09-30)

Built as planned (`docs/DECISIONS.md`, "Project files and figure export"):
- Opening, saving and exporting use command-line paths (`--project`) rather than a
  folder picker.
- Export writes SVG, PDF, JSON sidecars and a manifest to `runs/<time>_studio/`.


**Project file:** `*.ndstudio.json` (`*.unitwave.json` since the rename; old files
still open), a plain JSON file that holds:
- the data source (backend and eid, or Phy path), with a hash of the spike files;
- the QC config hash;
- the selected units and filters, including the region level and tree
  selection;
- each analysis's parameters: event, window, bin, baseline, seed.

It holds no results. Reopening it recomputes everything, so a project can never
show numbers that disagree with the data. If the hashes changed, it warns and
names the file.

**Export:**
- Each figure as SVG and PDF, from the same `viz/` functions at a set size and
  font.
- A sidecar `.json` next to each figure, with every number plotted: bin centres,
  mean, SEM, n_trials, n_excluded, unit ids, and the parameters. Anything a
  figure shows is then traceable to a file (§6's spirit).
- Figures and sidecars go to `runs/<run_id>/` with the manifest §7 already asks
  for: git SHA, config hash and seed.

**Tests:**
- Round trip: save a project, reload it, and get identical PSTH arrays.
- A changed spike file triggers the hash warning.
- The exported sidecar equals the engine's output.

## Further steps: 4b, 5–6, 6b and 7–14 (each marked planned or built)

Each step says:
- what it adds;
- the null it needs if it labels units (R4: a label such as "tuned" or
  "connected" is a claim, and ships with a null and a multiple-testing correction
  over the units tested);
- what Phy data lacks for it;
- its tests.

As in steps 1–4, the engine goes in `analysis/`, the UI draws only, and a test is
written first where the contract is clear.

## 4b. Homepage and data selection — built (2026-09-30)

Added before step 5, as **4b** so the later steps keep their numbers.

**Part (a) as built** (`docs/DECISIONS.md`, "Homepage and data selection, part
(a)"), with these differences from the plan below:
- The manifest went to version 2, with `region_units` plus motion-energy and pupil
  modalities; it is stored in `derived/manifest-v2/`.
- The default trial filter is `bwm_include` plus excluding no-go, set in
  `configs/catalog.yaml`. It also applies to sessions opened with `--eid` or
  `--phy`.

**Part (b) as built** (`docs/DECISIONS.md`, "Homepage and data selection, part
(b)"), with these differences from the plan below:
- Labs past the palette's eight colours share one muted "other labs" colour.
- Only a click that barely moved opens a session from the 3D view, so a drag
  never does.
- The Phy root is `data_root/phy` (`configs/catalog.yaml`).

**Part (a): the homepage, filters, trial filters and opening a session.**
- **Starting up:** the server starts with no session and serves a homepage at
  `/`. Opening a session is a POST behind the existing Origin and Content-Type
  checks. It loads through `load_session` with a visible loading state, then
  shows the session view, with a Home link back. `--eid`, `--phy` and
  `--project` still skip the homepage.
- **Switching session** resets the unit selection, caches and test results.
- **Session list:** built from the BWM manifest (`data/manifest.py`), written once
  under `derived/` if no copy exists. Filtering and counts live in
  `analysis/catalog.py`, not the page. It shows a sortable table (lab, subject,
  date, probes, units, trials, included trials, Beryl regions) and marks
  sessions already in the local cache.
- **Session filters,** with live counts ("N sessions, M probes, K units match"):
  - lab, subject and date range;
  - a region from the iblatlas tree, with descendants and a minimum number of
    units in configs;
  - minimum good units, minimum included trials, and number of probes;
  - behaviour modalities.
- **Unit counts are labelled** as the release's good units, not Studio's QC
  count. For d23a44ef, Studio QC passes 390 of the release's 398.
- **Trial filters** (per trial: every BWM session runs the same task):
  - `bwm_include`, contrasts, block, outcome, and excluding no-go trials.
  - They are built on step 5's `analysis/conditions.py` definitions, so there is
    one definition only.
  - Excluded trials are counted, and every caption gives the trial count after
    filtering.
  - The filters are part of the responsiveness and selectivity cache keys, the
    project file and the export records.
  - Phy sessions offer only what their events CSV has, with the reason shown for
    the rest.
- **Tests:**
  - each session filter on a hand-built manifest;
  - region counts with descendants;
  - trial-filter counts, with excluded trials counted;
  - a test result is never shown under another trial filter;
  - a project round trip with trial filters;
  - no session at start, old flags skip the homepage, and switching resets state.

**Part (b): 3D overview, session sets, Phy folders, recent projects.**
- **3D overview:** every matching probe drawn from the manifest's tip and top
  positions, converted to CCF by `analysis/atlas.py`, coloured by lab. Hovering
  names the session and probe; clicking opens it. It updates with the filters.
- **Session sets:** named JSON files holding eids, the manifest version, the
  filters used and a hash. Reopening one warns if the manifest version changed.
  Step 12 will use them. The session view still holds one session at a time.
- **"Open a Phy folder":** a path field completing against a configured Phy root,
  pairing each folder with the `events.csv` beside its `params.py`. It refuses
  paths outside the root, traversal and symlinks out, and missing files are
  refused in plain language.
- **Recent projects** from `data_root/projects`, newest first.
- **Tests:**
  - probe lines converted exactly as unit positions are;
  - a session-set round trip, and the warning on a changed manifest version;
  - Phy path refusals;
  - the recent-projects list matches the folder.

## 5. Condition-split PSTHs and tuning — built (2026-09-30)

Built (`docs/DECISIONS.md`, "Condition-split PSTHs, tuning curves and
selectivity").
- **As planned:** choice is permuted within signed-contrast strata, and block
  uses pseudo-sessions.
- **Also stratified:** side within choice, and outcome within signed contrast.
- **Strata:** 0% contrast is split by side.
- **Block window:** block uses the pre-event window.
- **Validity:** the pseudo-session null is valid on average over block
  sequences, not for every single session.


**Adds:**
- **Condition-split PSTHs:** PSTHs split by a task variable (stimulus side, signed
  contrast, choice, feedback, block prior), overlaid, with a trial count per
  condition.
- **Tuning curves:** response-window rate against signed contrast, mean ± SEM
  per level.
- **Where:** `analysis/conditions.py` and `analysis/tuning.py`.

**Null, for "selective" or "tuned" labels:**
- **Why a plain shuffle fails:** IBL's task variables are correlated. Choice
  follows stimulus side, and the block prior predicts side. So a plain label
  shuffle calls a purely stimulus-driven unit "choice-selective".
- **Choice and side:** a conditional permutation, shuffling labels only within
  strata of the other variables (the condition-combined test IBL's Brain Wide Map
  used).
- **Block:** the pseudo-session null (`evaluation/nulls.py` already has a seeded
  port of IBL's block generator), because blocks are autocorrelated.
- **Tuning:** Spearman correlation with contrast, against contrast labels
  permuted within stimulus side.
- **Correction:** Benjamini–Hochberg (BH) across the units tested, with explicit
  seeds.

**Phy data:**
- Every condition needs its canonical column in the events CSV (`contrastLeft`,
  `contrastRight`, `choice`, `feedbackType`, `probabilityLeft`). A condition
  without its column is unavailable, with the reason shown.
- The pseudo-session null exists only for IBL's block generator, so block
  selectivity is refused for other tasks.

**Tests:**
- A split PSTH checked by hand, extending the prototype's hand-computed PSTH
  example with condition labels.
- A tuning curve checked by hand.
- **Calibration:** a simulated unit driven only by stimulus side, with choice
  correlated to side:
  - the conditional null keeps choice p-values uniform;
  - a plain shuffle doesn't. This is the test that justifies the null.
- The pseudo-session null reused unchanged.

## 6. Movement controls — built (2026-09-30)

Built (`docs/DECISIONS.md`, "Movement controls").
- **As planned:** the wheel-speed panel, the early/late reaction-time split, the
  movement-free test and the "movement-locked" label, with reaction times
  permuted within contrast.
- **Strata:** signed contrast, with 0% split by side (step 5's strata).
- **Statistic:** rate 0–200 ms after minus 200–0 ms before each trial's first
  movement, averaged over trials.
- **Movement-free:** applies to the test only, at stimulus onset. The plots keep
  every trial.
- **Phy:** locking also needs `contrastLeft`/`contrastRight` in the events CSV.
- **Flagged:** under the default trial filter, only 25 of 290 d23a44ef trials are
  movement-free for the 300 ms window. They are mostly low contrast.

**Adds:** separates rate changes around an event from movement, the confound step 3
found. 320 of 390 units "change around stimulus onset", with the baseline in the
quiescence period.
- **Wheel speed:** the wheel-speed PSTH drawn beside the neural one, from
  `behaviour.wheel`.
- **Reaction time:** stimulus-aligned PSTHs split by early and late movers.
- **A movement-free test:** the responsiveness test restricted to trials where
  the first movement comes after the response window ends.
- **A "movement-locked" label:** it compares alignment to each trial's own first
  movement with alignment to shifted movement times.

**Null:**
- **Movement-free test:** step 3's circular-shift null on the restricted trials,
  reporting how many trials remain.
- **"Movement-locked":** reaction times permuted across trials within each
  contrast level. A unit is labelled only if alignment to its own trial's movement
  beats the permuted alignments.
- **Correction:** BH across the units tested.

**Phy data:**
- Phy import reads spikes and events only, so there is no wheel. The wheel-speed
  panel and wheel-based movement onsets are declared missing.
- Only `firstMovement_times` supplied in the events CSV enables the movement-free
  test.

**Tests:**
- Hand-checked trial selection by reaction time.
- A hand-checked wheel-speed alignment.
- **Simulation:** a unit locked only to movement:
  - responsive at stimulus onset over all trials;
  - not responsive in movement-free trials;
  - labelled movement-locked.
- Null calibration on units with no locking.

## 6b. Single-trial view — built (2026-10-01)

Built (`docs/DECISIONS.md`, "Single-trial view").
- **As planned:** everything below, with matplotlib PNGs (no client-side renderer).
- **Neighbouring trials:** centred on the current trial, consecutive in the table,
  each marked when it fails the filters. N is 3, at most 9.
- **Wheel position:** relative to its value at zero.
- **Overlapping events:** go cue and response are drawn as wider halos under
  stimulus onset and feedback, which they overlap in IBL.
- **Also fixed:** exports had ignored the view's trial filters since step 4b.

**Adds:** every shown unit's spikes in one trial, with the task events on the same
plot. The engine goes in `analysis/trial_view.py`.
- **Population raster:** one row per shown unit, under the unit table's filters
  (QC, probe, region node, responsive only). Rows are grouped by probe with a
  separator and ordered by depth within each probe, with region colour bands at
  the current level and the selected unit's row highlighted.
- **Time axis:** from the trial's start (`intervals_0`) minus a pre-pad to its end
  (`intervals_1`) plus a post-pad; pads in `configs/`. Zero is the trial start, or
  a chosen event, and the caption says which.
- **Events:** every task event the trials table has, as labelled lines with one
  legend: stimulus on, go cue, first movement, response, feedback (reward and
  error drawn differently), stimulus off. An event missing on this trial is listed
  as "not recorded on this trial" and not drawn.
- **Header:** the trial number (0- or 1-based, stated), stimulus side and signed
  contrast, choice, outcome, block, reaction time, `bwm_include`, and whether the
  trial passes the current trial filters.
- **Behaviour, on the same time axis:** wheel position and speed, with the speed
  computation shared with `wheel_speed_psth`. Motion energy and pupil are
  optional traces. A missing signal is disabled with its reason.
- **Neighbouring trials:** N consecutive trials in one plot, N capped in
  `configs/`, with trial boundaries marked.
- **Navigation:** previous/next buttons, arrow keys and a trial box. They step
  through trials passing the trial filters, or all trials with a toggle, and show
  "trial k of n (filtered)". Clicking a raster row selects that unit. Clicking a
  trial in the selected unit's event-aligned raster opens that trial.
- **Saved and exported:** the trial, alignment, pads and number of trials go in
  the project file. Export writes SVG, PDF and a JSON sidecar with every spike
  and event time plotted.

**Null:** none. It is descriptive, with no labels, and the caption says so.

**Phy data:** events come from the events CSV, so only its columns are drawn. Phy
has no wheel or cameras, so those traces are disabled with the reason.

**Tests:**
- On a hand-built session (2 units, 2 trials): exactly the spikes inside the
  padded trial window, relative to the chosen zero.
- Row order: probe, then depth.
- A missing event is listed as not recorded and not drawn.
- Single-trial wheel speed equals that trial's row in the wheel-speed PSTH
  computation on the same bins.
- "Next" skips trials failing the trial filter; the toggle includes them.
- A project round trip keeps the trial view state.

## 7. Unit quality panel — built (2026-10-01)

Built (`docs/DECISIONS.md`, "Unit quality panel").
- **As planned:** everything below, as a Quality tab beside Activity in the
  Selected unit card.
- **IBL label criteria:** IBL's three label criteria, from the ONE cache when the
  session is there.
- **Waveforms:** read from local files only. IBL's are found through the manifest.
- **Presence ratio:** it matches IBL's stored values exactly on d23a44ef probe00.
- **Not built: drift.** Sessions don't load per-spike depths.

**Adds:** a per-unit panel showing:
- the ISI histogram with refractory lines;
- the autocorrelogram;
- the sliding refractory-period details (pair counts against the Poisson bound
  at each refractory period tested);
- firing rate and spike count across the session (presence and stability);
- the mean waveform when available;
- every QC reason.

**Null:** none new. It shows the QC results already computed. The sliding RP test
already carries its own confidence level, and it labels quality, not responses. Any
new quality label (drift, say) must use IBL's metric definition and its threshold
from config.

**Phy data:**
- **Waveforms:** they need the raw `.dat` file (not read) or `templates.npy` and
  the whitening inverse.
- **Amplitudes:** `amplitudes.npy` is in template units, not volts, so IBL's
  amplitude and noise-cutoff metrics stay missing.
- **IBL waveforms:** available through ONE (`clusters.waveforms`, cached for
  d23a44ef).

**Tests:**
- ISI histogram and autocorrelogram counts checked by hand.
- The panel's sliding-RP details reproduce `sliding_rp_pass`'s verdict for every
  probe00 cluster.
- A presence ratio checked by hand.

## 8. Cross-correlograms — built (2026-10-01)

Built (`docs/DECISIONS.md`, "Cross-correlograms and putative connections").
- **As planned:** interval jitter and BH across the pairs tested. The close-pair
  flag is declared missing without site positions.
- **Exact jitter null:** no seed, since nothing is random.
- **Excitatory label only:** a strong peak makes a jitter "trough" in the reverse
  direction, so inhibition is not labelled.
- **Test set:** the shown units, at most 30.
- **Flagged:** in LP all 6 labelled pairs are close pairs. Whether to exclude close
  pairs from the test is open.

**Adds:**
- **Views:** cross-correlograms for chosen pairs, within or across probes, raw
  and jitter-corrected.
- **Label:** putative monosynaptic connections.

**Null, for "connected":**
- **Test:** interval jitter (Amarasingham et al. 2012). Spikes are resampled
  within fixed windows of a few ms, which keeps slow co-modulation and breaks
  millisecond timing.
- **Correction:** BH across the pairs tested, reporting their number. Pairs grow
  as n².
- **Seed:** explicit.

**Phy data:**
- Works from spike times alone.
- Sorting hides near-simultaneous spikes on nearby channels (overlapping
  templates), which makes a false dip at zero lag. Close pairs need
  `channel_positions.npy` to be flagged; without it the flag is declared missing.

**Tests:**
- Cross-correlogram counts checked by hand.
- **Calibration:** jitter p-values are uniform on independent Poisson pairs.
- An injected 2 ms excitatory coupling is detected.
- A same-channel pair is flagged.

## 9. Population trajectories — built (2026-10-01)

Built (`docs/DECISIONS.md`, "Population trajectories").
- **As planned:** fit on alternate trials and shown on the others; axes `pc_k`;
  no labels.
- **Added:** soft normalisation and centring from the fit half, 30 ms smoothing,
  and at least 5 trials per condition in each half.
- **Variance:** the caption reports each component's share of the shown trials'
  variance.

**Adds:**
- **Plot:** condition-averaged population activity projected on principal
  components, in 2D or 3D.
- **Cross-validation:** components are fit on odd trials and data projected from
  even trials, the heatmap rule from step 3.
- **Naming:** axes are named `pc_k` (R5): no interpretation in code or plots.

**Null:** none while it is descriptive. A claim that trajectories differ between
conditions would need a distance statistic against condition labels permuted
within strata (as in step 5). Until that exists, no labels.

**Phy data:** works. There is no region grouping without channel locations (step 13).

**Tests:**
- Planted low-rank data is recovered.
- Cross-validated projection never uses held-out trials in the fit.
- The page shows no labels.

## 10. Decoding in Studio — built as S3 (2026-10-01)

**Adds:**
- **Scope:** decode a task variable from the selected units of one session
  through the existing six-row evaluation contract (`evaluation/contract.py`),
  with the split registry (R1, R2).
- **Display:** the page shows the full table and the null verdicts, never a bare
  score, per the Phase 8b rules.
- **Phase 3 result:** the gate did not pass (`docs/NEGATIVE_RESULTS.md`). So a
  result that doesn't beat `null_trialstruct` is said plainly.

**Null:** already in the contract: `null_shuffle`, `null_trialstruct`, and
pseudo-sessions for block.

**Phy data:**
- Targets need canonical trial columns, and wheel velocity needs a wheel (see
  step 6).
- The split registry is built around IBL session manifests. It needs a
  single-session path for a non-IBL folder, still with whole-trial blocks and a
  gap.

**Tests:**
- Studio's result equals the CLI's for the same units, split and seed.
- The guards refuse a random time-point split.
- The page renders all six rows and the verdict.

## 11. Unit browsing (session picking moved to step 4b)

**Adds:**
- **Opening sessions: replaced by step 4b** (homepage and data selection), which
  covers the session list, Phy folders and the folder-picker question.
- **Browsing units:** unit search by id or region, next and previous with the
  keyboard, and pinned units.

**Null:** none. There are no labels.

**Phy data:** nothing beyond step 4b.

**Tests:**
- Search by id and region.
- Keyboard next and previous follow the table's order.
- Pinned units survive a filter change.

## 12. Across-session region summaries — built as S5 (2026-10-01)

**Adds:**
- **Per region:** the fraction of responsive or tuned units per Beryl region
  across sessions.
- **Distribution:** always shown per session, never only pooled (§5).
- **Map:** a Swanson flatmap from iblatlas.

**Null, for region-level claims** (for example, "region X has more responsive
units than chance"):
- **Test:** sessions are the unit of inference. The null permutes region labels
  across units within each session, or uses a mixed model with session as a
  random effect.
- **Correction:** FDR across regions.
- **Minimum:** a minimum number of sessions per region, with smaller regions
  refused.
- **Counts:** units and sessions are reported per region.

**Phy data:** no regions, so excluded until channel locations are imported (step 13).

**Tests:**
- Aggregation checked by hand.
- The per-session distribution is present.
- A region below the session minimum is refused.
- **Calibration:** the region-level null stays calibrated when units are pooled
  unevenly across sessions.

## 13. Phy sync and channel locations

**Adds:**
- **Sync:** event times are aligned from the behaviour or NIDQ clock to the
  probe clock, from SpikeGLX sync pulses (offset plus linear drift). The fit
  residuals are reported, and it refuses above a tolerance. This replaces step
  1's rule that events must already be on the probe clock.
- **Channel locations:** histology-aligned channel positions are imported (IBL's
  alignment output `channel_locations.json`, or a CSV of channel to CCF position
  and acronym). This gives Phy units regions, the region tree and the 3D view.

**Null:** none. Validation is by fit residuals and known answers, not labels.

**Phy data:** this step fills the gaps listed in steps 1, 2, 5 and 12.

**Tests:**
- **Sync:**
  - planted drift and offset are recovered from simulated pulses;
  - on d23a44ef, sync pulses from ONE reproduce IBL's own alignment within a
    stated tolerance.
- **Channel locations:** an import round trip, with units mapping to the same
  acronyms as the BWM backend.

## 14. Installer (Phase 8b)

**Adds:**
- **Installer:** a one-step installer for macOS, Windows and Linux. The options
  to evaluate are conda constructor, pixi and PyInstaller.
- **Contents:** Python 3.11, the dependencies with their pins (the llvmlite/numba
  note in DECISIONS), and the vendored three.js.
- **In-app data:** downloads with progress for sessions, meshes and volumes.
- **Licence first:** the repo needs a licence before anything is distributed.
  This is why the GPL `slidingRP` package was not used.

**Null:** none.

**Phy data:** nothing specific, but folders must stay local and are never
uploaded.

**Tests:**
- A clean-machine install smoke test per OS in CI.
- The app starts offline with its vendored assets.
- A check that every bundled dependency's licence is compatible with the repo's.

## Not in these steps

- Spike sorting and curation. They stay upstream by design (CLAUDE.md §2).
- Hosted deployment. Studio is a local app only.
