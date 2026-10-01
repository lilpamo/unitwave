# Decisions

Record every non-obvious decision here: added dependencies, deviations from
CLAUDE.md, and choices about scope from §2. One entry per decision, newest
first.

---

### 2026-10-01 — S1 robustness pass: eight sessions, three fixes (`analysis/correlograms.py`, `data/atlas_meshes.py`, `viz/studio_plots.py`, page scripts)

**What was run (the user's step S1):**
- **Sessions:** eight BWM sessions that differ from d23a44ef, each opened with the
  homepage's open request.
- **Every view:** a scratch script made the page's 47 requests per session and
  timed each one:
  - the unit table, and rasters and PSTHs at every event;
  - split PSTHs and tuning for every condition;
  - responsiveness, with and without the movement-free option, and selectivity
    for every comparison;
  - the wheel, and movement locking;
  - the trial view, unit quality and one pair;
  - connections on the largest region with 2–30 units;
  - the 3D view and probe strip, and trajectories in 2-D and 3-D;
  - project save and reopen, and export.
- **In the browser:** three sessions were opened from the homepage (6a601cc5,
  3a3ea015, and connections run from the page's button).

| Session | Why it was chosen | Failed | Fix |
|---|---|---|---|
| 3a3ea015 | Smallest: 2 good units. One shank (`probe00a`) of a Neuropixels 2.0 probe | Heatmap unit axis read 0.00, 0.25 … 2.00 | Whole-unit ticks |
| 0c828385 | One probe. No video (no pose or motion energy) | Nothing | — |
| b182b754 | One probe, deep structures only (MB, HB). Missing events: stimulus 4, first movement 16, feedback 5 | Connections on MRN (28 units): 164 s | Fast connection test: 8.1 s |
| 6a601cc5 | Two probes. Fast MRN units (up to 90 Hz). 188 trials without a stimulus time | Connections on MRN (27 units) never finished (over 30 min). One mesh download stalled and hid every region mesh. One cut short and left the page an empty response | Fast connection test: 47 s, 37 s in the page. Meshes drawn one by one, retried |
| 7f5df7eb | One probe, cortex only (343 good units) | Nothing | — |
| c16d3557 | One probe, deep only (HB, CB). 248 of 459 good units in no Beryl region | Nothing | — |
| 872ce8ff | Two probes, mostly hippocampus. No video | Nothing | — |
| dd4da095 | Largest: 536 good units, 2,725 in all. Two probes. No video | Nothing | — |

**Fix 1, the connection test (test-first, from 6a601cc5):**
- **The cause:** step 8 convolved one jittered spike at a time, so a test cost
  (spikes) × (observed count). One directed test in 6a601cc5's MRN took 58–102 s
  (probe00_98 → probe00_117: 515,252 and 536,596 spikes, 144,142 lags).
- **The fix, part 1:** draws are convolved in pairs, level by level, with batched
  FFTs. Mass at or above observed + 1 is folded into one bin after each level,
  which is exact because counts are never negative. That pair now takes 1.1 s.
- **The fix, part 2:** a region's tests run on threads. Each test is exact and
  independent, and a test checks the table equals testing each direction in
  turn. On 72 real pairs, threaded and serial results are bitwise identical.
- **Precision, against the old computation on real pairs:**
  - errors in p are at most 2.6e-12;
  - p below `P_RESOLUTION` (1e-9) is now reported as 1e-9, which is
    conservative and far below any BH threshold (0.05 / 870 ≈ 6e-5);
  - both p come from the last two bins of the distribution, not sums over many
    bins;
  - negatives from rounding are not clipped: clipping biased sums upward, by up
    to 5e-9.
- **Determinism:** the expected count uses NumPy sums, not a BLAS product. BLAS
  gave a last-digit difference between a thread and the main thread.
- **A bug in the first version, caught by the suite:** a pair where no lag could
  ever fall in the window crashed. It has its own test now.

**Fix 2, region meshes (test-first, from 6a601cc5):**
- **What happened:** the Allen server stalled on structure 679 (CS) until the
  60 s timeout, and the page drew no region meshes at all. It also cut short
  structure 771 (P) after 63,508 of 916,133 bytes. That error (`IncompleteRead`)
  is not an `OSError`, so the server's refusal missed it and the page got an
  empty response.
- **The fix:**
  - a failed download is tried once more;
  - any download failure becomes a plain refusal that names the structure;
  - the page draws each mesh on its own, names the regions not drawn and why,
    and forgets the failure so the next redraw asks again;
  - the homepage's brain outline retries the same way.
- **Checked in the browser:** with a failed download simulated, the other regions
  were drawn and the note named the missing one. After the network was restored,
  only that mesh was fetched again.

**Fix 3, heatmap ticks (test-first, from 3a3ea015):** the unit axis uses whole
numbers.

**Slower than 2 s on the largest session (dd4da095); not changed, for review:**

| View | Time | Where the time goes |
|---|---|---|
| Movement locking | 22.2 s | The permutation loop (13 s) and spike-window counts (7 s) |
| Responsiveness | 12.5 s | FFTs for the all-shifts null (7.5 s) |
| Responsiveness, movement-free trials | 11.6 s | The same |
| Export | 8.5 s | Writing SVG and PDF in matplotlib (5.3 s, 1.9 s of it embedding fonts) |

Everything else on that session took under 1.3 s. Connections on its largest
region (DP) took 1.7 s. On the other sessions, locking took 5.8–16.4 s and
responsiveness 7.6–12.7 s.

**Not covered, because the release lacks them:**
- **No missing wheel:** the behaviour release has wheel data for all 459 sessions
  (`missing_wheel_sessions: []`). Phy folders cover the no-wheel case.
- **No session with two shanks of one Neuropixels 2.0 probe:** the release has 4
  shank insertions (`probe00a`, `probe00b`), one per session. 3a3ea015 covers a
  single shank.

**No new dependency:** `scipy.fft` is part of SciPy.

### 2026-10-01 — Population trajectories (`analysis/trajectories.py`, `configs/trajectories.yaml`)

**What (the user's step 9):** a Trajectories tab beside Heatmap in the Population
card.
- **Views:** the shown units' condition-averaged activity on its principal
  components, in pc_1–pc_2 or in 3-D, with each component plotted against time
  below.
- **What it uses:** the page's event, window, bins, trial filters and split (one
  line per condition, or "all trials" without a split).
- **Descriptive:** no statistic, no null, no label. Axes are `pc_k` (R5). The
  caption says "descriptive (no test)", and a test checks that nothing sent to the
  page uses label words.
- **Saved and exported:** the tab and its dimensions are saved in the project.
  Export adds `trajectories.svg/.pdf/.json`, with the trajectories, the components
  and the units they belong to.

**Cross-validation, the heatmap's rule:**
- **The split:** each condition's trials are split alternately
  (`psth.alternate_halves`). The 1st, 3rd, ... trials fit and the 2nd, 4th, ... are
  shown.
- **Everything learned comes from the fit half:** each unit's soft-normalisation
  scale (rate range + 5 Hz, Churchland et al. 2012), its mean and the components.
  The shown half is normalised, centred and projected with those numbers.
- **Checks:** a test shows that flooding the shown trials with spikes leaves the
  components unchanged. The caption reports each component's share of the *shown*
  trials' variance.

**Choices, flagged for review:**
1. **Smoothing:** a Gaussian of 30 ms sigma, applied to every condition average
   before the fit, the same on both halves.
2. **Equal weight per condition:** conditions are averaged first, regardless of
   their trial counts. Small conditions are therefore noisier, and a condition
   needs at least 5 trials in each half (`configs/trajectories.yaml`). Below that it
   is left out and named in the caption.
3. **No baseline subtraction:** the toolbar's baseline option applies to the
   heatmap, not here. Each unit is centred on its fit-half mean instead.
4. **Fixed signs:** each component is flipped so its largest loading is positive,
   so the same data always draw the same picture.
5. **3-D uses matplotlib's own 3-D axes,** so there is no new dependency.

**Checks (test-only spike trains, never shown as data):**
- **A planted rank-2 structure is recovered on the shown trials:** pc_3 stays at
  the noise floor, and the shown trajectories map onto the planted time courses
  with R² of 0.99 and 0.93.
- **What the test asserts:** the rank and the R², not a total variance share. That
  share depends on Poisson noise: 0.70 at 60 trials per condition, 0.91 at 300.
- **Exclusions:** too-few-trial conditions are left out and counted, and signs are
  fixed.

**Real data (d23a44ef, 390 QC-passing units, stimulus onset, BWM inclusion):**
- **By choice:** 108 + 38 fit trials and 107 + 37 shown (right, left). Shown-trial
  variance on pc_1–3 is 59%, 17% and 3%, against 65%, 17% and 4% on the fit half.
- **By signed contrast (9 levels):** shown-trial variance on pc_1–3 is 48%, 12%
  and 4%. The conditions separate after stimulus onset.
- **Speed:** 0.5–1 s per view.

**No new dependency.**

### 2026-10-01 — Rename: UnitWave Studio

**Decision (the user's):** the app is renamed **UnitWave Studio**, in two parts.

**What changes:**
- **Part 1, the name users see:**
  - every user-facing string: page titles, the app header, the server's startup
    message, export manifests (an `app` field) and figure metadata (the creator
    written into SVG, PDF and PNG files);
  - the docstrings that name the app, and the plan doc's title;
  - CLAUDE.md's title and §1, and a rewritten README;
  - the `description` in pyproject.toml;
  - file endings: projects `.ndstudio.json` → `.unitwave.json`, session sets
    `.ndset.json` → `.unitwave-set.json`.
- **Part 2, the Python package:**
  - `neurodecoder` → `unitwave` (moved with `git mv`, so history follows), with
    every import, `python -m` command and pyproject's `name`;
  - environment variables `UNITWAVE_DATA_ROOT` and `UNITWAVE_NETWORK_TESTS`.

**What deliberately doesn't change:**
- **The data folder** `~/data/neurodecoder` (over 20 GB; `configs/data.yaml`).
- **Cache keys,** so cached sessions load without a rebuild.
- **Run folder names** (`runs/<time>_studio`) and **branch names.**
- **The local repository folder and the GitHub repository:** the user renames those.
- **The historical record:** older entries in this file (including the decision
  titled "Direction change: Neurodecoder Studio"), NEGATIVE_RESULTS.md,
  PRIOR_ART.md and ROADMAP.md keep their wording. CLAUDE.md's pointer to that
  decision keeps its real title.

**Backwards compatibility:**
- **Old files open:** projects ending `.ndstudio.json` and sets ending
  `.ndset.json` still open, and the recent-projects and sets lists show both
  endings.
- **Old files are never deleted or overwritten:** saving a project opened from an
  old file writes the new ending beside it. If that name is taken, it takes the
  next free name (`-2`, `-3`, ...). The page says where a save will go.
- **Old environment variables (part 2):** `NEURODECODER_DATA_ROOT` and
  `NEURODECODER_NETWORK_TESTS` keep working, with a one-line deprecation warning.
  When both old and new are set, the new one wins.

**Part 2, done:**
- **The package moved** with `git mv neurodecoder unitwave`: 84 files, as renames.
  Every reference changed with one pattern, the package name as a whole word not
  after "/", which keeps `~/data/neurodecoder` and the uppercase variable names.
  That is 421 replacements, and every diff line in a moved file is that one name.
  Commands are `python -m unitwave.studio.server`, and the export manifest's
  `command` is `unitwave.studio export`.
- **Environment variables** are read through `unitwave/env.py`, which tests cover:
  new name, old name with one warning line, both set, unset.
  - 19 tests now read `env("DATA_ROOT", ...)` instead of `os.environ.get(...)`,
    and their unused `import os` went.
  - `test_load.py` sets `UNITWAVE_DATA_ROOT`.
  - No other test changed.
- **Before and after:** 559 passed and 2 skipped before (at ad85466). After, 563
  passed and 2 skipped: the same 561 test IDs plus the 4 new environment tests.
- **Cache keys are identical** (checked on three sessions). d23a44ef loads from the
  cache in 0.4 s without calling a backend.
- **Kept on purpose:**
  - the comment commands in `configs/runs/phase3_*.yaml`: Phase 3's run
    manifests record those files' byte hashes, and NEGATIVE_RESULTS and older
    entries here cite them;
  - `docs/PROMPTS.md` and ROADMAP.md's `neurodecoder analyze`: historical.
- **Consequence:** the Studio configs' comments now name `unitwave/...`, so their
  byte hashes changed. Projects saved before warn once that `configs/analysis.yaml`
  and others changed. The change is comment-only, and re-saving clears it.

### 2026-10-01 — Cross-correlograms and putative connections (`analysis/correlograms.py`, `configs/correlograms.yaml`)

**What (the user's step 8):**
- **A Pairs tab** in the Selected unit card: the cross-correlogram with a chosen
  partner, its expectation under interval jitter, the jitter-corrected correlogram
  (observed − expected) and the pair's test results.
- **A Connections section:** tests every pair among the shown units, both
  directions, with BH across all of them. It refuses more than `max_units` (30)
  units.
- **A "Conn." column** shows each unit's outgoing and incoming putative
  connections.
- **Saved and exported:** the partner is saved in the project. Export adds
  `ccg.svg/.pdf/.json` and `connections.csv`.

**The null, exact rather than sampled:** interval jitter (Amarasingham et al. 2012).
- **What is jittered:** the second unit's spikes, each uniformly within its fixed
  5 ms window. That keeps everything slower than 5 ms and breaks millisecond timing.
- **The statistic:** the count of pairs at lags 1–4 ms.
- **Why it can be exact:** under jitter the count is a sum of independent
  per-window counts. Their exact distribution comes from convolution, and the
  expected correlogram has a closed form.
- **So there is no seed:** the plan asked for an explicit seed, but nothing is
  random, as in step 3's all-shifts null.
- **Checked against Monte Carlo:** 4,000 jitters match the exact mean and tail
  probability.

**Choices, flagged for review:**
1. **Excitatory only (one-sided).** The jitter expectation is the true correlogram
   smoothed over 5 ms windows. A real A → B peak at +2 ms raises the expectation
   at B → A's lags too, so B → A shows a significant "trough": p ≈ 3e-14 two-sided,
   on an injected A → B coupling with no B → A coupling. Labelling inhibition
   would call that every time, so troughs are shown, not labelled.
2. **Close pairs are labelled but flagged:** units within 50 µm on one probe. Sorting
   deletes their near-simultaneous spikes, and smoothing that zero-lag dip lowers
   the expectation at 1–4 ms, which can make an excess.
   - **On d23a44ef:** in LP all 6 labelled pairs are close; in CA1, 3 of 9.
   - **The alternative is your call:** exclude close pairs from the test
     altogether.
3. **The test set is exactly the shown units,** and a result belongs to that set.
   Changing the filters means a new test.
4. **One correlogram implementation:** the quality panel's autocorrelogram now
   uses `cross_correlogram` (half-open bins), minus each spike's pair with itself.

**A bug found and fixed:** jitter windows assigned by `floor(t / D)` misplace
boundaries late in a session (`129.04 / 0.005` < 25808). Window probabilities
stopped summing to 1, and p-values went negative. It now uses exact comparisons
with the window boundaries, and a test checks that shifting both trains by 3000 s
changes nothing.

**Checks (test-only spike trains, never shown as data):**
- **By hand:** correlogram counts, and the jitter expectation.
- **Calibration:** on pairs sharing a slow 1 Hz rate modulation but no fine
  coupling, the one-sided p is uniform (KS p > 0.01; 0.94 when measured), with
  4.3% below 0.05.
- **Injected coupling:** a 2 ms coupling is found (p < 1e-6) in its direction only.

**Real data (d23a44ef, QC-passing units):**

| Region | Units | Tests | Labelled | Close-flagged among them | Time |
|---|---|---|---|---|---|
| CA1 | 13 | 156 | 9 | 3 | 16 s |
| PO | 30 | 870 | 0 | 0 | 33 s |
| LP | 28 | 756 | 6 | 6 | 34 s |

- probe00_446 → probe00_468 (CA1, not close): a sharp peak at +1 to +1.5 ms,
  4,113 pairs per 0.5 ms bin against about 1,450 nearby, q = 2e-244.
- probe00_468 (82 Hz) receives 6 of CA1's 9.

**Limitations:**
- Common input at millisecond timescales looks the same as a connection, hence
  "putative".
- An effect spread over more than 5 ms is part of the null.

**No new dependency.**

### 2026-10-01 — Unit quality panel (`analysis/unit_quality.py`, `data/cluster_files.py`, `configs/unit_quality.yaml`)

**What (the user's step 7):** a Quality tab beside Activity in the Selected unit
card. It shows the QC verdicts already computed and labels nothing new.
- **ISI histogram and autocorrelogram.**
- **The sliding refractory-period test at every period tested:** violations against
  the most allowed, and where the unit first passes.
- **Rate across the session,** with IBL's presence ratio.
- **The mean waveform,** when there is one.
- **Every QC reason,** and for IBL sessions in the ONE cache, the three criteria
  behind IBL's `label` with their values.

Export adds `quality.svg/.pdf/.json`, with every count, test row and waveform
sample.

**Choices, flagged for review:**
1. **The refractory details reuse the QC code.** `qc/refractory.sliding_rp_details`
   shares its setup with `sliding_rp_pass` (moved into `_prepare`). The pass/fail
   loop is unchanged.
   - The pinned agreement with IBL's flags is still 553 of 674.
   - A new test checks the details' verdict equals `sliding_rp_pass` on all 674
     probe00 clusters.
2. **For IBL sessions the panel runs the refractory test with IBL's settings**
   (10% contamination, 90% confidence), for reference. Studio's IBL QC uses IBL's
   label, and the panel says so. For Phy folders it is part of the QC, with
   `configs/qc_phy.yaml`'s settings.
3. **IBL's label criteria use IBL's fixed thresholds** (max confidence ≥ 90%, noise
   cutoff < 5, median amplitude > 50 µV), from `brainbox` `compute_labels`. They
   explain a stored label rather than make a new one. On d23a44ef probe00 they
   reproduce IBL's `label` for all 674 clusters.
4. **Presence ratio is IBL's definition, exactly:** 10 s bins from the first spike,
   `np.arange(start, end + 5, 10)` of them. That includes a last bin no spike can
   reach when the recording ends past the middle of a 10 s window.
   - It equals IBL's stored value for all 674 probe00 clusters.
   - In Studio, start and end are the first and last spikes of the probe's
     *loaded* units. For BWM sessions (good units only) that can differ slightly
     from IBL's span. No threshold is applied.
5. **Waveforms come from local files only:**
   - **IBL:** `clusters.waveforms.npy` in the ONE cache, in µV. The folder comes
     from the manifest's lab, subject, date and session number, because ONE's
     local mode can't resolve an eid offline.
   - **Phy:** the spike-weighted mean template, unwhitened with
     `whitening_mat_inv.npy`, in template units.
   - **Otherwise:** a plain reason naming the missing files.
   - On d23a44ef only probe00's waveforms are cached; probe01's panel says so.
6. **No drift:** IBL's drift needs per-spike depths, which sessions don't load.
   Per the plan, any drift label waits for IBL's definition and a config threshold.

**A bug the server tests caught:** an autocorrelogram lag within rounding of the
window's outer edge overflowed past the last bin. It's fixed, with a test.

**Real data (d23a44ef):**
- probe00_3: passes QC; the refractory test passes from 1.25 ms; IBL's label is 3
  of 3 (max confidence 93%, noise cutoff −0.54, median amplitude 122 µV).
- probe00_27: fails Studio's QC (task firing rate 0.0013 Hz), with a presence ratio
  of 0.084.
- Speed: the panel takes about 0.5 s per unit.

**No new dependency.**

### 2026-10-01 — Single-trial view (`analysis/trial_view.py`, `configs/trial_view.yaml`)

**What (the user's step 6b):** every shown unit's spikes in one trial, or a few
consecutive trials, with the task events and behaviour on the same time axis. It is
descriptive: no statistic, no null, no label, and the caption says so.
- **Rows:** the unit table's filters (QC, probe, region node, responsive only),
  grouped by probe with a separator, the most superficial unit on top. Region bands
  at the current level, and the selected unit marked.
- **Window:** the trial's start minus a pre-pad to its end plus a post-pad,
  half-open. Zero is the trial start or a chosen event.
- **Events:** every event column the trials table has, with one legend. A missing
  event is listed as "not recorded on trial k" in the header and the figure, and
  not drawn. Aligning to it is refused.
- **Header:** trial number, stimulus, choice, outcome, block, reaction time,
  `bwm_include`, and whether it passes the current trial filters.
- **Behaviour:** wheel position and speed, plus optional motion energy and pupil
  traces. A missing signal is disabled with its reason.
- **Navigation:** previous/next, arrow keys and a trial box. They step through the
  filtered trials, or all trials with a toggle, showing "trial k of n filtered
  trials". Clicking a raster row selects the unit. Clicking a row of the unit's
  event-aligned raster opens that trial (the server sends each row's trial, as it
  sends the heatmap's units).
- **Saved and exported:** the trial, alignment, pads, number of trials, the
  all-trials toggle and the traces are saved in the project. Export writes
  `trial.svg/.pdf/.json`, with every spike and event time plotted.

**Choices, flagged for review:**
1. **Matplotlib PNGs, no client-side renderer.** Hovering a row names the unit,
   and the exact spike times are in the export's JSON. Hover on single spikes
   didn't seem worth a new dependency, so I didn't ask.
2. **Trial numbers are the 0-based row of the trials table**, stated in the
   header. On d23a44ef that equals IBL's `trial_id`.
3. **Neighbouring trials are centred on the current trial,** consecutive in the
   table whether or not they pass the filters. Each is marked when it fails them.
   N is 3 when the option is on, at most 9 (`configs/trial_view.yaml`).
4. **Wheel position is plotted relative to its value at zero.** The raw position
   is cumulative over the session.
5. **Wheel speed** uses `movement.binned_wheel_speed`, now shared with the
   wheel-speed PSTH, on 20 ms bins tiled from the zero. A test checks a trial's
   trace equals its row of that computation.
6. **Overlapping events:** in IBL the go cue comes within ~1 ms of stimulus onset,
   and feedback within ~1 ms of the response. Go cue and response are drawn wider
   underneath as a halo, so both lines show at their true times.
7. **Event colours** are the categorical slots in task order. Feedback keeps one
   hue: reward solid, error dotted.

**A bug found and fixed (from step 4b part (a), 3881fe1):** exports ignored the
view's trial filters.
- `view_to_query` sent them under `trials`, but the server reads `tf`, so exported
  figures were computed on every trial while the page used the filter.
- The sidecars recorded what was used (no filter), so no file is internally wrong,
  but exports made with a filter don't match the page and should be redone.
- Now fixed, with a test, and the real-data export test checks 290 kept trials.

**Also:**
- `conditions._NAMES` is public as `LEVEL_NAMES`, so the header names levels as
  the splits do.
- Projects hash `configs/trial_view.yaml`.
- A bad trial number gets a plain refusal.

**Real data:**
- **Speed:** on d23a44ef (390 units), the engine takes ~0.1 s per trial and the
  figure ~0.6 s.
- **Missing events:** d23a44ef has none. Of the 26 cached sessions on the current
  loader, 24 have trials with a missing event, mostly first movement. On 9fe512b8,
  trial 236 has no first movement or feedback time, yet the wheel turns ~0.6 s
  after stimulus onset. The view shows the wheel and draws no movement line.

**No new dependency.**

### 2026-09-30 — Movement controls (`analysis/movement.py`, `configs/movement.yaml`)

**What (the user's step 6):**
- **Wheel speed:** |d position / dt| (rad/s) from `behaviour.wheel`, mean ± SEM per
  bin on the unit PSTH's event, window, bins and trials, drawn under it on the same
  time axis.
- **Reaction time split:** "Reaction time" in Split by, early vs late at the
  median (first movement - stimulus onset). Trials without a first movement are
  excluded and counted.
- **Movement-free test:** a checkbox in Responsiveness runs step 3's shift test on
  the trials whose first movement comes after the response window's end (300 ms).
  It is kept apart from the all-trials result, and the summary names the trials.
- **"Movement-locked" label:** a Movement section runs it; the table's "Mov."
  column shows ↑, ↓ or · per unit.

**The locking statistic and null:**
- **Statistic:** the mean over trials of (rate 0–200 ms after minus rate 200–0 ms
  before) each trial's own first movement.
- **Null:** the same with reaction times permuted within signed-contrast strata,
  0% split by side (step 5's strata). Each trial keeps its stimulus, and gets a
  reaction time from its own contrast.
- **Why it separates the two:** a stimulus-locked unit blurs the same way under
  true and permuted reaction times, so it isn't labelled; a movement-locked unit
  is sharper at its true movement times.
- **Settings:** two-sided, p = (1 + #|null| ≥ |observed|) / (1 + n), 10,000
  permutations, seed 0, windows in `configs/movement.yaml`. BH across the units
  tested, at step 3's α.

**Choices beyond the plan, flagged for review:**
1. **The movement-free box applies to the test only.** The plots keep every
   trial, and the box says so. It is defined at stimulus onset only.
2. **Locking needs the contrast columns** as well as `firstMovement_times`, for
   the strata.
3. **Projects now hash `configs/selectivity.yaml` and `configs/movement.yaml`.**
   Step 5 left the selectivity config out. Files saved before this say "was not
   recorded when the project was saved" instead of failing to open.
4. **Exports** add `wheel.svg/.pdf/.json` when the session has a wheel, and
   `movement_locking.csv` when the test was run.

**Checks (test-only Poisson spike trains, never shown as data):**
- **By hand:** reaction times, movement-free trials (the boundary and missing
  movements), and a wheel-speed PSTH.
- **Simulation:** units that fire only after movement are responsive at stimulus
  onset over all trials (> 90%) and not on movement-free trials (< 20%). A
  movement-locked unit gets p < 0.01; a stimulus-locked one p > 0.05.
- **Calibration:** on 150 stimulus-locked units, locking p-values are uniform (KS
  p = 0.2), with 0 labelled.
- **A test design bug found:** trials evenly spaced with background spikes only
  around trials made the shift null in the movement-free test wrong. The tests use
  irregular spacing and background over the whole session, like real data.

**Real data (d23a44ef, 390 QC-passing units, stimulus onset):**

| Trials | Movement-free trials | Responsive on them | Movement-locked |
|---|---|---|---|
| all | 87 of 410 | 153 | 235 (on 410 trials, 28 s) |
| BWM inclusion (default) | 25 of 290 | 117 | 110 (on 290 trials, 14 s) |

Over all trials, step 3 found 320 of 390 units responsive.

**Flagged: movement-free trials are few, and not a random subset.**
- Under the default filter only 25 trials remain, so the test has little power.
  153 or 117 responsive units is not "how many respond to the stimulus without
  movement": fewer trials find fewer units.
- 62 of the 87 all-trial movement-free trials have reaction times over 2 s, which
  BWM inclusion drops as disengaged.
- They over-represent low contrast: only 5 of the 92 100%-contrast trials qualify.
  A unit that loses its response on them may be contrast-driven.
- The 300 ms window is the user's call. A shorter response window keeps more
  trials.

**Limitations:**
- Within a contrast, trials are treated as exchangeable. If reaction time tracks
  the unit's state (engagement, say) for other reasons, the locking label can
  over-call.
- The labels are not exclusive: a unit can be both stimulus-responsive and
  movement-locked.
- Phy folders have no wheel, so the panel says why. Their events CSV enables the
  movement-free test and locking when it has `firstMovement_times` and contrasts,
  as the d23a44ef export does (67 of 161 units locked, all trials).

**No new dependency.**

### 2026-09-30 — Homepage and data selection, part (b) (`analysis/catalog.py`, `studio/sets.py`, `studio/entry.py`)

**3D overview:**
- **What it shows:** every probe of the matching sessions as a line from the
  manifest's insertion tip to its top, converted to CCF by `analysis/atlas.ccf_um`,
  the same conversion as unit positions (`probe_lines`, tested against it).
- **Lab colours:** the dataviz palette's eight categorical slots in fixed order,
  largest lab first. The BWM has 12 labs, so the four smallest (churchlandlab,
  steinmetzlab, mrsicflogellab, hoferlab) share the muted ink as "other labs"; a
  ninth hue is never generated. The legend names every colour.
- **Interaction:** hovering names the lab, subject, date and probe; clicking opens
  the session. It updates with the filters.
- **A bug found and fixed:** a drag to rotate ended in a click, which would open
  whatever probe was under the pointer. Only a click that barely moved opens a
  session now.
- **Meshes:** `/mesh/<id>` no longer needs an open session.

**Session sets (`data_root/sets/<name>.ndset.json`):**
- **What a set holds:** plain JSON with the sorted eids, the manifest version, the
  session and trial filters used, and a sha256 over all of it.
- **Reopening:** a set restores the filters and the selection. It warns if the
  manifest version changed, and refuses a file whose hash doesn't match.
- **Opening by name only:** names are plain (letters, digits, space, `_`, `.`,
  `-`), and sets are opened by name, never by path.
- **One session at a time:** the session view still holds one session.

**Open a Phy folder:**
- **The root:** `configs/catalog.yaml` has `phy_root: phy`, relative to
  `data_root`.
- **Path checks:** every path is resolved with symlinks followed and must stay
  inside the root, so `../` and symlinks pointing out are both refused.
- **Plain-language refusals:** a folder without `spike_times.npy` or `params.py`,
  or without `events.csv` beside `params.py`, is refused with a message saying
  which.
- **Completion** offers only folders inside the root, and marks which are Phy
  folders and which have events.
- **Demo folder:** the Phy-format d23a44ef export was copied to
  `data_root/phy/d23a44ef/probe00`, with its `events.csv` and README. The original
  in `derived/` stays for projects that point at it.

**Recent projects:** the `*.ndstudio.json` files in `data_root/projects`, newest
first, opened by name only.

**No new dependency:** three.js was already vendored.

**Observed while previewing:** twice, a different session was opened on the
running server by something outside my steps, most likely the user trying it in
another tab. The page's own open code hadn't run. Opening is a POST, so this is
expected with more than one client; the page always shows which session is open.

### 2026-09-30 — Homepage and data selection, part (a) (`analysis/catalog.py`, trial filters in `analysis/conditions.py`, `studio/`)

**Homepage:**
- **Starting up:** the server starts with no session and serves a homepage at `/`.
  Opening a session is a guarded POST (the existing Origin and Content-Type
  checks) with a visible loading state, then `/session`, with a Home link back.
- **Old flags:** `--eid`, `--phy` and `--project` still open a session directly and
  print its `/session` URL.
- **Switching session** replaces the Studio object, so selections, caches and test
  results start empty.

**Manifest version 2 (`data/manifest.py`)**, bumped for two additions:
- **`region_units`:** the release's good units per Allen acronym per probe, so the
  region filter can count any node's descendants.
- **Motion energy and pupil modalities,** read from each session shard's
  `meta.json` by the same rule the BWM backend uses. A real-data test checks that
  d23a44ef's listed modalities equal what the backend loads.
- **Why not "wheel and pose only":** the user offered labelling the behaviour
  filter that way instead. But Studio already loads motion energy and pupil, the
  version was being bumped anyway, and reading all 459 shards' metadata takes
  0.9 s.
- **Consequence:** Phase 3's saved split files record manifest version 1, and
  `splits/guards.py` rejects them against version 2. Phase 3 is parked.

**Where the catalog lives:** `data_root/derived/manifest-v2/` (sessions,
insertions, region_units and provenance). It is built on first start (about 1 s)
and read afterwards. A new manifest version gets a new folder, so new code never
reads an old copy.

**Session filters and counts** are in `analysis/catalog.py`; the page only draws
them.
- **The filters:**
  - lab, subject and date range;
  - a region, with descendants and at least `min_region_units` good units (10 by
    default, in `configs/catalog.yaml`, adjustable on the page);
  - minimum good units, minimum included trials, and number of probes;
  - behaviour modalities, all required.
- **Unit counts are the release's good units, labelled as such.** d23a44ef lists
  398; Studio's QC passes 390.
- **Real example:** HPF with at least 10 units, left-camera pupil, and at least
  100 good units gives 73 sessions, 125 probes, 15,418 units, 2,965 of them in
  HPF.

**Trial filters, one definition:** `TrialFilter` in `analysis/conditions.py`, on
step 5's condition definitions.
- **The filters:** `bwm_include`, excluding no-go, contrasts, blocks and outcomes.
- **Counting:** every excluded trial is counted under each reason it fails, and
  once in the total. Captions give the count kept and why the rest were excluded.
- **Records:** the filter's canonical key is part of the responsiveness and
  selectivity cache keys. A test checks a result is never shown under another
  filter.
- **Projects and exports:** the filter is part of the project view (older files
  open on all trials, as they were computed) and of every export sidecar.
- **Block selectivity** under a filter gets the full trial table plus a mask, so
  pseudo-sessions keep IBL's block structure.
- **Phy sessions** offer only the filters their events CSV supports, with the
  reason shown for the rest (for example, "trials have no bwm_include column").

**Default trial filter: `bwm_include` and exclude no-go** (`configs/catalog.yaml`),
applied when a session is opened from the homepage or with `--eid`/`--phy`,
minus what the session can't support.
- **What `bwm_include` is on d23a44ef:** exactly a reaction-time window. All 120
  excluded trials have first movement less than 80 ms (58) or more than 2 s (62)
  after stimulus onset.
- **Why:** those are anticipatory and disengaged trials, which muddy
  stimulus-onset responses, and the release's own analyses use the same rule.
- **Before and after on d23a44ef:**

  | | Trials | Responsive at stimulus onset | Responsive at error feedback |
  |---|---|---|---|
  | All trials | 410 | 320 of 390 | 183 of 390 |
  | `bwm_include` | 290 | 301 of 390 | 153 of 390 (67 error trials, was 106) |

- **Not filtered:** unit QC still uses every trial for its task-period rate. QC
  decides whether a unit is valid, not what an analysis includes.

### 2026-09-30 — Condition-split PSTHs, tuning curves and selectivity (`analysis/conditions.py`, `analysis/tuning.py`, `configs/selectivity.yaml`)

**What (the user's step 5):**
- **Conditions from the trials table:** stimulus side, signed contrast, choice,
  outcome and block. Only those whose columns exist are offered, so Phy sessions
  get what their events CSV has.
- **Split PSTH:** one mean ± SEM trace per condition, with a legend giving n per
  condition, and the raster grouped by condition.
- **Tuning curve:** mean ± SEM response rate per level in `configs/analysis.yaml`'s
  response window, with n per level.
- **Selectivity:** AUROC between two conditions per unit, against a null, with
  Benjamini–Hochberg across the units tested. `n_tests`, the null, the number of
  draws, the seed and the window are always shown.
- **Settings:** `n_permutations`, `n_pseudo_sessions` (10,000 each), `seed` and
  `min_trials` are in `configs/selectivity.yaml`.

**Conventions, checked on d23a44ef rather than assumed:**
- **Contrast:** each trial sets exactly one of `contrastLeft` or `contrastRight`,
  and the side without a stimulus is NaN. Zero contrast is 0 on the stimulus side.
  Trials that break this are excluded and counted.
- **Choice:** on correct trials a right stimulus has `choice = -1` and a left one
  `+1`, so the levels are named "right (-1)" and "left (+1)".
- **No-go trials** (choice 0) are excluded and counted.

**Nulls:**
- **Choice:** labels permuted within signed-contrast strata, as the user asked.
- **Block:** IBL-generator pseudo-sessions (`evaluation/nulls.py`), as the user
  asked.
- **Speed:** AUROC comes from rank sums. Ranks don't depend on labels, so each
  unit's ranks are computed once and every draw is one matrix product, which
  takes under a second for 390 units.

**Choices beyond the user's spec, flagged for review:**
1. **Side:** permuted within choice strata. This is the mirror of the choice/contrast
   confound: without it, a choice-driven unit would be called stimulus-selective.
2. **Outcome:** permuted within signed-contrast strata, because errors concentrate
   at low contrast.
3. **Choice and outcome strata split 0% contrast by stimulus side.** At 0% the
   rewarded side still differs and choice tracks it, so one merged 0% stratum
   would leave choice confounded with side. Display and tuning keep IBL's single
   0%.
4. **Block:** uses the pre-event (baseline) window. The block prior predicts the
   stimulus side 80% of the time, so a post-stimulus window would count stimulus
   responses as block selectivity. IBL's Brain Wide Map decoded block before the
   stimulus.

**Checks (test-only Poisson spike trains, never shown as data):**
- **By hand:** a tuning curve, AUROC with ties, and response rates.
- **Stratified permutations** keep every stratum's label counts, and are seeded.
- **Calibration:** p is uniform for side, choice, outcome and block (KS p from
  0.15 to 0.62). BH rejects in at most 5 of 40 all-null datasets.
- **Choice confound:** on units driven only by the stimulus side, a plain shuffle
  flags 100% as choice-selective; the stratified null flags 6.7%.
- **Block:** a plain permutation flags drifting units far more often than
  pseudo-sessions do. **The pseudo-session null is valid on average over block
  sequences, not for every sequence.**
  - Across 30 sequences, drifting units reached p < 0.05 3.3% of the time
    (median 0%, maximum 15%).
  - One atypical sequence gave 18%.
  - The test averages over 12 sequences.

**Real data (d23a44ef, 390 QC-passing units, stimulus onset):**

| Condition | Selective | Direction |
|---|---|---|
| side | 64 | 40 higher for right |
| choice | 55 | 32 higher for left |
| outcome | 3 | |
| block | 0 | pre-event window, smallest p 0.0034 |

**Limitation:** within strata, trials are treated as exchangeable. Slow drift in a
unit's rate is accounted for only by the block null.

### 2026-09-30 — Phy QC gains IBL's sliding refractory-period test (`qc/refractory.py`)

**Decision (the user's, of three options):** port ibllib's **MIT-licensed**
`brainbox.metrics.single_units.slidingRP_viol`, with attribution. It needs no new
dependency.
- **The test:** for refractory periods of 1.25–10 ms, a unit passes if its count of
  close spike pairs is at or below the 10% Poisson quantile expected at 10%
  contamination. Low-rate units fail by design.
- **Config:** `refractory_contamination: 0.1` and `refractory_alpha: 0.1` are in
  `configs/qc_phy.yaml`. The algorithm's fixed parts (20 kHz samples, 0.25 ms
  bins, the tested refractory periods) are constants in the module.
- **Correlogram:** phylib's autocorrelogram (BSD-3-Clause) is computed with sorted
  searches instead of its shift loop.

**Rejected:**
- **Depending on `slidingRP` 1.1.1:** it is GPL-3.0 and adds statsmodels and
  colorcet. Distributing Studio, for example in the planned installer, would then
  need a GPL-compatible licence, and the repo has none yet.
- **Hill et al. ISI violations:** simple, but not aligned with IBL's QC.

**Checks:**
- **Faithful port:** identical to ibllib's original code (run with phylib's
  original correlogram) on all 73 checked clusters of d23a44ef probe00.
- **Against IBL's stored flags:** the port agrees on 553 of 674 clusters. 94
  clusters pass only IBL's flag and 27 pass only the port. IBL's stored value
  comes from the newer GPL implementation, which works at 1/30,000 s resolution
  and passes more units, so the two are not the same test.
- **Real-data regression test:** it pins 553 of 674.

**Effect on Phy QC (probe00 of the Phy-format d23a44ef folder):** 161 units pass,
down from 200.
- 101 pass both Phy QC and IBL's `label == 1` (was 103).
- 60 pass only Phy QC (was 97).
- 13 pass only IBL's QC (was 11).

The rest of the gap is IBL's noise-cutoff and amplitude metrics. They need spike
amplitudes in volts, which a Phy folder doesn't give.

**Consequences:**
- Phy projects saved before this warn that `configs/qc_phy.yaml` changed when
  reopened.
- Studio's start-up on a Phy folder takes about 7 s longer for 674 clusters.

### 2026-09-30 — Probe filter in Studio

**Decision (the user's):** a probe filter, All or one probe. It applies to:
- the unit table and the region tree's counts;
- the population heatmap and mean;
- the 3D view (units, tracks and region meshes);
- the probe strip;
- the responsiveness test.

**How:**
- **One selection rule:** the filter is part of the server's unit selection, so
  every view uses the same units.
- **Testing:** results are keyed by (event, unit set, probe). Benjamini–Hochberg
  runs over exactly the units tested on that probe. An all-probe result is a
  separate test, never a subset of another. Changing probe clears the
  responsive-only filter, as changing event does.
- **Stripe colours:** a stripe beside the heatmap rows shows each row's probe,
  using the dataviz palette's categorical slots in their fixed order. Each
  probe's colour is set by its place in the session's probe list, so filtering
  never repaints a probe. A legend names every colour. Past eight probes the rest
  share the muted ink.
- **Captions:** the population caption names the probes included. The test
  summary names the probes tested.
- **Projects and exports:** the probe is part of the saved view. Version 1 files
  written before this open on all probes. The export's `population.json` lists
  each row's probe.

**Result (d23a44ef, First movement):** probe01 alone gives 241 of 277 units
responsive (86 up, 155 down), with BH over 277 tests.

### 2026-09-30 — Project files and figure export (`studio/project.py`, `studio/export.py`)

**Project file (`*.ndstudio.json`):**
- **What it holds:**
  - the data source (an IBL eid and backend with its release, or a Phy folder
    plus events CSV);
  - the sha256 of each Phy file that changes what Studio shows, and of the
    events CSV;
  - a fingerprint of the loaded Session (every spike train, the units and
    trials tables);
  - the QC and analysis config hashes;
  - the view: event, window, bin, baseline, level, node, filters, unit.
- **No results.** Opening a project reloads and recomputes everything.
- **What changed is named, not blocked.** "`events.csv` changed", "`cluster_group.tsv`
  is new", "the loaded data differ", "`configs/qc.yaml` changed". These appear in
  a banner and on the console.
- A view that relied on the responsiveness filter reruns the test on opening,
  because results are never stored.
- **Writing:** saves are atomic, and the file name must end in `.ndstudio.json`.

**Where projects go:**
- `--project FILE` alone opens a project.
- With `--eid` or `--phy`, it starts a new one there. It refuses if FILE exists.
- Without `--project`, Save writes to `data_root/projects/<name>.ndstudio.json`,
  numbered so nothing is overwritten.

This settles step 1's open folder-picker question for now: paths come from the
command line, not a dialog.

**Figure export:**
- **Destination:** each export is a new `runs/<UTC time>_studio/` folder
  (`runs/` is gitignored, as §7 expects).
- **Figures:** `unit` and `population` as SVG and PDF, drawn by the same viz/
  builders as the page. They use the light theme at a fixed size, are titled
  with the page caption, and keep text editable (`svg.fonttype none`,
  `pdf.fonttype 42`).
- **Sidecars:** a JSON file beside each figure holds every array it was drawn
  from: bin centres, mean, SEM, trial counts, raster times, unit order, rates and
  scaled rows. Tests check the sidecars equal the engine's output.
- **`responsiveness.csv`** is included when the test ran for that event.
- **`manifest.json`:** git SHA and dirty flag, the project snapshot, the view,
  the responsiveness config, the software versions, and `seed: null` (nothing
  random).

**Safety:** the two write endpoints (`POST /api/project`, `/api/export`) accept
only `application/json` from the page's own origin. A cross-site page can't
trigger them, and a test covers both refusals.

**Duplication, knowingly:** `_git` and `_jsonable` are small copies of those in
`cli/evaluate.py`. Importing them would pull the evaluation stack into Studio, and
moving them into a shared module means refactoring validated code outside this
step.

**Checked on real data:**
- **Save and reopen:** saving d23a44ef at First movement / CA1 / `probe00_433`
  and reopening restored the view with no warnings.
- **A changed file:** a Phy project whose events copy lost its last trial
  reopened with both warnings, and with 409 trials recomputed.

### 2026-09-30 — Responsiveness against a shift null (`analysis/responsiveness.py`, `configs/analysis.yaml`)

**Decision:**
- **Statistic, per unit and event:** the mean over trials of (rate in 0–300 ms −
  rate in −200–0 ms), two-sided.
- **Null:** the spike train circularly shifted against the event times. It keeps
  the train's own timing and the events' spacing, and breaks their alignment.
- **p** = (1 + #shifts with |statistic| ≥ |observed|) / (1 + #shifts).
- **Correction:** Benjamini–Hochberg across the units tested together, α = 0.05.
- **Reported with every result:** the windows, the number of shifts, the number
  of tests and the trial counts.

**Different from the plan:** the plan was 1,000 random shifts with a seed. That
floors p at about 0.001, and with 390 units under BH it would need about 10,000
shifts per unit. Instead, **every** circular shift on a grid is evaluated at
once, by FFT cross-correlation. Shifts closer than `min_shift_s` = 10 s to zero,
either way, are excluded.
- **Deterministic:** no random draws and no seed, so R7 is satisfied trivially.
- **Grid:** 5 ms, 456,801 shifts per unit on d23a44ef. On stimulus onset, 1 ms
  gave 321 responsive units, 2 ms 322 and 5 ms 320. 5 ms runs in 11 s for 390
  units, against 52 s at 1 ms.
- The statistic is computed on that grid. A test checks it against direct
  counting.
- Units are processed 8 at a time, since each null has millions of values.

**Checks:**
- **Hand-computed case:** one spike after every event gives p = 1/(1 + n_shifts)
  exactly.
- **Calibration:** Poisson units with no event locking give uniform p (KS test).
  BH rejects in at most 5 of 40 all-null datasets.
- **Real-data negative control:** 410 random fake event times gave 0 of 390
  units responsive for each of two seeds, and 15–16 with uncorrected p < 0.05
  (about 4%).
- **Cross-check:** a per-trial paired Wilcoxon signed-rank test with BH gives
  328 of 390 at stimulus onset (shift null: 321 at 1 ms) and 175 at error
  feedback (178).

**Finding (d23a44ef, 390 QC-passing units, 5 ms grid):**
- **Stimulus onset:** 320 of 390 responsive (154 up, 166 down).
- **At 1 ms:** first movement 329, reward 311, error 178.

That is a large fraction. The baseline window falls in IBL's enforced quiescence
period, and movement follows within a few hundred ms, so "responsive to stimulus
onset" includes movement-related change. The UI and captions say "rate changes
around this event", never "responds to the stimulus". Phy-format probe00: 117 of
200.

**Cross-validated heatmap sorting:** the population heatmap is now sorted by peak
time on odd trials and shows even trials, and its caption says so. This fixes
the prototype's circularity. The diagonal persists on held-out trials at
d23a44ef's stimulus onset.

**Server:** now threaded (`ThreadingHTTPServer`), so an 11 s test doesn't freeze
the plots. Figures use matplotlib's object API, with no shared pyplot state.

**Not done:** the refractory-period metric for Phy QC, which the plan suggested
considering. It is still open.

### 2026-09-30 — Atlas and 3D view, built (`analysis/atlas.py`, `data/atlas_meshes.py`, `studio/`)

**What was built:**
- **Region levels:** Allen, Beryl or Cosmos, default Beryl.
- **Region tree:** counts per node, and selecting a node includes its
  descendants.
- **3D brain (three.js):** whole-brain and region meshes in Allen colours, the
  units, both probe tracks, and the selected unit. Clicking a unit selects it.
- **Probe strip:** region runs along the shank.
- **Redesign:** a left rail, cards, and light and dark themes from the dataviz
  reference palette. Plots use a single-hue blue ramp for magnitude, blue–grey–red
  for signed values, and a muted event line.

**Decisions:**
- **Coordinates:** IBL xyz → CCF µm uses iblatlas's bregma landmark. A test checks
  it against `AllenAtlas.xyz2ccf`.
- **Real-data check:** all 398 d23a44ef units land in a voxel of their own Allen
  region in the 25 µm annotation volume (`tests/test_atlas.py`).
- **Meshes:** Allen's per-structure OBJ files, downloaded on first view into
  `data_root/atlas/ccf_2017_meshes/` (the user approved the downloads). They are
  written atomically and refused if the content isn't an OBJ. All three levels
  for d23a44ef, plus the whole brain, came to 30 files and 27 MB.
- **Volumes:** the Allen volumes for the test (`annotation_25.nrrd`,
  `average_template_25.nrrd`, 37 MB) are in `data_root/atlas/`.
- **Region runs on the probe strip** span recorded units only. They are not
  histological boundaries, and the strip says so.
- **Allen's root colour is white,** so the page draws it in muted ink.
- **The page computes no numbers.** Regions, trees, positions, tracks and heatmap
  row order all come from the API. Static files are served only from
  `studio/static/`, which a test guards.

**Finding:** at Beryl, 73 of the 390 QC-passing units are at `root`.
- 24 are in fibre tracts: ml, alv, sptV, icp, arb.
- 49 carry only a coarse parent label: MY 31, CB 17, TH 1.
- At Cosmos only the 24 fibre-tract units remain at root.

The tree shows these as "in no Beryl region" rather than dropping them. At Allen
level, the coarse parents' meshes (CB, MY, TH) are large and dominate the 3D view.

**Consequences:** CLAUDE.md §8's Studio line now names three.js. The Phy data has
no positions or regions, so its 3D view, levels and tree are disabled, each with
its reason.

### 2026-09-30 — Studio step added: Atlas and 3D view; three.js for the 3D brain

**Decision (the user's):** a new step 2 in `docs/proposals/studio_next_steps.md`,
after the Phy import. It adds:
- a region level selector (Allen, Beryl, Cosmos; **default Beryl**), with Allen
  names and colours from iblatlas;
- a hierarchical region filter;
- a 3D brain from the Allen CCF meshes, cached locally, showing each probe track
  and the selected unit's site;
- a 2D probe strip;
- a visual redesign.

Responsiveness and the project file move to steps 3 and 4.

**three.js** will draw the 3D brain. It has no Python-side equivalent that fits
a browser UI. It is MIT licensed, and it runs as an ES module with no build
step, so §8's "plain page, no build step" holds. It will be kept in the repo as a
local file rather than loaded from a CDN, because Studio is a local app that must
work offline. **Installed 2026-09-30:** r170 (npm `three@0.170.0`),
`three.module.min.js`, `OrbitControls.js` and `OBJLoader.js`, with its MIT
`LICENSE`, in `neurodecoder/studio/static/vendor/three/`.

**Chart library: none.** Decided at the start of the step, following the plan's
recommendation. Plots stay matplotlib PNGs. Heatmap rows are clickable and
hoverable through a row→unit map the server sends with the image.

**Consequences:**
- The Allen CCF 2017 structure meshes are a new external data source. They are
  downloaded once into `data_root/atlas/`, never into the repo.
- iblatlas's annotation volume is needed for the coordinate test.
- Unit QC keeps reading the Allen acronym, whatever level is displayed.

### 2026-09-30 — Phy import and Phy QC (`data/backends/phy.py`, `qc/phy.py`, `configs/qc_phy.yaml`)

**Decision:** Studio reads a Kilosort/Phy output folder (one probe) plus a CSV of
trial events into the same `Session` as the other backends. It adds a Phy-specific
unit QC (the user's choice over failing every unit):
- the group must be in `[good]`;
- the task-period rate must be at least 0.1 Hz, with the same task-period
  definition and threshold as `configs/qc.yaml`.

**How it reads a folder:**
- `params.py` is parsed with `ast`, never executed. Phy itself executes it.
- `spike_times.npy` must hold integer samples.
- **Group:** from `cluster_group.tsv`, else `cluster_KSLabel.tsv`, else missing.
  `group_file` records which file. Kilosort writes `cluster_group.tsv` as a copy
  of its own labels, so a group is not proof of manual curation.
- **Depth:** the peak channel of the cluster's most-used template. The plan said
  amplitude-weighted. Peak channel needs no amplitude threshold, and Phy splits
  and merges don't break it.
- **Events:** the CSV uses canonical trial names and needs `intervals_0` and
  `intervals_1`. Events outside the span of the recorded spikes are refused as
  being on the wrong clock. A smaller clock offset can't be detected; that needs
  sync, a later step.
- **Missing fields:** region, IBL label, 3-D position and whole-recording rate
  are declared missing.
- **No cache:** a folder loads in seconds.

**Real-data check:** IBL's own Kilosort output for d23a44ef probe00 (ONE,
revision 2024-05-06), rewritten into Phy's format. It is not a folder Kilosort
or Phy wrote.
- `tests/test_phy.py`: all 674 clusters, identical spike counts, times within
  half a sample, and Kilosort's 206 good.
- PSTHs match the BWM backend's for the 114 shared units within bin-edge
  rounding: at most 2 spikes in one bin.
- A copy for trying Studio is in `data_root/derived/phy_export_d23a44ef/`, with
  a README stating its provenance.

**Finding:**
- Kilosort's `good` is much looser than IBL's QC label. On probe00, Phy QC passes
  200 units and IBL's `label == 1` (the BWM release) 114.
- 103 pass both, 97 pass only Phy QC, and 11 pass only IBL's QC.
- Phy QC is therefore not equivalent to the BWM units. A refractory-period
  metric from spike times is proposed for step 3.

**No new dependency. No `PREPROC_VERSION` bump:** a new config for a new data
source changes no existing output.

**Alternatives considered:**
- `spikeinterface`'s Phy reader: a large dependency for a few `.npy` files, and
  it executes `params.py`.
- Mapping Phy groups onto IBL's numeric label: rejected, because they measure
  different things.

### 2026-09-30 — Direction change: Neurodecoder Studio, a post-sorting analysis app

**Decision (the user's):** Neurodecoder becomes **Neurodecoder Studio**, a local
app for analysing data after spike sorting. You load sorted Neuropixels units plus
task events, browse units, and run event-aligned and tuning analyses in a GUI, with
strict statistics. Spike sorting and curation stay in existing tools (Kilosort,
Phy). This reverses CLAUDE.md §2's "not a general neuroscience analysis library" and
narrows §1's cross-animal reliability question to parked work. CLAUDE.md §1–§3 and
§5–§9 were updated to match, after the prototype. First step: a prototype on branch
`studio-prototype`, from `phase3-gate-audit`.

**Why:** Phase 3's gate did not pass on the confirmation set
(`docs/NEGATIVE_RESULTS.md`, 2026-09-30). Only movement state beat
`null_trialstruct`, so the reliability layer has no decoding signal to be reliable
about yet.

**Kept:** the data layer (`data/`, the session contract, cache, backends), unit QC
(`qc/`), R2, R4, R6 and R7, and "missing means missing". R4 now applies to
analyses: responsiveness claims need a shuffle null (next step, not the prototype).

**Stack for the prototype: no new dependency.**
- Python's standard-library `http.server` serves one HTML page and three endpoints.
- matplotlib, already in §8, renders the plots server-side as PNGs.
- The page is plain HTML and JavaScript, with no build step.

The UI calls `neurodecoder/analysis/` only. Streamlit, Panel and NiceGUI were each
one large dependency; none was needed for three plots and a table. Plotly stays
excluded (§8). Revisit if interactivity like zoom or hover becomes necessary.

**Analysis parameters are not preprocessing:** PSTH window, bin width and baseline
are chosen per plot and never written to a cache, so R6's `PREPROC_VERSION` does
not apply to them. The heatmap's row scaling is for display only
(`scale_rows_for_display`), not an R3 normalisation.

**Known caveat, not fixed in the prototype:** the population heatmap is sorted by
peak time on the same trials it displays. That produces a diagonal even from noise.
Cross-validated sorting (sort on half the trials, show the other half) belongs with
the shuffle-null step.

**Alternatives considered:** Streamlit (fastest to write, but a heavy dependency
and a rerun-the-script model); a desktop Qt app (packaging cost, and Phase 8b
already chose a local browser UI).

**Consequences:** Phases 4–7 and 9 are parked with Phase 3. Phase 8b's local-app
plan (installer, refusal screens, front end only) carries over to Studio.

### 2026-09-30 — Block split: adjacent pairs, scored per fold — made AFTER the confirmation set

**Made after seeing the confirmation set's block table**
(`runs/20260929T114437Z_phase3_confirmation/block`: `model` AUROC 0.000 in 9 of 10
sessions, 0.019 in the tenth). That result exposed a defect in the split as first built
(`docs/NEGATIVE_RESULTS.md`, 2026-09-30). The fix was accepted on a synthetic
no-signal test, never on a real-data score. It changes block only: no null,
threshold, target or split changed for any other target.

**Decision** (the user chose pairs, then per-fold scoring after pairs alone failed
the no-signal test):
- **`leave_one_block_out` holds out adjacent pairs of biased blocks**
  (`splits/registry.py`): one 0.2 and one 0.8 block per fold; with an odd count,
  the last block joins the final pair. Every biased block is tested exactly once,
  with the same 2 s gap either side. The split's params record
  `blocks_per_fold: 2`, so its hash differs from the first version's.
- **The guard rejects a fold whose test or training trials hold one label**
  (`splits/guards.py`). Each session record now keeps every trial's
  `probabilityLeft` (`trial_prior`), so the check reads the labels. A split
  without it (the first version) is refused.
- **The contract scores each fold separately** (`evaluation/contract.py`). A
  session's metrics are the mean over its folds whose primary metric is defined.
  A fold whose test labels hold one class has no AUROC, which can happen with
  shifted or pseudo labels. `n_samples` and `n_folds` count the scored folds.
  Providers with one fold (every other split) are scored exactly as before.
- The per-fold normaliser and the other guard checks are unchanged.

**Why per-fold scoring:** each fold's model has its own offset, set by its
training class balance. Pooling lets those offsets rank samples across folds,
which inverted decoders with no signal (single blocks: AUROC ~0) and still biased
them with pairs (noise medians 0.34–0.41). Within a fold, every sample shares the
offset.

**Acceptance test:** `tests/test_lobo_no_signal.py`. Pure Poisson noise with
IBL-generated block labels goes through the real split, provider, decoders and
`evaluate`. `model`, `model_with_task`, `null_shuffle` and `null_pseudosession`
must each score a median AUROC within 0.5 ± 0.07 over 6 sessions. They score
0.489, 0.532, 0.519 and 0.499; `null_trialstruct` scores 0.491 and
`baseline_rrr` 0.499.

**Alternatives considered:**
- Weighting both classes equally when fitting. It changes every block decoder
  and was not tested.
- Dropping the split and returning block to within-session with the
  pseudo-session null.

**Consequences:**
- The block rows of both runs on the first version are invalid.
- Block is re-run on the confirmation set only
  (`runs/20260930T082034Z_phase3_confirmation_block_pairs`). The first table is
  already seen, and choice and wheel velocity already fail the gate there, so
  block can't change its outcome.
- A per-session table from a fold split gains an `n_folds` column.
- The no-signal test adds about 3 minutes to the suite.

### 2026-09-29 — Phase 3 gate: the incremental row, leave-one-block-out and pseudo-sessions for block — adopted AFTER the first table

**Made after seeing the first table** (`runs/20260929T070943Z_phase3_first_table`,
gate not passed, `docs/NEGATIVE_RESULTS.md`). Proposed in
`docs/proposals/phase3_gate_methods.md`; the user adopted (a), B1 and B2 and
rejected a blocked scheme for per-bin targets. Changing the evaluation after a
result is itself a degree of freedom (split doc, route 4), so:
- **no row, null, threshold, target or existing split was changed or removed**;
  every change below adds something;
- **the gate call rests on the confirmation set**
  (`configs/runs/phase3_confirmation.yaml`: 10 sessions drawn with seed 1 from
  the manifest minus the first table's 10, before any of this was built). The
  first table's sessions are "seen" and are re-run for comparison only.

**(a) `model_with_task`, and the gate moves to it.**
- A new row: the model under test given its spike features **and** the
  `null_trialstruct` features, on the same split, test samples and gapped
  training CV (`SpikesAndTaskRidge` / `SpikesAndTaskLogistic`).
- **Two penalties.** The task block's per-feature penalty is `ratio × λ`,
  implemented by scaling the standardised task columns by 1/√ratio. The ratio
  (10^-2 … 10^2, 5 values, `configs/baselines.yaml: with_task`) is chosen with
  λ by the same training CV. Without it, one shared λ over ~100–400 spike
  features and ~30 task features would make "adding spikes" look harmful for
  fitting reasons.
- **Both verdicts are printed**: `model vs null_trialstruct` (spikes alone vs
  task, unchanged) and `model_with_task vs null_trialstruct` (does neural
  activity add information beyond the task?).
- **`GATE = (model_with_task, null_trialstruct)`** for every target, because
  that is the question CLAUDE.md §5 and split-doc route 1 ask. Each report ends
  with `GATE (...): PASSED / NOT PASSED`. Passing the spikes-alone verdict stays
  the stronger, separately reported claim.

**B1. Leave-one-block-out, block only** (`splits/registry.py`:
`leave_one_block_out`; escalated per CLAUDE.md §10 and approved).
- Blocks are runs of `probabilityLeft ∈ {0.2, 0.8}` among the session's trials;
  the 90-trial unbiased start is in no block.
- Each biased block is one fold's test set, from its first trial's start to its
  last trial's end. Training is every earlier trial that ends at least `gap_s`
  (2 s, as in the within-session split) of empty bins before the test block,
  and every later trial that starts that far after it. Trials in the gaps are
  in neither partition for that fold. The guard also requires the gap to be at
  least the context window, and sample windows must lie inside their
  partition's intervals.
- Sessions need ≥ 4 biased blocks (`MIN_LOBO_BLOCKS`), otherwise the builder
  raises.
- The guard (`_check_session_folds`) checks per fold: no training interval
  shares a bin with the test block, the gap holds, no trial is in both, and no
  trial is tested twice.
- The normaliser is fit per fold on that fold's training data only (R3),
  hashed as `<split hash>#fold<k>`, and every fold's normaliser is saved
  (`normalizers.json`).
- ~~Per-session metrics pool every fold's test predictions, so each block is
  scored once and every session has both classes.~~ **Superseded 2026-09-30:**
  single-block folds with pooled scoring inverted decoders with no signal. The
  split now holds out adjacent pairs, and each fold is scored on its own (entry
  "Block split: adjacent pairs, scored per fold").
- Refused for per-bin targets. Choice, wheel velocity and movement state stay
  on the within-session split.
- The split is its own kind with its own hash (`split_leave_one_block_out.json`,
  recorded in the manifest); the within-session split file is unchanged.

**B2. `null_pseudosession`, block only** (`evaluation/nulls.py`,
`configs/nulls.yaml: null_pseudosession`).
- 100 pseudo block sequences per session from a seeded port of brainbox's
  `generate_pseudo_blocks`: 90 unbiased trials, then alternating blocks, first
  side at random, lengths exponential(60) truncated to (20, 100). Seeds are
  derived from the run seed and the eid.
- The model is refit and scored on each, with the same pipeline, split and
  neural data. The row is the per-session median, with its own
  `model vs null_pseudosession` verdict. It sits beside `null_shuffle`, which is
  unchanged.
- Needs at least `MIN_PSEUDO` = 20 pseudo-sessions.
- **Checked against the release:** block lengths from the port match the BWM
  sessions' (KS p = 0.107). The test compares **distinct** sequences only: BWM
  ephys sessions reuse pre-generated block sequences (18 distinct openings, 118
  distinct full sequences), so the sessions' block lengths are not independent
  draws, and a naive KS test rejects (p = 0.0001) for that reason alone.

**Also:** the per-session baselines fit sessions in parallel
(`configs/baselines.yaml: n_jobs: 6`) through `joblib`, which scikit-learn
already installs and depends on; results are identical to sequential fitting
(tested). Block LOBO with pseudo-sessions needs thousands of logistic fits per
session.

**Alternatives considered:**
- Replacing or weakening `null_trialstruct`: ruled out.
- B1 or B2 alone: B2 on the old split still scores on 1–4 test blocks, and B1
  without B2 leaves slow drift to the circular shift.
- A blocked within-session scheme for per-bin targets: not adopted.

**Consequences:**
- Reports now have up to eight rows: `null_pseudosession` for block,
  `model_with_task` for every target.
- `metrics.json` gains `gate`, `n_folds`, `n_pseudo` and `normalizer_hashes`
  (a list, replacing `normalizer_hash`); the manifest gains
  `split_leave_one_block_out_hash` and each target's split hash.
- Both run configs gain `split.leave_one_block_out: [block]`.

### 2026-09-29 — Local app for non-programmers (Phase 8b)

**Decision:** build a **local app**: a browser UI that runs on the user's own
machine, shipped as an installer. Not a hosted web app. This reverses CLAUDE.md
§2's "not a web app" for this one case; hosted web apps stay excluded.
- There are no server costs.
- Data stays in the lab.
- Multi-GB NWB files never need uploading.

**Purpose:** a non-programmer picks a recording (an IBL session or a supported
NWB file), picks a target, presses run, and reads a report with clear trust
flags. They can also browse past runs, read-only.

**Rules the app must keep:**
- **It is a front end only.** It calls CLI entry points (Phase 8's
  `neurodecoder analyze`) and reads their artifacts in `runs/`. It has no data,
  split, training or evaluation path of its own, so every result it shows has
  still passed:
  - the split registry;
  - `assert_split_valid`;
  - the six-row contract.
- **It always shows the full six-row table and the null verdicts,** never a
  bare score.
- **The LLM boundary (§6) is unchanged:** report prose comes only from
  artifacts.
- **Unsupported input gets a refusal screen that states the reason,** per
  Phase 11's "I can predict A and B, not C, and here is why". Never a
  traceback.

**Installation is the hardest part.** It needs:
- Python 3.11;
- PyTorch;
- the IBL stack with its llvmlite/numba pins (see "`ONE-api`/`ibllib` installs
  on this machine need llvmlite/numba pinned first");
- gigabytes of data.

Plan a one-step installer for Mac, Windows and Linux, with in-app data download
and progress. Packaging options to evaluate later: conda constructor, pixi,
PyInstaller.

**UI framework:** Streamlit, Panel, NiceGUI or similar, decided at Phase 8b,
not now.

**Timing:** build after Phase 8, never before Phases 4–7, which change what the
app shows. Estimate 60–120 h including packaging.

**Alternatives considered:**
- **A hosted web app:** compute costs, data uploads, and data governance.
- **A read-only viewer only:** doesn't serve non-programmers, who need to start
  a run.

### 2026-09-29 — NWB probe, and what full NWB intake would take (`neurodecoder/nwb/probe.py`)

**Decision:** `probe(path or URL)` / `python -m neurodecoder.nwb.probe` reports
what an arbitrary NWB file holds. It is thin, as the roadmap says: no
normalisation, no model, no `Session`. It reads metadata and small samples
only; a whole spike-times or raw-data array is never loaded, and URLs are
streamed with remfile. Any section that can't be read goes under `errors`,
and an unreadable file is a result, never an exception. It reports:
- **units:** count, columns, spike count (the ragged column's target, not its
  per-unit index);
- **QC columns:** any units column whose name contains "label" or "quality",
  which catches `quality`, `KSLabel`, and IBL's `ibl_quality_score` and
  `kilosort2_label`;
- **location:** per-unit location, from a units column or via
  `units.electrodes`;
- **intervals:** trials and other interval tables;
- **series:** every time series in acquisition and processing (shape, unit,
  rate, duration);
- **behaviour mapping:** name matches onto the pipeline's behaviour fields;
- **a usability tier.**

**Survey** (2026-09-29; the ROADMAP asks for ≥ 5 foreign files classified
without crashing, and that is met):

| File (dandiset, version) | Size | Units / spikes | QC column | Unit location | Intervals | Behaviour matched | Tier |
|---|---|---|---|---|---|---|---|
| 000409 `d23a44ef` (local) | — | 1,961 / 62.0M | `ibl_quality_score`, `kilosort2_label` | electrodes via units | trials 410 | wheel, pose, pupil, motion energy | decodable |
| 000409 `db4df448` (streamed, 72 s) | — | 709 / 15.6M | same | same | trials 402 | same | decodable |
| 000409 `3638d102` (streamed, 88 s) | — | 860 / 20.2M | same | same | trials 695 | same | decodable |
| 000017 Steinmetz 2019, `sub-Richards_ses-20171031` | 204 MB | 778 / 4.7M | none | electrodes via units | trials 260 | wheel, lick, pupil, motion energy | decodable |
| 000021 Allen Visual Coding Neuropixels, `ses-721123822` | 1.7 GB | 1,603 / 79.8M | `quality` | **none** (units link by `peak_channel_id`, not `electrodes`) | stimulus tables only, no trials | running wheel | partial |
| 000053 MEC linear track, `sub-npI1_ses-20190416` | 71.7 GB | 408 / 3.9M | none | **none** | trials 149 | none (track position isn't a pipeline field) | partial |
| 000059 medial septum cooling, `sub-MS10` processed | 3 MB | **no units table** (trials, position, speed, temperature only) | — | — | trials 142 | none | no units |
| 000128 MC_Maze, `desc-train` | 691 MB | 182 / 3.6M | none | electrodes via units | trials 2,295 | none (hand, cursor and eye position) | partial |

**What full intake would take** (for the Phase 11 plan):
1. **Spikes and units: small.** The standard `nwb.units` table was readable in
   every file that had one (6 of 7).
2. **Unit QC: needs a policy decision.** 4 of the 5 foreign files have no QC
   label at all, and Allen's `quality` means something different from IBL's
   label. Under our rule a missing label fails the unit, so today's QC would
   reject every unit in those files.
3. **Location: moderate, per dataset.** Two of the five foreign files have no
   standard per-unit location (Allen links units through `peak_channel_id`).
   Region names also need mapping to Beryl, and the conventions differ between
   datasets.
4. **Targets: the large part, and task-specific.**
   - Only Steinmetz 2019 shares IBL's wheel task.
   - The others need their own targets (reach kinematics, track position,
     running speed, stimulus identity) and trial definitions.
   - Name-matching behaviour to our fields is only a triage heuristic.

**Estimate:** spikes, units and location for standard files take about a
week. Each new task needs its own targets and a QC policy, a few days to a
week per dataset. A general "any NWB" decoder isn't realistic; intake per
dataset family is.

### 2026-09-29 — Running the contract and logging runs (`neurodecoder/cli/evaluate.py`, `configs/runs/`)

**Decision:** `python -m neurodecoder.cli.evaluate <run config>` runs the
contract for each target in the config and logs the run under
`runs/<UTC time>_<name>/` (CLAUDE.md §7).

- **`manifest.json`** is written first with status "running" and updated after
  each target. It records:
  - the git SHA, branch and whether the tree was dirty, plus the command;
  - the seed;
  - every config file used (`run`, `qc`, `preprocess`, `targets`, `nulls`,
    `evaluation`, `baselines`), with its sha256 and content;
  - `PREPROC_VERSION`, the preprocessing and target fingerprints, the split
    hash and the manifest provenance;
  - the sessions, library versions and per-target timings.
- **`split.json`** is the saved split.
- **Per target** (`<target>/`):
  - `report.txt` is the six-row table and verdicts;
  - `metrics.json` (strict JSON, NaN → null) holds every per-session metric of
    every row, the summary, the verdicts and any dropped trials. These are the
    numbers any later report must trace back to (§6);
  - `per_session.parquet` and `shuffle.parquet` hold the same numbers as tables.
  - `normalizer.json` (added after the first-table audit, R3) is the Normalizer
    fit on the split's training data. It is saved with its hash, which is also
    recorded in `metrics.json` and the manifest; `Normalizer.from_dict`
    re-checks it on load. The first-table run predates it: its normaliser was
    recomputed deterministically afterwards and saved as
    `normalizer.backfilled.json`, marked as recomputed.
- The code is committed before a long run, so the recorded SHA reproduces it.

**The first table's run config** (`configs/runs/phase3_first_table.yaml`;
the user chose each option below before the run):
- **Sessions:** NEDS's held-out test sessions that `bwm_ephys` 1.2.1 contains
  (8 of 10; `f140a2ec` and `d04feec7` are not in the release), plus 2 drawn
  at random with seed 0. So the table doubles as the base for the NEDS
  comparison.
- **Split:** within-session, 80% of trials train, with a 2 s gap.
- **Targets:** choice, block, wheel velocity, movement state, in that order.
- **Movement state trains on every 5th bin:** about 20 min per session with 20
  shuffle refits, against about 2 h on every bin. Evaluation stays on every
  task-period bin.
- **The model row is the ridge/logistic baseline itself.** Phase 3's number is
  the baseline's. `baseline_ridge` is therefore the same decoder, and each
  report says that verdict is vacuous.
- **20 shift draws**, as in `configs/nulls.yaml`.

### 2026-09-29 — The data provider, trial-structure decoders, and a contract fix (`neurodecoder/evaluation/data.py`)

**Decision:** `SplitData(split, target, context_bins=…)` is the real
`DataProvider`. It prepares every session of the split once:
- load it through the BWM backend;
- apply unit QC and bin it (`preprocess_session`);
- build the target (`targets/`);
- fit one `Normalizer` on the split's training data.

It keeps binned counts and targets, not spike trains, and normalises on each
request (about 150 MB per session instead of about 500 MB). Preparing
everything up front suits Phase 3's within-session split; Phase 4's
cross-session runs over hundreds of sessions will need a lazier provider.

- **Per-bin samples** (wheel velocity, movement state) are every window end in
  the partition's span whose bin lies in the task period and has a defined
  target. That's "task period, every bin" as the user chose.
  - `train_stride` (default 1) thins training samples only. How dense movement
    state's training should be is still the user's call.
- **Trial samples** (choice, block) are the target's usable trials that the
  split lists in the partition, at the target's own window. The context is
  forced to match the window: 5 bins for choice, 15 for block.
  - A trial whose window reaches outside the partition's span is dropped and
    counted (`dropped`), never silently. In the fixture, that happens when
    block's window starts before the train block does.
- **Shifts:**
  - shifted samples use the null's rotation (`evaluation.nulls`) at the same
    ends; a bin whose rotated target is undefined is dropped;
  - trial labels rotate over the session's usable trials before the partition
    filter;
  - `shifts()` draws within the task period (per-bin) or over the usable trials
    (trial targets).
- **`TrialStructureRidge` / `TrialStructureLogistic`** are the
  `null_trialstruct` decoders. They're the per-session baselines fitted on
  `task_features` only, with the same gapped CV; the contract hands them data
  with `z = None`.
- **Contract fix (from #22), found by the end-to-end test.** A shift draw that
  no session could take crashed the contract. That happens whenever every
  session is too short, e.g. choice with fewer than 200 usable trials. Such a
  draw is now undefined: NaN for every session, and the verdict line says
  "undefined, no test session has a defined null_shuffle" instead of claiming
  the model failed to beat it.

### 2026-09-28 — Baselines: ridge/logistic and multi-session RRR (`neurodecoder/models/baselines/`, `configs/baselines.yaml`)

**Decision (the user chose each option below before any code):**

| Choice | Setting |
|---|---|
| Per-bin inputs (wheel velocity, movement state) | each unit's summed activity in five 200 ms chunks of the last 1 s (context 50 bins) |
| Trial inputs (choice, block) | each unit's count over the target's window (the Brain Wide Map decoders' input); RRR sees the window bin by bin |
| Tuning | gapped blocked 5-fold CV inside the training data only: contiguous folds, with a max(context, 2 s) gap dropped either side of each validation fold; lowest mean validation MSE / log loss wins |
| Scope | ridge/logistic and RRR; the Poisson GLM is deferred (not a contract row, and it's an encoding model) |

- **Per-session models.** Ridge and logistic are fit per session: per-unit
  weights don't transfer across sessions. Predicting a session the model wasn't
  trained on raises, which is enough for Phase 3's within-session table. **Cross-session
  baselines (e.g. region-pooled features) are a Phase 4 decision.**
- **Penalty scale, corrected before any test metric existed.** The first grid
  put α on sklearn's summed-loss scale, from 10⁻³ to 10⁵. On `d23a44ef`'s wheel
  velocity, training-only CV picked 10⁵, the top edge; the CV optimum was 10⁶,
  and the edge cost 24% in validation MSE. Features are now standardised on
  the training data, and the penalty is per sample (mean loss + λ‖w‖²), with λ
  from 10⁻⁴ to 10⁴ in 17 steps. So one λ means the same for 230 trials or 78k
  bins. The same session now picks λ = 31.6, well inside.
- **Edge flag:** a λ at either end of the grid is flagged in the model
  (`at_grid_edge`), not silently kept.
- **Ridge** solves every λ and fold from one Gram matrix per session (19 s on
  `d23a44ef`'s 78,836 training bins with 1,950 features). It matches sklearn's
  `Ridge` on standardised features.
- **Logistic early stopping.** Logistic walks λ from strong to weak,
  warm-started, and stops a fold once its validation loss has risen twice in a
  row. The unreached weak end is where fits are slowest and never chosen; a λ
  not reached by every fold can't be selected. Movement state on `d23a44ef` costs:
  - 56 s using every 5th bin;
  - 297 s using every bin, with λ = 0.32 either way.

  **Open question for the first table:** with 20 shuffle refits of the model
  under test, training on every bin costs ~2 h per session for movement state.
- **RRR in two steps.** Session-specific unit weights U_s and shared temporal
  filters V (weights U_s Vᵀ over unit × chunk):
  1. per-session ridge on all features;
  2. V = top-r right singular vectors of the stacked session weights;
  3. U_s refit on X_s V (ridge, or logistic for classification) with the
     session's step-1 λ.

  The rank comes from {1, 2, 3} by the same folds (V re-estimated per fold), on
  the mean over sessions of relative validation MSE (or log loss). Alternating
  least squares with the same CV would cost hours on per-bin targets.
- **RRR measured:** 64 s per session on wheel velocity. On `d23a44ef` plus one
  other session it chose rank 1, with a filter weighted on the last 200–400 ms.
  Synthetic tests check it recovers planted rank-1 and rank-2 shared filters.

### 2026-09-28 — The evaluation contract (`neurodecoder/evaluation/contract.py`)

**Decision (the user chose each option below before any code):**
`evaluate(task, *, model, baseline_ridge, baseline_rrr, trialstruct, ceiling,
seed)` is the only entry point that scores a model. Every row is a required
argument, and it always fits and scores all six rows in CLAUDE.md §5 order.

| Row | What is fitted |
|---|---|
| `null_shuffle` | **the model under test**, refit once per shift draw on targets shifted within each session (`evaluation.nulls`); per-session median over draws |
| `null_trialstruct` | the trial-structure decoder, given each session's data **with spikes and units removed** (`z = None`) |
| `baseline_ridge`, `baseline_rrr` | the baselines, same split |
| `model` | the model under test |
| `ceiling_within` | the model on a within-session split of exactly the test sessions; for a within-session task it *is* the model row, and the report says so |

- **Fits receive the train partition only**, loaded one session at a time
  (`SessionData`, via a `DataProvider`). Predictions are scored per test
  session, and invalid predictions (wrong shape, NaN, a probability outside
  [0, 1]) raise.
- **The shuffle row re-trains the model under test**, so it measures what that
  model extracts from target autocorrelation alone.
  - It uses 20 draws by default. A costly model may use fewer, but never fewer
    than 5.
  - A session's k-th draw is the same shift for its train and test data, so a
    within-session model sees one consistent rotation.
  - Draws are seeded per session from the run seed.
  - A session without a valid shift sits the null out: it isn't used for
    training that draw, and its shuffle metric is NaN.
- **Verdicts:** for each of `null_trialstruct`, `null_shuffle` and
  `baseline_ridge`:
  - the test is a one-sided Wilcoxon signed-rank test over test sessions on the
    per-session difference in the primary metric, at α = 0.05. The line also
    gives the median difference and the win count;
  - "beats" needs p < 0.05 **and** a positive median difference;
  - fewer than 5 sessions can never reach α (p ≥ 2⁻ⁿ), so they never count as
    beating, and the line says why.

  A failure prints CLAUDE.md §5's consequence plainly: "this is not decoding",
  "no signal beyond the target's own autocorrelation", or "a more complex model
  is not justified".
- **Primary metric: AUROC for classification** (threshold-free, robust to
  choice's 26/74 imbalance) and R² for regression. Balanced accuracy is still
  reported, and the NEDS comparison uses it.
- **Per-bin targets are scored on every task-period bin** with a defined target
  and a full window in its span, the same span the shuffle rotates in.
- **The split guard runs first,** on the task's split and the ceiling's, at the
  task's context length.

**Still to come:** the real `DataProvider` (sessions → normalised counts,
windows, targets, null features) and the baselines. Until then the contract is
tested on synthetic sessions with stub decoders.

### 2026-09-28 — Null inputs (`neurodecoder/evaluation/nulls.py`, `configs/nulls.yaml`)

**Decision (the user chose each option below before any code):** the module
builds the inputs for the contract's two null rows; the baseline models fit
them. Nothing was fitted or scored to make these choices.

**`null_shuffle`:** the target is circularly shifted within its session, which
breaks the neural-behaviour alignment and keeps the target's autocorrelation.
- **20 shifts per session**, drawn with an explicit seed, distinct and uniform
  over [minimum, length − minimum].
- **Per-bin targets rotate within the task period** and are undefined (NaN)
  outside it. The minimum shift is **30 s**, about 20× the slowest
  decorrelation measured on three sessions. Autocorrelation falls below 0.1
  after:
  - 0.4–0.5 s for wheel velocity;
  - about 1 s for |velocity|;
  - 1.2–1.3 s for movement state, in two of the sessions. The third never
    dropped below 0.1 within 100 s, which is slow drift a circular shift keeps.
- **Trial-level labels rotate over the target's trials,** and each trial's
  window stays at its own time. The minimum shift is **100 trials**:
  - that is longer than any biased block (21–99 trials, median 47) and the
    90-trial opening block;
  - sessions with fewer than 200 usable trials (about 5%) get no shuffle null,
    reported as missing.

**`null_trialstruct`:** features from task variables only, never spikes.
- **Per-bin targets:**
  - one-hot time since each trial start, stimulus onset and go cue, in 0.1 s
    steps to 3 s, then one "later" feature (95 features in total);
  - the current trial's signed contrast and block prior.

  Behaviour-timed events (first movement, response, feedback) and the choice
  are never read, so beating this null means spikes add more than task timing.
- **Choice:** the current signed contrast and block prior, plus the previous 10
  trials' stimulus side, choice and reward.
- **Block:** the previous 10 trials' side, choice and reward only.
  - The block window ends before stimulus onset, so this null gets neither the
    current stimulus nor the label. The history is what reveals the block (80/20
    stimulus sides).
  - Stimulus side is taken from which contrast column is set, so a 0% trial
    still has a side, which follows the block prior like any other trial.
- **Movement onset** has no null yet; asking for one raises.

**Tests:**
- the features don't depend on spike counts, and don't change when
  behaviour-timed columns are permuted;
- history uses only earlier trials;
- the block null never sees its trial's own block, stimulus, choice or reward;
- shifts keep autocorrelation (lags 1–25) while decorrelating from the original.

**Smoke run on `d23a44ef`:** per-bin features are 183,447 × 95 (70 MB,
0.11 s). Every target gets 20 valid shifts: 115,658 task-period bins for the
per-bin targets, 290 trials for choice and 224 for block (block's valid shifts
are 100–124).

### 2026-09-28 — Decoding metrics (`neurodecoder/evaluation/metrics.py`, `configs/evaluation.yaml`)

**Decision:** metrics are computed **per session**, and summarised only as a
distribution across sessions: median, quartiles, range and the number of
sessions where the metric is defined. Nothing pools samples across sessions.
`test_no_pooled_r2_across_sessions` shows why: a decoder that only knows each
session's mean scores R² > 0.9 pooled, and ≈ 0 in every session.

- **Classification (binary):**
  - balanced accuracy, AUROC, F1, log loss and ECE, computed from
    p = P(class 1);
  - the hard prediction is class 1 when p ≥ 0.5, the plain argmax, so there's
    no threshold to tune.
- **ECE is top-label:** the predicted class's confidence, in 10 equal-width bins
  `[k/10, (k+1)/10)` (the last closed). 10 is netcal's default, which Phase 6
  will use; the bin count is `configs/evaluation.yaml: ece_bins`.
- **Regression:** R² (against the session's own mean), Pearson correlation, MAE
  and RMSE.
- **Undefined is NaN, never an error or a default:**
  - AUROC and balanced accuracy when a session has one class;
  - R² and correlation for a constant target;
  - correlation for a constant prediction;
  - F1 when there are no positives at all.

  Inputs with NaN are refused, so undefined samples (e.g. unlabelled
  `movement_state` bins) must be dropped explicitly first.
- **scikit-learn** (in the Fixed stack and `pyproject.toml`, but never installed
  in the local venv until now) supplies the standard metrics. Tests check each
  against it.

### 2026-09-28 — Targets (`neurodecoder/targets/`, `configs/targets.yaml`)

**Decision (the user chose each option below before any code):** five targets,
all built from the BWM release that the manifest and splits already use. The
release's wheel is within 0.52 mrad of raw, and its `firstMovement_times` is
within 1.7 ms of ONE's 2025-03-03 revision (see "BWM compressed backend").

| Target | Kind | Definition |
|---|---|---|
| `wheel_velocity` | per bin, rad/s | (position at bin end − position at bin start) / bin width; positions linearly interpolated at bin edges, never extrapolated (NaN outside the wheel's samples) |
| `movement_state` | per bin, 1 / 0 / NaN | 1 if the bin lies wholly inside a wheel movement, 0 if wholly inside a quiescent period, NaN otherwise; epochs from IBL's detector as shipped in `bwm_behavior` 2.0.0 (ibllib 4.0.1 `extract_wheel_moves`, quiescence ≥ 0.2 s) |
| `choice` | per trial, 1 / 0 | clockwise (`choice == +1`) / counter-clockwise (−1); window: the 100 ms before first movement |
| `block` | per trial, 1 / 0 | left block (`probabilityLeft == 0.8`) / right (0.2); the unbiased 0.5 block has no label; window: 0.4 s to 0.1 s before stimulus onset |
| `movement_onset` | per trial, event time | `firstMovement_times`, with the bin containing it; no window (how to score an event target is a Phase 3 choice) |

- **Wheel filter: bin displacement.** The alternatives were:
  - IBL/NEDS: interpolate to 1 kHz, 8th-order 20 Hz Butterworth run forwards and
    backwards, differentiate, and sample at the bin end. Each value then mixes
    in tens of ms of future movement, and IBL's filter settings come with it.
  - Causal smoothing over the last N bins: adds a lag and a new tunable.

  Bin displacement is the exact mean velocity over each spike bin, has nothing
  to tune and uses no sample from after the bin. It differs from NEDS's target
  (NEDS decodes |v| after the Butterworth filter), so the Phase 3 NEDS
  comparison must rebuild NEDS's target to compare like with like.
- **Trials: `bwm_include` only.** It is the BWM paper's rule (reaction time
  0.08–2 s, a choice made, no missing key events). It matches that rule on
  99.98% of the release's 295,920 trials (71 differ) and keeps 66.4%. NEDS's
  looser rule (reaction time 0–10 s, trial ≤ 10 s) would keep 83.9%.
- **Windows: the BWM paper's, in `configs/targets.yaml`.** The block window is
  pre-stimulus, so activity carrying the upcoming choice (which correlates with
  block) can't stand in for block. NEDS's window (0.5 s before to 1.5 s after
  stimulus onset, for every target) is left for the Phase 3 comparison, as an
  alternative config.
  - **Exact bin arithmetic:** a window is a whole number of bins (block 15,
    choice 5, at 20 ms). Its `end_bin` is the last bin that finishes by the
    window's stop, so it never reaches past it. It may start up to one bin
    before the window's start.
  - **Handing off to windows:** `end_bin` goes to `preprocess.windows`
    `extract` as `ends`, with `context_bins` from the target.
- **Movement state: moving / quiescent from IBL's epochs.** In all 459
  sessions, movements never overlap each other or a quiescent epoch, and
  quiescent epochs are exactly the gaps of at least 0.2 s between movements
  (plus the stretch before the first). A bin straddling an epoch edge is
  unlabelled, which adds no threshold.
- **Versioning:** `TARGETS_VERSION = 1`. `TargetConfig.fingerprint()` hashes it
  with the windows, and every target records it alongside the preprocessing
  fingerprint of the bins it sits on (R6).

**Checked on `d23a44ef`** (no decoding metric computed):
- **Wheel velocity** is defined in every one of the 183,447 bins. The smallest
  non-zero |v| is 0.003 rad/s, because positions are interpolated at bin edges
  3.1 ms off the wheel's 100 Hz samples.
- **Movement state** over the task period: moving 35.0%, quiescent 63.6%,
  unlabelled 1.4%.
- **IBL's detector and our velocity agree.** They are independent
  constructions from one wheel:
  - mean |v| is 1.079 rad/s in moving bins and 0.018 in quiescent ones (61×);
  - 99.3% of moving bins have non-zero velocity;
  - 65.5% of quiescent bins have exactly zero; the rest hold sub-threshold
    jitter that IBL's detector allows.
- **Trial targets:** 410 trials, 290 `bwm_include`. That gives 290 choice
  labels (26% clockwise), 224 block labels (46% left; the 66 unbiased-block
  trials have none) and 290 onsets.
- **Onsets agree with IBL's movement epochs:** 98.6% lie inside one (allowing
  one bin before its start), a median 2.0 ms from the nearest movement start.

### 2026-09-28 — Held-out configuration split, with within-lab percentiles (`neurodecoder/splits/registry.py`)

**Decision (definition from the user):** a session's configuration is its count
of QC-passing units under the task-period rule. `held_out_config(manifest,
units, preproc)` works from `release_units` without loading sessions:
- **Test:** sessions below the 10th percentile **of their own lab's** sessions.
- **Train:** sessions at or above their lab's 25th percentile.
- **Dropped:** sessions in between.

Animals may be shared between train and test, as for `held_out_region`. The
split records each lab's cutoffs: its percentiles, the same cutoffs as whole
unit counts, and its session count. It also records every session's count and
lab. The guard recomputes every lab's cutoffs from those records, and checks
each session against its own lab's.

**Why within lab (option chosen by the user):** percentiles taken across the
whole release gave 46 test sessions (≤ 46 units) and 344 train (≥ 90). But the
test set was 30.4% hausserlab (13.5% of sessions) and 17.4% wittenlab (8.1%),
so the two labs made up 48% of it. The cause is lab protocol, not yield:
- 95% of hausserlab's sessions use one probe, so it has the lowest session
  counts (median 91.5, against 128–234 elsewhere), yet its units per probe (83)
  are typical;
- 43 of those 46 test sessions were single-probe.

So that split would have measured lab shift. Units per probe still left two
labs at 37% of test. Within-lab percentiles bring every lab's test share to
within 1.3 points of its share of the data.

**On the full manifest** (459 sessions; built in 0.4 s; passes the guard):
- **Sizes:** 50 test, 346 train, 63 dropped.
- **Test animals:** 40. 31 have one test session, 8 have two and 1 has three.
  38 of the 50 test sessions come from animals with other sessions in train.
- **Unit counts:** test median 32 (2–109), train median 165.5.
- **Probes:** 90% of test sessions are single-probe, against 38% of train and
  48% overall. The split still mostly holds out single-probe recordings, but
  now within each lab, which is the configuration shift it's meant to measure.

| Lab | Sessions | Test below (≤ units) | Train from (≥ units) | Test | Train | Test share | Share of all |
|---|---|---|---|---|---|---|---|
| angelakilab | 41 | 61.0 (60) | 90.0 (90) | 4 | 31 | 8.0% | 8.9% |
| churchlandlab | 36 | 71.0 (70) | 112.0 (112) | 4 | 27 | 8.0% | 7.8% |
| churchlandlab_ucla | 41 | 77.0 (76) | 122.0 (122) | 4 | 31 | 8.0% | 8.9% |
| cortexlab | 42 | 88.1 (88) | 97.0 (97) | 5 | 32 | 10.0% | 9.2% |
| danlab | 46 | 63.0 (62) | 90.0 (90) | 5 | 34 | 10.0% | 10.0% |
| hausserlab | 62 | 32.4 (32) | 55.75 (56) | 7 | 46 | 14.0% | 13.5% |
| hoferlab | 17 | 46.6 (46) | 90.0 (90) | 2 | 13 | 4.0% | 3.7% |
| mainenlab | 44 | 65.4 (65) | 122.75 (123) | 5 | 33 | 10.0% | 9.6% |
| mrsicflogellab | 25 | 44.8 (44) | 73.0 (73) | 3 | 20 | 6.0% | 5.4% |
| steinmetzlab | 31 | 17.0 (16) | 70.0 (70) | 3 | 23 | 6.0% | 6.8% |
| wittenlab | 37 | 31.4 (31) | 79.0 (79) | 4 | 28 | 8.0% | 8.1% |
| zadorlab | 37 | 110.2 (110) | 134.0 (134) | 4 | 28 | 8.0% | 8.1% |

**Consequence:** "few units" is relative to the lab. A zadorlab test session
(≤ 110 units) can hold more units than a hausserlab training session (≥ 56).
The split measures low yield for the lab's protocol, not an absolute unit
count.

### 2026-09-28 — Unit QC uses the task-period firing rate (`PREPROC_VERSION` 2)

**Decision (signed off by the user on the evidence below):** unit QC's 0.1 Hz
floor now applies to each unit's firing rate during the task: spikes from the
first trial's `intervals_0` to the last trial's `intervals_1`, both included,
over that time. `min_label` (1.0) and `exclude_regions` (void, root) are
unchanged. The release's `firing_rate` (spike count over the whole recording,
first spike to last) stays in the units table as a reported column only.
`PREPROC_VERSION` goes to 2, so every earlier fingerprint, and any split built
against one, is refused.

**Why:** the normalisation work found units that pass the old floor but are
nearly silent during trials. Recordings run before and after the task (the task
is a median 70% of the recording, 28–111 min), and some units fire almost only
then.

**Evidence** (all 75,395 good units in `bwm_ephys` 1.2.1; measured in 63 s):

| | Whole-recording rule (v1) | Task-period rule (v2) |
|---|---|---|
| Units removed | 83 | 1,402 |
| Pass both / only v1 / only v2 / neither | 73,964 / 1,348 / 29 / 54 | |
| Per-session loss, 50th / 90th / 95th / max (share of good units) | | 1.0% / 4.4% / 6.1% / 34.6% |
| Sessions losing 0 / < 1% / 1–5% / 5–10% / 10–25% / > 25% | | 154 / 231 / 193 / 24 / 10 / 1 |
| QC-passing units per session, 10th / 25th / 50th / 75th / 90th | 48 / 91 / 141 / 217.5 / 304.4 | 46.8 / 89.5 / 138 / 213 / 296.2 |

- A typical unit's two rates agree: the median task/whole ratio is 1.00
  (10th–90th percentile 0.63–1.33).
- **15 sessions lose more than 25% or end with fewer than 20 units.** 14 of
  them already had fewer than 20 under v1, and v2 removes at most one unit from
  each.
- **The outlier is `0cc486c3` (ibl_witten_29, wittenlab), 131 → 87 units.** Its
  46 lost units fire at a median 0.72 Hz over the recording, but only 0.1% of
  their spikes fall inside the task (53–2,870 s of a 0–4,400 s recording).
- **On `d23a44ef`, 397 → 390 units, identical across the BWM, NWB and ONE
  backends.** `probe00_27` (0.0965 Hz whole-recording, 0.0013 Hz in the task)
  was already out. The new rule removes 6 of the 9 near-silent units from
  "Normalisation" plus `probe01_280`. (The chat report said 7 of 9; it is 6.)

**Where the rates come from:**
- **Loaded sessions:** `apply_unit_qc` computes the rates from the session's own
  spikes and trials, and keeps them as a `task_firing_rate` column. No backend
  or session cache changes.
- **Release-wide builders** (`held_out_region`, `held_out_config`) must not load
  sessions, so they read a precomputed table (location chosen by the user):
  - it lives at `<data_root>/derived/bwm_ephys-1.2.1/task_rates-v1.parquet`
    (1.3 MB), with `task_rates-v1.provenance.json` beside it;
  - it's built once by `python -m neurodecoder.cli.build_task_rates` (54 s,
    6 processes, decoding every spike shard);
  - `load_task_rates` refuses a table whose provenance (release name and
    version, `TASK_RATES_VERSION`, unit count) doesn't match;
  - `release_units` joins it onto `units.parquet` and refuses mismatched units.
  After the join, a split build takes about as long as before (0.06 s).
- **Both paths use the same definition.** Both call `qc.units.task_period` and
  `in_task`, and the table uses the BWM backend's shard decoder. On `d23a44ef`
  the table and the session path agree exactly for all 398 units (tested).

**Caches and splits:** nothing on disk depended on preprocessing. The session
cache is keyed on backend, source and loader version, and holds pre-QC
sessions, so it stays valid. No split files had been saved.

**`held_out_region`, rebuilt for all 266 regions:**
- **The four regions with ≥ 20 test sessions are unchanged:** CP 43 / 358 (and
  still 30 parent-label sessions), MRN 33, APN 26, PO 23.
- **13 regions change test size by one or two sessions.** Examples: CA1 14 → 15,
  PRNr 13 → 12, PRM 5 → 3. PRP and SPIV lose their only test session; PC5 gains
  one.
- **Regions with a test set:** 144 → 143. With ≥ 10 test sessions: 16 → 16.
  With ≥ 5: 42 → 41.
- **Training sets:** among the 142 regions with a test set under both rules, 22
  gain one training session and one loses one.

**Tests whose hardcoded counts this rule changes:**
- `test_unit_qc::test_real_bwm_floor…`: 83 → 1,402 removed.
- `test_unit_qc::test_real_three_backends…`: 397 → 390.
- `test_binning::test_real_round_trip…`: 397 → 390.

The CP counts in `test_region_split` don't change. Fixtures changed only to
give QC a `task_firing_rate` or a real task period (`test_unit_qc`,
`test_binning`, `test_region_split`).

**Known and not acted on: within-session drift.** Units that appear or vanish
partway through the task still pass. `probe00_514` on `d23a44ef` fires at 0.011
Hz in the `within_session` train block and 2.07 Hz in the test block (task rate
0.66 Hz), giving z-scores up to 179. 57 of its session's units have a test-block
mean z beyond ±0.5. A per-block rate criterion would tie QC to a split, so drift
is logged here rather than filtered.

### 2026-09-28 — Windows (`neurodecoder/preprocess/windows.py`)

**Decision:** `window_plan(split, context_bins=…, stride_bins=…)` runs
`assert_split_valid` once for that context and returns a plan. For a session
and partition, the plan allows a span of bins:
- **`within_session`:** the partition's block.
- **Other kinds:** the whole binned session, and only if the session is in that
  partition.

`extract(z, binned, partition, ends=None)` returns `(n_windows, n_units,
context_bins)` float32 windows. It refuses any end whose window would leave the
span. `unwindow` puts windows back on the session's bins.

- **A window ending at bin e covers bins e − context + 1 … e.** Default ends
  are every `stride_bins`-th bin from the first full window. Where targets sit
  relative to a window, and which ends have a target, is left to `targets/`.
  `extract` takes ends chosen elsewhere and still enforces the span.
- **Context and stride are required,** with no defaults. Their values belong
  in the run configs that come with models; the context also feeds the guard.
- **Windows come from a strided view,** and indexing copies only the chosen
  windows. **Batching is the caller's job:** on `d23a44ef`'s train block (78,885
  bins, 397 units) windows take 0.13 GB at stride 50, 1.25 GB at stride 5, and
  would take about 6 GB at stride 1. The data loader should pass `ends` in
  batches.
- **For cross-session splits the span is the whole binned session,** including
  time before the first trial and after the last. Restricting to the task
  period depends on the target, so `targets/` does it by choosing ends.

**Verified on `d23a44ef`:**
- The Phase 2 criterion "round-trip a session to tensors and back to spike
  counts within float tolerance" passes: counts → normalised → windows
  (context 50, stride 50) → `unwindow` → inverse → counts matches exactly on
  every bin a window covers.
- Train and test windows share no bin and are at least a context apart (the
  roadmap's `test_no_temporal_leakage`, now at the window level).
- Train gives 1,577 windows and test 731, extracted in ≤ 0.1 s.

### 2026-09-28 — Normalisation (`neurodecoder/preprocess/normalize.py`)

**Decision (option chosen by the user):** per-unit statistics where the units
have training data, pooled statistics otherwise. `fit_normalizer(split, binned,
preproc)` takes a split, never a bare session, and reads only training data: the
train block of each `within_session` session, or the sessions of the train
partition. Calibration and test data are never read.

The mode is set **per split kind**:
- **`within_session` → `per_unit`:** each unit's mean and std over its train
  block.
- **Every cross-session kind → `pooled`:** one mean and std over all units and
  bins of the train partition, applied to every unit, train and test alike.

The per-split-kind mode is how the user's chosen option reads ("the
within-session ceiling is normalised more finely than cross-session rows").
Its reason: normalising training units per unit and unseen test units pooled
would itself shift the inputs between train and test.

- **std floor = √(min_firing_rate_hz · bin_s)**, the std of a Poisson unit at
  the QC minimum rate (0.045 at 0.1 Hz, 20 ms). It's derived from existing
  config, so there's no new tunable. Without it a unit silent in training would
  divide by zero.
- **Sums are exact int64 sums,** so the statistics don't depend on session
  order. The `Normalizer` is a hashed dataclass that serialises to JSON for the
  model artifact (R3). `from_dict` refuses edited content.
- **`transform` returns `(n_units, n_bins)` float32** (291 MB for
  `d23a44ef`). It refuses binned data with another fingerprint, and in
  `per_unit` mode a session or unit set it wasn't fit on. `inverse_transform`
  undoes it.

**Verified on `d23a44ef`** (`within_session`, 0.8, 2 s):
- Counts round-trip through `transform` and `inverse_transform` exactly after
  rounding. This is half of Phase 2's success criterion "round-trip a session
  to tensors and back"; the windows module is the other half.
- Fit takes 0.04 s and transform 0.22 s.

**Finding, not acted on: within-session non-stationarity.**
- **9 of 397 units hit the std floor.** They are nearly silent in the train
  block (≤ 0.07 Hz) but pass QC's 0.1 Hz. QC uses the release's whole-recording
  firing rate, which includes the post-task period.
- **Some fire mostly after the task.** `probe01_953` spikes only from 2,637 s,
  and the trials end at 2,330 s. `probe01_1157` averages 9 Hz overall but
  0.01–0.18 Hz during trials.
- **Some appear partway through,** probably from drift. `probe00_514` is at
  0.011 Hz in train and 2.07 Hz in test, giving z-scores up to 179.
- **57 units have a test-block mean z beyond ±0.5.**

A task-period firing-rate criterion would change the signed-off QC, so it's
left for the user.

### 2026-09-28 — Held-out region split, and `iblatlas` as a dependency (`neurodecoder/splits/registry.py`)

**Decision:** `held_out_region(manifest, units, preproc, region=R)` follows the
split doc's stricter definition, with one tightening. `units` is the BWM
release's `metadata/units.parquet`, the same table the BWM backend loads
sessions from. Only units that pass `preproc.qc` count.
- **Test:** at least 20% of the session's QC-passing units have Beryl region R.
  The denominator includes units with no Beryl region (fibre tracts, ventricles,
  parent-only labels).
- **Train:** no QC-passing unit in R, **and none that could be in R.** A unit
  could be in R when it has no Beryl region and its Allen label contains R or
  lies inside it: `STR` for CP, `TH` for PO, `MB` for MRN.
- **Everything else is dropped.** The split still records every session's
  counts (`n_units`, `n_in_region`, `n_possibly_in_region`), so the file shows
  why a session is out. The guard re-checks both rules from those counts.
- **R must be a Beryl region.** The split records R, the 20% threshold and the
  `iblatlas` version.
- **No calibration partition,** as for `held_out_session`.

**Why the tightening:** 10,059 of the 75,395 release units (13%) have no Beryl
region. Most are fibre tracts, but many carry only a parent label: `MB` 1,465,
`P` 751, `STR` 526, `TH` 310. Under the literal "zero units in R" rule such a
session trains while possibly holding units from R. Excluding them costs
training sessions:
- 30 for CP (388 → 358);
- 50 for MRN (349 → 299);
- 107 for APN (408 → 301);
- 42 for PO (409 → 367);
- none for cortical regions such as MOp, MOs and VISp.

**Why `iblatlas`:** the containment test needs the Allen hierarchy. `iblatlas`
is IBL's atlas package, a dependency of `ibllib` (already in the Fixed stack),
though `ibllib` itself isn't installed here. 1.2.1 is the version already used
in NEDS's environment. It adds only `pynrrd` beyond the Fixed stack, plus
matplotlib (Fixed, never installed until now). Its Beryl mapping agrees with
the release's `beryl_acronym` for all 75,395 units.

**Verified on the full manifest** (every Beryl region, 0.4 s each, all pass the
guard): 144 of 266 regions have a test set.
- **≥ 20 test sessions: 4 regions.** CP 43 (358 train), MRN 33, APN 26, PO 23.
- **≥ 10: 16 regions.**
- **≥ 5: 42 regions.**

**Test animals are not kept out of training.** For CP, 36 of the 43 test
sessions come from animals with other sessions in train. The split doc doesn't
ask for animal disjointness. Without it, a held-out-region result measures
region transfer to animals the model has already seen, so it can't be read as
cross-animal. Requiring disjointness would shrink test sets further.

**Signed off by the user (2026-09-28):** both the parent-label tightening and
leaving animals shared between train and test, as described above.

### 2026-09-28 — Split registry and guards (`neurodecoder/splits/`)

**Decision:** four of the six split kinds in `docs/SPLITS_AND_LEAKAGE.md` are
built by `splits/registry.py`, checked by `splits/guards.assert_split_valid`,
and saved as JSON with a hash.

| Kind | Builder | Partitions |
|---|---|---|
| `held_out_animal` | `held_out_groups(by="subject", n_test, n_calibration, seed)` | whole animals; calibration is its own set of animals |
| `held_out_lab` | `held_out_groups(by="lab", …)` | whole labs, the same way |
| `held_out_session` | `held_out_session(seed)` | one session of each animal with ≥ 2 sessions is test; everything else is train |
| `within_session` | `within_session(trials, train_fraction, gap_s)` | per session, one early train block and one late test block |

- **Seeds are required keyword arguments** with no default (R7). Groups are
  sorted and then shuffled with `numpy.random.default_rng(seed)`. The saved
  file, not the seed, is the record: another numpy version could draw
  differently from the same seed.
- **Group splits cover the whole manifest** (all 459 sessions). Filtering for
  a task, such as sessions with pose, happens downstream and must not re-split.
- **`within_session` blocks are bounded by trials.** The train block runs from
  the first trial's start to the end of the last train trial
  (`round(train_fraction · n_trials)` trials). The test block starts at the
  first later trial whose first bin is at least `ceil(gap_s · rate)` empty bins
  after the train block. Trials in between belong to neither partition, and so
  does time before the first trial or after the last. `gap_s ≥ 2 s` is
  enforced when the split is built.
- **No calibration partition yet for `held_out_session` or `within_session`.**
  Phase 7's cross-animal conformal needs one only for animal splits. It can be
  added when a within-session calibration result is actually needed.
- **The hash** is the sha256 of canonical JSON (sorted keys, no NaN) of the
  whole split: kind, params, partitions, per-session records, the manifest
  provenance and the preprocessing `{fingerprint, bin_ms}`. The fingerprint
  covers `PREPROC_VERSION`. `load_split` refuses a file whose content no longer
  matches its hash. **`save_split` never replaces an existing file with a
  different split**, so a split used by a result can't change under it.
- **The guard raises and never warns.** It checks:
  - the split's `manifest_version`, release versions and preprocessing
    fingerprint equal the current ones (by default, this code's and
    `configs/*.yaml`'s);
  - partitions are known, non-empty and include train and test;
  - for group splits, no subject or lab is shared by any pair of partitions,
    calibration included;
  - for non-temporal splits, no session is shared;
  - for `within_session`, blocks share no bin and have at least
    `max(context_bins, 2 s)` empty bins between them;
  - each listed trial is in exactly one partition and lies wholly inside that
    partition's block.

  Each check has a test that feeds it a broken split.
- **The split file stores every trial's interval,** so the guard can check
  trial integrity without loading sessions. That makes a `within_session`
  file over all 459 sessions 12.6 MB. A `held_out_animal` file is 0.07 MB.

**Verified on the full manifest (459 sessions, 139 animals, 12 labs):**
- **`held_out_animal`** (20 test, 15 calibration, seed 0): 354 / 52 / 53
  sessions. The roadmap asks for ≥ 20 test sessions.
- **`held_out_lab`** (2 test, 1 calibration): 343 / 44 / 72 sessions.
- **`held_out_session`**: 352 train, 107 test.
- **`within_session`** on every session (0.8, 2 s): at most 1 trial per
  session falls in the gap. Actual gaps range from 3.4 s to 66.3 s (median
  4.8 s), because the test block starts at a trial start.
- Every split passes the guard at a 50-bin context. Every split builds in under
  a second, from a manifest that builds in 0.7 s.

**Not built yet:**
- **`held_out_region`** needs per-session counts of QC-passing units per
  region. The rule is ≥ 20% of units in R for test and zero units in R for
  train. The manifest's `regions` come from the release's good units, not from
  our QC.
- **`held_out_config`** needs a definition of a "standard" probe configuration.

The roadmap's Phase 2 success criterion ("all five split types generate and
validate") isn't met until both exist. The split doc's table lists six kinds,
so the roadmap's "five" is out of date by one.

### 2026-09-28 — Binning and `PREPROC_VERSION` (`neurodecoder/preprocess/binning.py`)

**Decision:** `preprocess_session(session, config)` applies unit QC and then
bins spikes into `(n_units, n_bins)` `uint16` counts. `bin_spikes` bins
without QC, and its result carries no fingerprint.

- **The grid is anchored at t = 0 on the session clock:** bin *k* covers
  `[k·w, (k+1)·w)`, and `first_bin = floor(t_start · rate)`. Backends whose
  time bounds differ slightly still line up bin by bin (BWM and NWB differ by
  one bin at the end on `d23a44ef`). A spike exactly on an edge goes to the
  upper bin.
- **The bin width is an integer number of ms that divides 1000**
  (`configs/preprocess.yaml`, default **20 ms**, matching NEDS; 10 and 50
  also valid). A spike's bin is `floor(t · rate)` with an integer rate, which
  avoids `floor(t / w)` misplacing spikes through float error
  (`0.06 / 0.02 = 2.999…`).
- **Counts are `uint16`,** and a bin over 65,535 raises instead of wrapping.
  The real maximum on `d23a44ef` is 19 per 20 ms bin.
- **`PREPROC_VERSION = 1`** is a code version, bumped by hand whenever
  binning (or anything it calls) gives different output for the same input.
  **`PreprocConfig.fingerprint()`** is the sha256 of `PREPROC_VERSION`, the
  bin width and the unit-QC hash. It's what caches and the split registry
  will record (R6): changing any threshold, the bin width or the code version
  changes it.

**Verified on `d23a44ef` (20 ms):**
- `(397, 183,448)` counts, 146 MB, QC plus binning in 0.43 s.
- Each unit's counts sum to its spike count.
- **ONE and NWB bin byte-identically** (same spike times).
- **BWM differs from NWB in 0.248% of spikes** (64,293 of 25,887,688), each
  moved by one bin. BWM stores 100 µs ticks, so only spikes within 50 µs of
  an edge can move; the expected fraction for evenly spread rounding is
  50 µs / 20 ms = 0.25%. The test enforces the 0.5% worst case.
- The same input gives byte-identical counts and the same fingerprint (the
  roadmap's `test_binning_determinism`).

### 2026-09-28 — Phase 2 order, and unit QC (`neurodecoder/qc/units.py`, `configs/qc.yaml`)

**Signed off by the user before any Phase 2 code (CLAUDE.md §10):**
- **Module order:** `qc/units` → `preprocess/binning` (defines
  `PREPROC_VERSION`) → `splits/registry` + `guards` → `preprocess/normalize`
  → `preprocess/windows` → `targets/`. Binning comes before splits because
  the split file's hash covers `PREPROC_VERSION`, and splits come before
  normalisation because `normalize.py` takes a split object
  (`docs/SPLITS_AND_LEAKAGE.md`).
- **Unit QC**, with the thresholds fixed now, before any metric exists, so
  they can't be tuned against results (R6):

| Criterion | Value | Evidence at sign-off (BWM release) |
|---|---|---|
| IBL QC label | `== 1.0` | label = fraction of 3 metrics passed; **100%** of label-1 units pass IBL's sliding-RP test |
| Location | not `void`/`root` | 307 label-1 units are located there (see "Session manifest") |
| Firing rate | `≥ 0.1 Hz` | removes 83 of 75,395 good units (0.11%); 1 Hz would remove 8.3%, NEDS's 5 Hz 45.8% |
| Separate RPV ceiling | none | redundant with label 1 for IBL data; revisit for non-IBL NWB (Phase 11), which has no IBL label |

0.1 Hz was chosen to stay closest to BWM's published good-unit set, which
Phase 3's baselines will be compared against.

**Implementation:**
- `unit_qc(units, qc)` returns `passed` plus a `reason` naming **every**
  failed criterion. A missing label, firing rate or location value **fails**
  the unit instead of passing it (§7).
- `apply_unit_qc(session, qc)` keeps the passing units and their spikes, and
  raises if none pass (§10).
- **Location comes from whichever field the backend provides:** BWM's
  `acronym`, ONE's Allen `atlas_id` (0 = void, 997 = root) or NWB's full
  names in `location`. With none of them, QC raises rather than silently
  skipping the in-brain criterion.
- `UnitQC.hash()` changes whenever any threshold does. `PREPROC_VERSION`
  (next module) will incorporate it.

**Verified on `d23a44ef`:**
- After QC, **all three backends keep exactly the same 397 units**: BWM's 398
  good units minus `probe00_27` (0.0965 Hz).
- ONE's `atlas_id == 997` and NWB's `location == "root"` pick out the same
  units.
- Across the whole BWM release, QC removes exactly the 83 sub-0.1 Hz units.

**Known cross-backend residual:** BWM's good-unit table also drops 6 label-1,
in-brain units for an unrecorded reason (see "Session manifest"). QC on
ONE/NWB would keep those 6, so the unit sets can differ by them on the
sessions where they occur. `d23a44ef` has none.

### 2026-09-28 — BWM backend loads behaviour from `bwm_behavior` 2.0.0

**Decision:** `load_session_bwm(eid, root, behaviour_root=None)` decodes the
per-session `sessions/<eid>.zip` shard of the `bwm_behavior` 2.0.0 release
(version pinned through its `manifest.json`), following ibl-ai-agent's own
decoder. Unknown encoding kinds raise. `load_session("bwm")` passes the release
from config. `LOADER_VERSION` is now 2, and the cache source gains
`behaviour_version`, so every previously cached BWM session (including the 50
from the Phase 1 check) becomes a miss; those entries are orphaned on disk.

| Key | From the shard |
|---|---|
| `wheel` | `wheel.position` only, **not** the stored velocity, which bakes in IBL's smoothing filter (R6) |
| `pose_{left,right,body}` | each keypoint's `_x`, `_y`, `_likelihood` columns, under IBL's ALF names |
| `motion_energy_{left,right}` / `motion_energy_body` | `whiskerMotionEnergy` / `bodyMotionEnergy` |
| `pupil_left` | `pupilDiameter_raw`, **not** `pupilDiameter_smooth` (R6) |
| `pupil_right` | declared missing: the right camera has no pupil diameter in this release |
| `lick` | declared missing: not in `bwm_behavior` |

A source the build listed in a camera's `skipped_sources`, or a camera or
shard that's absent, is declared missing with that reason.

**What the release actually stores, measured against NWB on `d23a44ef`
(ibl-ai-agent's docs understate it):**
- **The wheel is resampled, not native.** It's the raw encoder position
  linearly interpolated onto an exact 100 Hz grid (366,885 samples versus
  NWB's 755,552 raw ones; only 15.7% of its times coincide with a raw sample)
  and rounded to 0.001 rad. It's within **0.52 mrad** of NWB's raw wheel
  interpolated at the same times (one encoder tick is 1.53 mrad).
  **Phase 2 consequence:** a wheel-velocity target differs slightly by
  backend. Build it from one declared source.
- **Camera times are an ideal grid, not the frame times.** Each camera keeps
  the real frame nearest each point of a uniform 60 Hz (left, right) or 30 Hz
  (body) grid, and stores `start + i/rate` as its time. The right camera is
  downsampled from ~150 Hz. Each grid time is within half a frame of its real
  frame's time (max 8.3 / 3.3 / 16.6 ms for left / right / body). If a dropped
  frame ever made two grid points pick the same frame, deduplication would
  shift every later time by a whole frame. It doesn't happen in this session
  (frame counts equal grid sizes), but the loader can't detect it from the
  shard alone.
- **Quantisation:** pose x/y to 0.5 px, likelihood to 8 bits, motion energy
  and pupil diameter to 0.05.
- **Right and body pose match NWB within quantisation** (x/y ≤ 0.25 px,
  likelihood ≤ 0.002) at the matching frames: same tracker output, correct
  decoding.
- **Left-camera pose does not match NWB, unexplained.** At the correct frames,
  x/y differ systematically: paw median ~11 px (max 447 px), pupil ~0.3–1.2 px
  (max 3.7 px). Likelihoods are mostly identical. It isn't a frame offset (the
  error is smallest at zero shift) and it isn't mirrored names (swapping
  left/right makes it much worse). The likeliest cause is different
  post-processing of left-camera x/y in one source, but it isn't recorded in
  either dataset. Don't mix left-camera pose across backends.
- **Pose NaNs differ by design.** NWB's converter blanks x/y where tracker
  likelihood is below **0.9** (NaN ⇔ likelihood < 0.9, exactly: max 0.898
  where NaN, min 0.902 where finite). BWM keeps the tracker's raw
  low-confidence estimates, as its docs say. Neither loader applies a
  threshold. To compare or combine backends, apply the same likelihood mask
  to both; choosing it is a Phase 2 preprocessing decision.
- **Keypoint names differ.** BWM uses IBL's ALF names (`paw_l`, `paw_r`,
  `pupil_{top,bottom,left,right}_r`, `tongue_end_l/r`, `nose_tip`,
  `tube_top`, `tube_bottom`, `tail_start`), while NWB's converter uses
  `left_paw`, `right_paw`, `right_pupil_*`, `left/right_tongue_end`, and the
  same names for the rest. BWM also has more left-camera keypoints (11 versus
  NWB's 6). Aligning NWB to the ALF names is a separate NWB backend change.

**Verified:** 11 tests.
- 6 run in CI on a synthetic shard in the release's exact format: every
  encoding kind, rejection of unknown kinds, canonical keys, position-only
  wheel, raw pupil, and skipped sources declared missing.
- 5 run on the real release against NWB: the wheel residual ≤ half a
  quantisation step; right and body pose within quantisation, with frame
  times within half a frame; left-camera frame timing within half a frame;
  and the capability report.

### 2026-09-28 — ONE backend (`neurodecoder/data/backends/one_backend.py`), and three-way agreement

**Decision:** `load_session_one(eid, one)` reads a session from IBL's public
Alyx through ONE, and `load_session(eid, "one")` registers it. **Revisions are
pinned** and both are part of the cache key:

| Pin | Value | Why |
|---|---|---|
| `SORTER_REVISION` | `2024-05-06` | the sorting NWB and the compressed BWM both use |
| `TRIALS_REVISION` | `2025-03-03` | the only trials revision on Alyx for these sessions, and the one NWB matches |

This is the direct answer to the Phase 0 finding that **ONE serves the newest
revision by default**, which is how NEDS's motion-energy inputs came from files
postdating its paper (`docs/PRIOR_ART.md` §C). Every `load_object` call here
passes an explicit revision; changing a pin is a deliberate edit that
invalidates cached sessions.

- **No new dependency.** It uses ONE-api only (already in the Fixed stack),
  not `ibllib`/`brainbox`. Phase 0 showed `ibllib`'s API drifts, and the
  high-level loaders aren't needed for this.
- **All units, not just good ones.** `units.label` carries IBL's QC label
  (1.0 = good), and callers filter. Extra columns: `cluster_id`,
  `cluster_uuid` (matching NWB's), `atlas_id` and `peak_channel`.
- **`units.x/y/z` are metres relative to bregma**, from each cluster's peak
  channel in `channels.mlapdv` (µm), the same convention as BWM. So ONE and
  BWM share a coordinate frame, while NWB's are Allen CCF µm.
- **`units.acronym` is declared missing:** ONE gives Allen CCF ids (kept as
  `atlas_id`), and mapping ids to acronyms needs `iblatlas`.
- **`spikes.clusters` indexes the cluster table, it is not `cluster_id`.**
  Getting that wrong would silently mis-assign every spike, so the backend
  range-checks the index and raises.
- **Behaviour:** the raw wheel only. A session with no wheel is declared
  missing rather than raising. Pose, pupil and motion energy aren't loaded
  yet.
- **`make_one(cache_dir)`** builds the client, caching downloads under
  `one_cache` from `configs/data.yaml` (`~/data/neurodecoder/one`).

**Three-way agreement on `d23a44ef`, now tested in code:**
- **ONE vs NWB: exact.** All 1,961 units, all 61,981,600 spike times
  element-for-element, all 13 trial fields, every `cluster_uuid` and `label`,
  the `depths`, and the 755,552-sample wheel. Both derive from the same
  sorting run, so exact equality is the right bar, and the NWB conversion
  preserves the data.
- **BWM vs NWB:** same good units and spike counts, spike times within 50 µs
  (BWM stores 100 µs ticks), all trial fields identical except
  `firstMovement_times` (see "BWM compressed backend").

This satisfies ROADMAP Phase 1's "three backends return byte-identical unit
counts and spike-count totals for the same eid", with the two documented,
explained discrepancies rather than a silent averaging over them.

**Cost:** about 60 s for this session from a warm ONE cache; the session's
files are about 1.6 GB, dominated by `spikes.times`/`spikes.clusters` for
both probes. The cache for the real tests was populated by copying the files
the Phase 0 NEDS run had already downloaded, rather than fetching them again.
Those tests skip when that cache is absent.

### 2026-09-28 — Phase 1 success check (`neurodecoder/cli/phase1_check.py`): passed

**Decision:** ROADMAP Phase 1's success criterion ("`load_session` works for
50 sessions across ≥10 subjects and ≥3 labs, from cache, in under 5 s each")
is checked by `python -m neurodecoder.cli.phase1_check`:
1. It builds the manifest and picks 50 sessions deterministically, with no
   randomness (R7): each subject's earliest session, taking labs in turn.
2. It loads each through `load_session(eid, "bwm")`, which fills the cache,
   then times a second, cached read.
3. It checks each session's unit and trial counts against the manifest.
4. It writes `runs/<UTC time>_phase1_check/report.json` with the git SHA and
   a dirty flag (§7), and exits non-zero on any failure.

The manifest needs the `bwm_behavior` folder, so `configs/data.yaml` gained
a `bwm_behavior` key, with `DataConfig.bwm_behavior_root` in `load.py`.

**Result, run 2026-09-28 (`runs/20260928T071223Z_phase1_check/`): PASSED.**
- 50 sessions from **50 subjects across all 12 labs**.
- **Cached reads: median 0.06 s, p95 0.25 s, max 0.37 s**, against the 5 s
  budget.
- First loads from source, including the cache write: median 0.50 s, max
  5.76 s (the largest session, 421 units).
- **All 50 sessions' unit and trial counts match the manifest.** Sessions
  range from 12 to 456 good units (median 145) and 402 to 1,440 trials.
- Cached read time rises with session size (correlation 0.81 with unit
  count), as real I/O should.
- Whole run 48 s; cache 3.1 GB for the 50 sessions.

**Caveats:**
- The cached reads came right after the writes, so the OS file cache was
  likely warm; clearing it needs admin rights. Cold reads will be slower,
  but the slowest warm read has a 13× margin.
- This run recorded `git_dirty: true`, because the check module wasn't
  committed yet (base commit `2ef7e03`). Rerun after committing for a report
  tied to committed code.

**What Phase 1 still lacks, even though this criterion is met:** the ONE
backend, behaviour in the BWM backend, and a three-way backend agreement
test. BWM and NWB agreement is shown on `d23a44ef`.

### 2026-09-28 — Session manifest (`neurodecoder/data/manifest.py`), and what BWM's "good units" are

**Decision:** `build_manifest(ephys_root, behaviour_root)` builds two tables
from the metadata of the extracted `bwm_ephys` 1.2.1 and `bwm_behavior` 2.0.0
releases, without loading any session. Both release versions are checked
through their `manifest.json`. `write_manifest` / `read_manifest` store them
as `sessions.parquet`, `insertions.parquet` and `provenance.json`, replacing
the old copy atomically.

- **`sessions`, one row per eid:** subject, lab, date, session number,
  `n_probes`, `n_units` (all clusters), `n_label1_units`, `n_good_units`,
  `n_trials`, `n_included_trials` (`bwm_include`), `regions` (sorted Beryl
  acronyms of the good units) and `modalities`.
- **`modalities` uses canonical names** (`wheel`, `pose_left/right/body`)
  and says what the **`bwm_behavior` release holds**, not what
  `load_session("bwm")` loads: that backend doesn't read behaviour yet.
  Motion energy and pupil aren't in `bwm_behavior` at all.
- **`insertions`, one row per probe:** unit and channel counts, plus the
  probe's **tip** and **top** positions in **meters**, bregma-relative (the
  same frame as the units' `x/y/z`). The tip is the mean position of the
  channels nearest the probe tip (smallest `localCoordinates_y`), and the top
  is the mean of the channels farthest from it. The release's channel table
  has `mlapdv_x/y/z` (µm) and `localCoordinates_x/y`, not the `x/y/z` /
  `axial_um` its docs list.
- **Every count is computed from the underlying tables** and must equal the
  release's own per-session values (`n_insertions`, `n_good_units`,
  `n_trials`, `n_included_trials`), or building raises.
- **No config keys.** The two release folders are arguments, and nothing
  needs to find the manifest through config yet. Keys get added when a
  command-line tool does.

**Finding: BWM's "good units" are not simply `label == 1`.** 75,708 clusters
have `label == 1`, but the release's good-unit table holds 75,395, a strict
subset (none outside it). The 313 dropped label-1 units, spread over 89 of
699 probes:
- **307 are located in `void` (187) or `root` (120)**, i.e. outside the brain
  or without a region. That matches ibl-ai-agent's `INVALID_ACRONYMS` /
  `INVALID_BERYL_ACRONYMS = {"void", "root"}` filter.
- **6 are unexplained:** 3 in SCiw, 2 in CUL4 5, 1 in CENT2. Their Beryl
  mappings are valid (SCm, CUL4 5, CENT2, checked with `iblatlas`), other
  units in the same regions on the same probes were kept, and their label,
  `bitwise_fail`, spike counts, firing rates and presence ratios are normal.
  The reason isn't recorded anywhere I found.

The manifest therefore checks that good units ⊆ `label == 1` (and raises
otherwise), rather than equality. It reports `n_label1_units` next to
`n_good_units` so the gap stays visible. **Phase 2 consequence:** ROADMAP's
planned `qc/units.py` default ("IBL's own QC label, plus firing rate floor,
plus RP ceiling") would keep units that aren't in the brain unless it also
excludes `void`/`root`. Decide that explicitly there. For session `d23a44ef`
the two sets happen to coincide, which is why the earlier NWB-vs-BWM check
matched exactly.

**Verified on the real releases** (building takes about 1 s):
- Totals equal every count verified earlier: 459 sessions, 139 mice, 12
  labs, 699 probes, 75,395 good units, 621,733 clusters, 295,920 trials.
- `d23a44ef`: 2 probes, 1,961 clusters, 398 good units, 410 trials.
- 15 sessions have no pose.
- The tip is deeper than the top on all 699 probes. The median tip-to-top
  span is 3.78 mm, matching Neuropixels 1.0's 3.84 mm recording length.
- 99.98% of good units lie between their probe's tip and top (±10 µm).

### 2026-09-28 — `load_session` entry point and `configs/data.yaml`

**Decision:** `load_session(eid, backend="bwm", *, config=None,
use_cache=True)` in `neurodecoder/data/load.py` is the single way to get a
session, as the ROADMAP Phase 1 interface describes. Each backend is
registered with its data **source** identity and **loader version**, which
together with the eid form the cache key.

| Name | Reads | `source` in the cache key |
|---|---|---|
| `"bwm"` (default) | `bwm_compressed.load_session_bwm` | `{"dataset": "bwm_ephys", "version": "1.2.1"}` |
| `"nwb"` | `dandi_nwb.load_session_nwb`, from a local copy if one exists, else streamed | `{"dandiset": "000409", "version": "0.260309.1324"}` |

- **Paths come from `configs/data.yaml`** (the first file in `configs/`, per
  §7): a `data_root` plus subpaths for the BWM release, the local 000409
  copies and the cache. `NEURODECODER_DATA_ROOT` overrides `data_root`.
  Unknown or missing keys raise, so a typo can't be silently ignored.
- **`LOADER_VERSION = 1`** now sits in each backend module, next to its
  pinned data version. Bump it whenever that backend's mapping into a
  `Session` changes. It's part of the cache key, so the bump invalidates
  that backend's cached sessions, as R6 does for `PREPROC_VERSION`. It lives
  in the backend, not in the registry, so the person changing a mapping sees
  it.
- **Local and streamed NWB share one cache key.** A local copy is the pinned
  version's asset (downloads are SHA-256 checked, and the streamed load is
  tested equal to the local one). Two local copies for one eid raise instead
  of one being picked.
- **`"one"` was added later** (see "ONE backend"), with its pinned sorting and
  trials revisions in the cache key. An unregistered name raises, listing the
  available backends.
- `use_cache=False` bypasses the cache completely: no read, no write.

**Verified:**
- `load_session` returns sessions identical to the direct backend calls for
  BWM and NWB on `d23a44ef`, and the second BWM call is served from the cache
  without calling the backend.
- Config parsing, the env override and key rejection are tested.
- The NWB file resolution is tested: local copy first, streaming otherwise,
  and ambiguous local copies rejected.

### 2026-09-28 — Session cache (`neurodecoder/data/cache.py`)

**Decision:** `SessionCache(root)` stores loaded `Session`s on disk, one
directory per key. `get_or_load(key_parts, loader)` returns a cached session,
or runs the loader and stores its result.

- **Keys are content addresses.** The key is the sha256 of the key parts plus
  `CACHE_FORMAT_VERSION`, independent of dict order. The required parts are
  `eid`, `backend`, `source` (the pinned data version, e.g. BWM `1.2.1` or
  DANDI `000409@0.260309.1324`) and `loader_version`. Changing any of them
  makes a new entry, so a stale session can't be served after a data or
  loader change. Entries live at `root/<first 2 hex chars>/<sha256>`.
- **Where ROADMAP's `(eid, PREPROC_VERSION, config_hash)` fits:** Phase 1
  caches raw sessions, with no preprocessing, so there's no
  `PREPROC_VERSION` yet. When Phase 2 caches binned tensors it adds
  `PREPROC_VERSION` and the config hash as further key parts. The key scheme
  already accepts extra parts.
- **Writes are atomic.** An entry is written to a `.tmp-*` directory next to
  its final location and renamed into place, and a failed write deletes the
  temporary directory. An entry exists only if its `meta.json` does, so a
  crash can never leave a half-written entry that later reads as valid.
- **No pickle.** Spikes are one flat `float64` `.npy` plus per-unit offsets.
  Behaviour series are `.npy` files, loaded with `allow_pickle=False`.
  Units and trials are parquet, and `meta.json` holds the eid, time bounds,
  unit and behaviour order, channel names, capabilities and the key parts.
  Pickle is fast, but it isn't safe to load and is fragile across versions.
- **Loaded entries are re-validated,** because every `Session` checks itself
  when built.
- **`loader_version` comes from the caller.** The cache doesn't import any
  backend; the planned `load_session(eid, backend=…)` entry point will
  supply each backend's loader version. Bump it whenever a backend's mapping
  changes, just as R6 bumps `PREPROC_VERSION`.

**Verified:** 12 tests cover:
- round trips that match in every field and dtype, on a fixture with text
  columns, float32, booleans, NaNs, a named multi-channel series and a unit
  with no spikes;
- the real BWM and NWB `d23a44ef` sessions round-tripping exactly;
- key determinism and required parts;
- a mid-write failure leaving no entry and no temporary directory;
- an entry without `meta.json` counting as a miss;
- a loader returning the wrong eid being rejected.

**Measured on `d23a44ef`:**

| Backend | From source | Cache write | **Cache read** | Entry size |
|---|---|---|---|---|
| BWM compressed | 3.4 s | 0.2 s | **0.2 s** | 198 MB |
| DANDI NWB (local) | 5.9 s | 0.6 s | **0.9 s** | 687 MB |

Both are well under ROADMAP Phase 1's 5 s target. A streamed NWB session
(~20 min) drops to 0.9 s after the first load. Disk cost is roughly 10 GB
per 50 BWM sessions or 34 GB per 50 NWB sessions: NWB includes every cluster
and the video signals. Nothing is compressed, because read speed is the
point. Compression is the lever if disk becomes the constraint.

**Alternatives considered:** pickle (unsafe to load, fragile); HDF5 (would
work, but parquet keeps pandas dtypes and index names without custom code);
hashing the backend's source code into the key (automatic, but any comment
edit would invalidate everything, including 20-minute streamed sessions).

### 2026-09-28 — BWM compressed backend (`neurodecoder/data/backends/bwm_compressed.py`), and `numcodecs`

**Decision:** `load_session_bwm(eid, root)` reads one session from an
extracted `bwm_ephys` release into a `Session`. It adds **`numcodecs`**
(0.16.5; requires only `numpy` and `typing_extensions`) as a dependency,
because the spike shards are `numcodecs` Blosc (zstd, shuffle) arrays,
exactly as ibl-ai-agent writes them.

- **The release version is pinned to `1.2.1`.** `manifest.json` must say
  `bwm_ephys` `1.2.1`, or the backend raises, for the same reason the DANDI
  and ONE revisions are pinned.
- **Spike decoding follows ibl-ai-agent's own reader:** times =
  (`cumsum(delta ticks)` + `time_origin_ticks`) × 100 µs. Shards whose
  `format`, `time_encoding` or `cluster_encoding` differ from the known
  values are rejected. Each shard's per-unit counts must equal its per-spike
  cluster assignments, and they must match the units table's cluster IDs
  and `spike_count`, or the backend raises.
- **Spike-time error:** the encoder rounds (`np.rint`) to 100 µs ticks. The
  shards in this release record origin 0, from an older encoder, so there's
  a single rounding and the error is at most **50 µs**. ibl-ai-agent's
  current encoder rounds the first spike separately, which would allow up
  to 100 µs; that doesn't apply to these shards. Measured on `d23a44ef`
  against NWB: max 50.00 µs, over all 398 units.
- **Unit IDs are `{probe_name}_{cluster_id}`**, the same IDs the NWB backend
  produces, so the two can be compared directly. Units are good units only
  (`label == 1`), and `x/y/z` are BWM's bregma-relative meters, which makes
  BWM the canonical source for coordinates.
- **Trials come from the Brain Wide Map paper's frozen trials table**
  (`provenance.yaml`: `trials_table: bwm_tables/trials.pqt`). All 13
  canonical fields are present, including `stimOff_times`, which
  ibl-ai-agent's docs don't list. `bwm_include` (290 of 410 trials in
  `d23a44ef`) is kept as an extra column; filtering is a later choice.
- **Behaviour is declared missing:** it lives in the separate
  `bwm_behavior` dataset, which isn't loaded yet.

**Cross-backend results against the NWB backend on `d23a44ef`:**
- Exactly the NWB units with `label == 1.0`, with the same IDs.
- Every unit's spike count is identical, and spike times are within 50 µs.
- 12 of 13 trial fields are identical.
- **`firstMovement_times` differs, and structurally.** Both sources put
  movement onsets on a 1 kHz grid, offset by a constant phase (BWM 0.12402 ms;
  ONE's `2025-03-03` revision, which NWB matches, 0.42318 ms). Every
  difference is that phase plus whole samples (−2 in 4 trials, −1 in 290,
  0 in 115, +1 in 1): 405 of 410 trials differ by under 1 ms, and the worst
  is 1.70084 ms. So ONE's 2025-03-03 revision re-extracted movement onsets on
  a shifted resampling grid. **Phase 2 consequence:** a movement-onset target
  can move by up to ~2 ms, occasionally into the next 20 ms bin, depending on
  which backend it's built from. Pick one source for that target and record
  the choice (Phase 2 ADR).

**Performance:** 3.5 s per session. A stable `argsort` on the int64 unit
indices took 14.8 of an initial 16.5 s. Casting to `uint16` (at most 459
good units per insertion in BWM) lets NumPy use its linear-time radix sort,
with an identical order. It falls back to int64 if an insertion ever has
more than 65,535 units.

**Alternatives considered:** Using ibl-ai-agent's package as a dependency
(not on PyPI, and it brings their whole tool stack); `blosc`/`blosc2` instead
of `numcodecs` (the shards are written by `numcodecs`, so its decoder is the
one guaranteed to match).

### 2026-09-27 — Add `remfile` to stream NWB files from DANDI

**Decision:** Add `remfile` (0.1.15; depends only on `h5py`, `numpy`,
`requests`) as a dependency. `load_session_nwb` accepts an `https://` URL as
well as a local path, and streams the URL with `remfile` → `h5py` → `pynwb`,
fetching only the byte ranges it reads. `dandi_asset_url(eid)` resolves an
IBL eid to its `desc-processed` asset's S3 URL through DANDI's REST API (stdlib
`urllib`, so the much heavier `dandi` package isn't needed).

**Why:** ROADMAP Phase 1 says to stream `desc-processed` assets from 000409
via `remfile` + `h5py` + `pynwb` and not bulk-download; ROADMAP §8 lists
`remfile` as the Phase 1 streaming addition.

**Pinned DANDI version:** the resolver reads the published, immutable
version **`0.260309.1324`** (2026-03-09), never the draft, which can change
under us just as ONE's data revisions do. For `d23a44ef` the asset in that
version has the same ID and SHA-256 as the local, checksum-verified file.
Moving to a newer version is a deliberate change, recorded here.

**Verified:** streaming session `d23a44ef` equals the local load in every
field: units, trials, all 1,961 spike trains, all 10 behaviour series and
the capability report (`test_streamed_load_equals_local_load`).

**Cost, measured on this machine:** opening the file takes about 38 s
(pynwb reads the file's structure with many small requests, so it's
latency-bound). A full streamed session load took about **20 minutes**,
versus about 6 s from a local file. `remfile` fetches one range at a time
over one connection, and a single connection to S3 from here gets about
0.3–0.4 MB/s. Streaming is therefore for opening a session once, or for
reading just its metadata or a part of it. Anything repeated should go
through the Phase 1 cache (`data/cache.py`), as the roadmap already says.
`remfile`'s `disk_cache` option and its private `_max_threads` setting could
speed this up, but they aren't used: the first belongs with the cache
design, and the second is a private API.

**Alternatives considered:** the `dandi` package (much heavier, and only
needed here for the asset lookup); downloading whole files (what the
roadmap says not to do across the dataset).

**Consequences:** the two streaming tests only run when
`NEURODECODER_NETWORK_TESTS=1` is set, so CI and routine runs stay offline
and fast. The full comparison test takes about 20 minutes here.

### 2026-09-27 — DANDI NWB backend mappings (`neurodecoder/data/backends/dandi_nwb.py`)

**Decision:** `load_session_nwb(source)` reads DANDI 000409 `desc-processed`
NWB files into a `Session`, from a local path or streamed from a URL (see
"Add `remfile`"). Every mapping was checked value by value against ONE for session
`d23a44ef-1402-4ed7-97f5-47e9a7a504d9` (sorting revision `2024-05-06`, trials
revision `2025-03-03`), with zero differences:

| Canonical | NWB | Check against ONE |
|---|---|---|
| `intervals_0/1`, `stimOn/stimOff/goCue/firstMovement/response/feedback_times` | `start_time`, `stop_time`, `gabor_stimulus_onset/offset_time`, `auditory_cue_time`, `wheel_movement_onset_time`, `choice_registration_time`, `feedback_time` | max \|diff\| = 0 on all 410 trials |
| `choice` | `mouse_wheel_choice`: `clockwise` → **+1**, `counter_clockwise` → **−1** | crosstab exact: 118 / 292, no off-diagonal |
| `feedbackType` | `is_mouse_rewarded` True → +1, False → −1 | exact: 304 / 106 |
| `contrastLeft/Right` | `gabor_stimulus_contrast` (**percent**) split by `gabor_stimulus_side`, ÷100, NaN off-side | exact |
| `probabilityLeft` | `probability_left` | exact |
| `label` | `ibl_quality_score` (0, ⅓, ⅔, 1) | equals ONE `clusters.metrics.label` |
| `depths` | `distance_from_probe_tip_um` | equals ONE `clusters.depths` |
| `firing_rate` | `firing_rate` | exact |

- **Unit ID** is NWB's `unit_name`, e.g. `probe00_0`, i.e. probe plus ONE's
  `cluster_id` (same order as ONE on both probes). `cluster_id`,
  `cluster_uuid` and `location` (full Allen region name, via each unit's
  `max_electrode`) are kept as extra columns for matching against other
  backends. `probe_name` uses IBL's lowercase `probe00`; NWB's column says
  `Probe00`.
- **Unrecognised values raise.** An unknown choice string (e.g. a no-go
  trial, of which this session has none), an unknown stimulus side, or a
  contrast outside IBL's percent set (0, 6.25, 12.5, 25, 50, 100) raises
  instead of being guessed. The percent check also catches a source that
  switches to fractions.
- **`behaviour.wheel` is the raw `WheelPosition`** (radians, 755,552 irregular
  samples), not IBL's smoothed 1 kHz position/velocity. The smoothing filter is
  a preprocessing choice (R6), and ROADMAP Phase 2 warns it changes wheel R².
- **`time_bounds` is the span of all loaded data** (spikes, trial times,
  wheel), because the processed file holds no raw recording to define it. For
  this session that's 0.0008–3668.940 s. The wheel's last sample is 2 ms after
  the last spike, so spike-only bounds would wrongly reject it.
- **Declared missing, with reasons:** `units.acronym` (NWB has full region
  names; mapping names to acronyms isn't implemented), `units.x/y/z` (NWB
  electrode coordinates are Allen CCF µm, and the BWM convention isn't
  verified yet), and `behaviour.lick` (events, not a sampled signal).
- **Per-camera video signals are loaded**, one key per camera, each on its own
  clock:
  - `motion_energy_{left,right,body}` come from
    `motion_energy/{Left,Right,Body}CameraMotionEnergy`.
  - `pupil_{left,right}` come from the **raw** `pupil/{Left,Right}PupilDiameter`,
    not `...Smoothed`, because smoothing is preprocessing (R6). The NaNs in it
    are kept (25 left, 221 right in `d23a44ef`).
  - `pose_{left,right,body}` come from `pose_estimation/{Left,Right,Body}Camera`.
    Each camera's keypoints are stacked into one series with named columns
    `{keypoint}_x`, `{keypoint}_y` (px) and `{keypoint}_likelihood`, keypoints
    in alphabetical order (body 1, left 6, right 11 in `d23a44ef`).
  - The **likelihood is kept unthresholded**, matching ibl-ai-agent's
    `likelihood_thr=0`, because filtering low-confidence points is a
    preprocessing choice.
  - Keypoint names are NWB's series names in snake_case
    (`RightPupilBottom` → `right_pupil_bottom`), taken as the file gives them.
    Matching them to BWM's pose naming is a cross-backend question for later.
- **Checked before stacking:** within each camera, every pose keypoint shares
  exactly the same timestamps, and those equal the camera's motion-energy and
  pupil timestamps. The backend raises if a keypoint is on its own clock, and
  it rejects rate-sampled series without timestamps rather than rebuilding
  them.
- **A signal absent from a file is declared missing** ("no `module/name` in
  this NWB file") instead of raising. About 4% of BWM sessions have no pose,
  per ibl-ai-agent's docs.

**Resolved contract issue:** the first version of this backend couldn't hold
IBL's three cameras (body ~30 Hz, left ~60 Hz, right ~150 Hz, with different
timestamps), because the `Session` contract had one `TimeSeries` per field.
The contract now has one key per camera (see "The `Session` contract"), and
this backend loads all of them.

**Consequences:** Loading this session takes about 6 s, including full
contract validation. The tests needing the real file are skipped in CI (the
file is 1.3 GB and not in the repo); the mapping tests run everywhere.

### 2026-09-27 — The `Session` contract (Phase 1, `neurodecoder/data/session.py`)

**Decision:** Every backend returns a `Session` that validates itself on
construction, so an invalid one can't exist:
- **Canonical names are IBL's ALF names**, as used by the compressed BWM
  (`stimOn_times`, `goCue_times`, `firstMovement_times`, `response_times`,
  `feedback_times`, `intervals_0/1`, `choice`, `feedbackType`,
  `contrastLeft/Right`, `probabilityLeft`, plus `stimOff_times`; units:
  `probe_name`, `acronym`, `x/y/z`, `depths`, `label`, `firing_rate`). Other
  backends (NWB, whose names differ, see `docs/PRIOR_ART.md` §D) map *to* these.
- **Every canonical field must be declared present or missing with a
  reason.** Present means it exists in the data and isn't entirely NaN;
  missing means it's absent (no placeholder column). Unknown field names are
  rejected, which catches typos. This is §7's "missing means missing",
  enforced in code.
- **Times are seconds on the session clock.** Spike times and behaviour
  timestamps must be finite and sorted, and all spike, trial and behaviour
  times must fall within `time_bounds`. A span over 24 h is rejected as
  "probably milliseconds", the trap SpikeLab's loader sets.
- Extra backend-specific columns (e.g. `cluster_uuid`) are allowed.
- **Behaviour signals from video have one key per camera** (revised
  2026-09-27): `motion_energy_left/right/body`, `pupil_left/right`,
  `pose_left/right/body`, plus `wheel` and `lick`. IBL films each session with
  three cameras at different rates (body ~30 Hz, left ~60 Hz, right ~150 Hz
  in session `d23a44ef`), with different timestamps, so a single
  `motion_energy` series can't hold them. The body camera doesn't see the
  pupil, so there's no `pupil_body`. The original single-key names
  (`motion_energy`, `pose`, `pupil`) are now rejected as unknown.
- **Multi-channel series must name their columns.** A `TimeSeries` with
  `(n_samples, n_channels)` data needs `channel_names` (one unique name per
  column), and 1-D data takes none. Per-camera pose is several tracked
  keypoints, each with x and y, and an unlabelled `(n, 10)` array would leave
  "which column is the left paw's y?" to guesswork.

**Why:** Phase 1's cross-backend tests are only meaningful if all backends
produce the same shape of object with the same field names, and if a missing
field can't masquerade as data.

**Alternatives considered:** Validating in a separate function that callers
could forget to run; NWB's column names as canonical (rejected: BWM is the
primary training source).

**Consequences:** The roadmap's interface says `Session.available:
CapabilitySet`; the class is named `Capabilities`, with the same role. What a
unit ID is (`cluster_uuid` vs `(pid, cluster_id)`) is left to each backend.
BWM and NWB were later shown to share `(probe_name, cluster_id)` exactly (see
`docs/PRIOR_ART.md` §D).

**Correction (2026-09-27):** an earlier version of this entry said BWM stores
trial times as float32 and so needs a ~1 ms comparison tolerance. That came
from ibl-ai-agent's schema docs. The extracted `bwm_ephys` 1.2.1
`metadata/trials.parquet` stores them as **double (float64)**, so trial times
can be compared exactly. BWM's per-unit `firing_rate` *is* single precision
(differs from NWB by at most 3.45e-6 on `d23a44ef`), so compare that one with
a tolerance.

### 2026-09-27 — NWB intake reads with `pynwb` directly; SpikeLab is not a dependency

**Decision:** `neurodecoder/nwb/` and the NWB data backend read files with
`pynwb`, which is already in the Fixed stack (§8). SpikeLab is **not** added
as a dependency. If we later want its analysis methods (STTC, slice stacks,
`RateData`), we wrap it by converting our `Session` into a `SpikeData`, and
still don't fork it. That's the Phase 0 task 3 "reuse vs wrap" outcome.

**Why:** This rests on actually loading DANDI 000409's processed NWB for
session `d23a44ef-…` through SpikeLab's `load_spikedata_from_nwb`
(2026-09-27). It works: 1,961 units and 61,981,600 spikes in 15 s, matching
ONE exactly (see `docs/PRIOR_ART.md` §A). But for our purposes:
1. It keeps 5 per-unit attributes out of the units table's 27 columns and
   drops `cluster_uuid` (the key for matching units across backends) and
   every IBL QC column `qc/` needs (`ibl_quality_score`,
   `sliding_rp_violation`, `noise_cutoff`, `presence_ratio`, amplitudes,
   drift).
2. It fills gaps silently: duration is inferred from the last spike and a
   missing start time becomes 0.0. §7 says missing means missing.
3. Spike times are in milliseconds; our interface (ROADMAP Phase 1) uses
   seconds, so every call would need a conversion that's easy to get wrong.
4. It reads spikes only, no trials or behaviour, which the same file holds
   (trials table; `wheel`, `motion_energy`, `pose_estimation`, `pupil`,
   `lick_times` processing modules).

A few lines of `pynwb` read all of it.

**Alternatives considered:** Wrapping SpikeLab's loader and reading the
missing columns separately (two readers for one file, for no gain); forking
it (ruled out by CLAUDE.md §2).

**Consequences:** One fewer dependency. The "prefer wrapping SpikeLab" rule in
§2 still applies to *analysis*; this decision covers *intake* only.

### 2026-09-27 — Downloaded data lives in `~/data/neurodecoder`, outside the repo

**Decision:** All downloaded datasets live under `~/data/neurodecoder/`, not
in the git repo:

| Path | What | Source | Checksum |
|---|---|---|---|
| `bwm_compressed/archives/bwm_ephys-1.2.1.tar` | ibl-ai-agent compressed BWM, spikes/units | `ibl-brain-wide-map-public` S3, `resources/ibl-agent-data/` | SHA-1 `b18c5c7a2944be510800010eb3df90aac84a2a52` |
| `bwm_compressed/archives/bwm_behavior-2.0.0.tar` | ibl-ai-agent compressed BWM, behaviour | same | SHA-1 `1c37dd1c38d46ec80067c8a25772dfe2468a1ce1` |
| `dandi/000409/sub-DY-016/…_ses-d23a44ef-…_desc-processed_behavior+ecephys.nwb` | DANDI 000409 processed NWB, same session as the NEDS run | DANDI asset `ecf201ee-c535-4371-a2bd-b6da93c0fbb5` | SHA-256 `563b902729aab23bf6ce3457396629521dcd2ffe9244e551ce46bf16c940dab4` |

Checksums come from ibl-ai-agent's `scripts/download_datasets.py` and DANDI's
asset metadata, and each download is verified against them.

**Why:** About 10.5 GB of archives plus about 12 GB extracted shouldn't sit
inside the repo tree (git operations, editor indexing, backups). A fixed path
recorded here means future sessions and forks find the data rather than
downloading it again.

**Alternatives considered:** A gitignored `datasets/` folder in the repo.
Also running ibl-ai-agent's own installer script, rejected because it writes
config into their repo; plain `curl` plus checksum verification is more
transparent.

**Consequences:** Phase 1's `data/backends/bwm_compressed.py` and the Phase 1
cache should take this root from config under `configs/`, not hard-code it.
Note that ibl-ai-agent's downloader fetches **`bwm_ephys` 1.2.1**, while
their `docs/bwm/README.md` still describes 1.2.0. Trust the extracted
dataset's own schema/version files over their README.

### 2026-09-27 — CI installs CPU-only PyTorch

**Decision:** `.github/workflows/tests.yml` installs `torch` from PyTorch's CPU
wheel index before `pip install -e ".[dev]"`.

**Why:** The default PyPI `torch` wheel on Linux bundles CUDA libraries
(about 2 GB). CI has no GPU and our Phases 0–4 need none, so CPU wheels keep
every run fast without changing which package versions get resolved.

**Consequences:** When GPU-dependent tests arrive (Phase 5+), they need a
separate job or marker; this job stays CPU-only.

### 2026-09-27 — NEDS's loader depends on a Hugging Face org that is now empty

**Decision:** Patched `load_ibl_dataset` in
`external/NEDS/src/utils/dataset_utils.py` (its only
`get_user_datasets(...)` call, line 209) to list the local `*_aligned`
directories under `cache_dir` instead of querying Hugging Face. Same
`org/eid_aligned` format, so the rest of the loader is unchanged. It is a
local patch to a gitignored clone.

**Why:** `get_user_datasets` calls `datasets.list_datasets()`, which pages
through every public dataset on Hugging Face, then filters to
`ibl-repro-ephys/`. The list is used only to check that the eid is
"published". The data itself is then read from local disk
(`load_from_disk(f"{cache_dir}/{eid}_aligned")`), i.e. whatever NEDS's own
`prepare_data.py` wrote. Two failures stack up:
(1) paging all of Hugging Face without logging in hits `HTTP 429 Too Many
Requests`; (2) even with no rate limit, a targeted query
(`HfApi().list_datasets(author="ibl-repro-ephys")`) returns **0 datasets**, so
the check would raise `ValueError: ... not found in the user's datasets`.
`create_dataset.py` and `train.py` both go through this loader, so NEDS's
training pipeline cannot run for an outside user without this patch, whatever
the package versions.

**Alternatives considered:** Pinning versions (doesn't help: the dependency is
on an external org's contents, not a package API); requesting access to the
org (unknown if it still exists privately).

**Consequences:** The data being trained on is exactly what `prepare_data.py`
produced locally, which was verified: prep's train/val/test rows
(249/36/72) match `create_dataset.py`'s written files exactly. Fourth NEDS
environment break so far; each has been a distinct cause (native build,
`ibllib` API, macOS `spawn`, Hugging Face org), not the same one repeating.

### 2026-09-27 — NEDS data prep loops forever on macOS unless forced to `fork`

**Decision:** Patched `external/NEDS/src/prepare_data.py` to call
`multiprocessing.set_start_method("fork", force=True)` right after its stdlib
imports. Local patch to a gitignored clone, as with the `SessionLoader` fix.

**Why:** `src/utils/ibl_data_utils.py` creates a `multiprocessing.Pool` in
three places (lines 201, 458, 491), even with `n_workers=1`, and
`prepare_data.py` has no `if __name__ == "__main__":` guard. macOS defaults
to the `spawn` start method, so every pool worker re-imports and re-executes
the whole script: it re-downloads session data, fails to start its own pool
(`RuntimeError: An attempt has been made to start a new process before the
current process has finished its bootstrapping phase`), dies, and the parent
pool respawns it. **The parent never exits.** One run did this silently for
2.5 hours (532 `RuntimeError`s in its log), downloading into the same cache
directory a later run was using. NEDS was developed on Linux SLURM clusters,
where the default is `fork`, so this never shows up there.

**Alternatives considered:** Adding a `__main__` guard (more invasive, since
the whole script body is top-level code); running on Linux (not available).

**Consequences:** Anyone reproducing NEDS on macOS needs this patch. More
generally: **a crashing child process does not mean the job exited.** Before
relaunching any background data job, check with `pgrep` that the previous
one is gone, and write the exit code to the log instead of trusting a piped
command's status (`cmd | tail` reports `tail`'s exit code, which is how the
first `SessionLoader` crash showed up as "exit 0"). Data written while two
runs shared a cache directory was discarded and re-downloaded, not reused.

### 2026-09-27 — `ONE-api`/`ibllib` installs on this machine need llvmlite/numba pinned first

**Decision:** Before installing `ONE-api`, `ibllib`, or anything that pulls in
`numba` (directly or via `iblutil`), run
`pip install "llvmlite==0.44.0" "numba<0.61.3,>=0.60"` first, in whatever venv
you're targeting.

**Why:** `pip install ONE-api` (and separately, installing NEDS's `env.yaml`
deps in a plain venv) fails building `llvmlite` from source — `numba`'s latest
version pulls `llvmlite>=0.49`, which has no prebuilt wheel for macOS x86_64+
Python 3.11 on PyPI (source build needs a matching LLVM toolchain we don't
have). `llvmlite==0.44.0` / a compatible `numba<0.61.3` do have prebuilt
wheels for this platform. Installing them first satisfies the pin before pip
tries to build the newer, wheel-less version as a transitive dependency.

**Alternatives considered:** Building LLVM via Homebrew to satisfy the source
build — much heavier, not attempted.

**Consequences:** Applies to this machine's `.venv` (project) and to
`external/NEDS/.venv` (gitignored, for the Phase 0 reproduction attempt).
Anyone re-running either install on similar hardware will hit the same failure
without this pin.

### 2026-09-27 — NEDS's `SessionLoader` call is incompatible with current `ibllib`

**Decision:** Patched both `SessionLoader` call sites in
`external/NEDS/src/utils/ibl_data_utils.py` (lines 107 and 263:
`SessionLoader(one, eid=eid)` → `SessionLoader(one=one, eid=eid)`) to keep the
Phase 0 reproduction attempt moving. This is a local patch to a gitignored
external clone, not a change to our own code. The first attempt patched only
line 107. Line 263 runs inside a pool worker and crashed the next run, so when
fixing an API break, grep for every call site first. The other IBL calls in
that file (`SpikeSortingLoader`, `BrainRegions`, `get_spike_counts_in_bins`)
were checked against `ibllib` 4.0.1 and are compatible.

**Why:** `external/NEDS` (cloned per the Phase 0 audit) pins `ibllib`
unversioned in `env.yaml`, so a fresh install pulls current `ibllib` (4.0.1).
Current `ibllib`'s `SessionLoader` is a dataclass with `one` as keyword-only;
NEDS's code passes it positionally, which now raises
`TypeError: SessionLoader.__init__() takes 1 positional argument but 2 ... were given`.
This is exactly the "NEDS environment resolution" failure point
ROADMAP.md Phase 0 predicted in advance.

**Alternatives considered:** Pinning `ibllib` to an older, contemporaneous
version instead of patching the call site — not attempted yet; would be the
more faithful fix if more breaks of the same kind turn up, since patching
call sites one at a time doesn't scale if the API drift is broader than this
one call.

**Consequences:** Confirms Phase 0 task 1's expected failure mode. If more
`ibllib`/`iblatlas`/ONE-api API breaks surface while trying to reproduce a
NEDS number, switch strategy to pinning old versions of the whole IBL stack
rather than continuing to patch individual call sites.

---

### 2026-09-27 — Phase 9 (future-horizon prediction) stays at full scope

**Decision:** Do not cut ROADMAP.md Phase 9 to a stretch goal. Keep it at its
original 30–60 h scope.

**Why:** Phase 9 is explicitly conditional on auditing SpikeProphecy
(arXiv 2605.12992) first. Cloned and read its source
(`external/SpikeProphecy`, gitignored): it forecasts future *neural population
spike counts* from past spikes, evaluated with a population-similarity metric
decomposition — a different task from Phase 9's future-*behaviour* decoding
evaluated against an autocorrelation-of-behaviour baseline. Its only
behaviour-decoding code is a same-timestep auxiliary classification head in an
Appendix C distillation experiment, not a t+100/250/500 ms forecast. See
`docs/PRIOR_ART.md` §E for the full audit.

**Alternatives considered:** Cutting Phase 9 to the small controlled
experiment the roadmap describes as the fallback — rejected because the
condition that triggers that fallback ("SpikeProphecy already covers this")
is false.

**Consequences:** Phase 9, if reached, still needs its own behaviour-forecast
implementation and its own autocorrelation baseline; nothing from
SpikeProphecy is directly reusable for it. `src/data/ibl_behavior_loader.py`'s
IBL trial-field extraction pattern may still be worth a look when we write
`neurodecoder/targets/`, independent of the forecasting question.

### 2026-09-27 — Pin local venv to Python 3.11 via Homebrew

**Decision:** Installed `python@3.11` via Homebrew (`/usr/local/opt/python@3.11`)
and recreated `.venv` with that interpreter, matching the `>=3.11,<3.12` pin in
`pyproject.toml`.

**Why:** The system Python was 3.13, which is outside the Fixed stack's pin
(§8). Installing `python@3.11` required first updating Xcode Command Line
Tools (26.2 → 26.6), which needed the user's own admin auth and was done by
the user, not this session.

**Alternatives considered:** Relaxing the pyproject.toml Python pin to allow
3.13; using pyenv instead of Homebrew.

**Consequences:** `.venv` now resolves to Python 3.11.16. `pytest` and
`pre-commit run --all-files` both re-verified passing under it.

## Template

### YYYY-MM-DD — Short title

**Decision:**

**Why:**

**Alternatives considered:**

**Consequences:**
