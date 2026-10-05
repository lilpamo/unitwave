# Negative Results

Failed experiments and dead ends. The most valuable and easiest-to-lose record
in this repo. One entry per experiment, newest first.

---

### 2026-10-05 — Dropping self-aligning shifts makes the periodic shift null invent responses

**What was tried (signed off by the user 2026-10-05, then withdrawn before shipping):** a
fix for the entry below.
- **The rule:** a shift whose self-overlap is above 0.5 is no null draw. The
  self-overlap is the mean, over events, of the other event onsets within 0.5 s (the
  windows' span) of each shifted event.
- **Also:** refuse presentations closer together than the windows.

**Hypothesis:** for strictly periodic presentations, the shifts at multiples of the
period copy the true response into the null. Dropping them would give back the power,
and the null would stay valid because the shifts are chosen from event times alone.

**What it looked like on real data (prototype only):**
- **Allen flashes:** 0 → 108 of 430 units responsive.
- **Drifting gratings:** 229 → 235.
- **MC_Maze:** 13 → 13.
- **IBL and Steinmetz:** unchanged; their self-overlap never exceeds 0.29 and 0.39, so
  no shift was dropped.

**Result: it fails calibration.** 200 hand-built Poisson units ignoring presentations
every 2 s, 5 seeds:
- **The proposed null:** 4, 6, 7, 9 and 12 units labelled responsive after BH; 7–14% of
  p below 0.05; every false discovery at the smallest possible p.
- **Today's null on the same units:** 0 in every run, about 5% below 0.05.
- **Other designs** (near-regular MC_Maze-like trials with CV 0.11, drifting-grating-like
  periodic blocks, IBL-like irregular trials): no false discoveries with either rule.

**Why it failed:**
- **The draws aren't distinct:** with strictly periodic events, shifts a whole number of
  periods apart give almost the same statistic. The 57,000 draws are a few distinct
  phases of one cycle, about (2 − 1) s / 0.5 s of windows.
- **The reported p is far too small:** when the true value beats them all, which
  happens a few percent of the time under the null, p is reported as 1/57,711.
- **Today's null is valid only by accident:** it is calibrated because the shifts at
  multiples of the period reproduce the true value and tie with it. That same fact
  removes its power.
- **No version of this test can give small p-values** for strictly periodic events: the
  shift null can't separate the response from the stimulus cycle.

**Consequence:** the 108 flash units are not a result. The code was reverted before
commit; today's null (calibrated) is unchanged. The calibration test is kept for the
next attempt.

**Do not retry unless:** the test isn't a shift null.
- **A paired test within presentations:** each presentation's response against its own
  baseline, with presentations as the samples, and its own calibration on hand-built
  periodic units, bursting units included.
- **Or refuse:** decline the test on strictly periodic presentations, saying why.

### 2026-10-02 — Responsiveness's shift null has almost no power for periodic presentations

**What was tried:** Studio's responsiveness test on the Allen Visual Coding session
(DANDI 000021, ses-721123822), for each passive task.
- **The test:** the mean of response (0–300 ms) minus baseline (−200–0 ms), against
  the spike train circularly shifted, BH across units, as built for IBL.
- **The tasks:** flashes, static gratings and drifting gratings.

**Hypothesis:** visual units respond to flashes and gratings, so many would be labelled
responsive, as on IBL stimulus onsets.

**Result:**
- **Flashes:** 0 of 430 QC units responsive. The most flash-driven unit (+27.9 Hz
  over baseline) gets p = 0.014 against 56,001 shifts, and nothing survives BH. Yet
  171 of 430 units tell dark from light flashes.
- **Static gratings:** 15 of 448.
- **Drifting gratings:** 229 of 447.

**Why it failed:**
- **Flashes are strictly periodic:** every 2.002 s, with no gaps. Shifting the spike
  train by any whole number of periods re-aligns it with the flashes, so the null
  holds near-copies of the true response and the test loses its power. It errs
  conservative (it misses responses; it doesn't invent them), but "0 responsive"
  reads as a finding.
- **Static gratings run back to back** every 0.250 s. The baseline window lies inside
  the previous presentation's response, so response and baseline barely differ.
- **Drifting gratings** recur every 3.003 s, but in three blocks with gaps, so most
  shifts don't re-align. Some power remains.

**Do not retry unless:** the null is built for presentations. Options:
- compare against blank presentations or the spontaneous intervals;
- use a paired test of response against baseline within presentations, with
  presentations spaced apart;
- refuse the shift test when events are periodic.

Until then, Studio's responsiveness counts for these tasks aren't findings. The
choice of fix is the user's.

### 2026-09-30 — Phase 3 decision gate: NOT PASSED on the confirmation set; Phase 3 parked

**Gate:** `model_with_task` vs `null_trialstruct`, per target, one-sided Wilcoxon
over test sessions at α = 0.05, on the confirmation set
(`configs/runs/phase3_confirmation.yaml`: 10 sessions no diagnostic had touched),
seed 0.

**Runs:**
- choice: `runs/20260929T114437Z_phase3_confirmation`. It ran on HEAD `5bfb109`
  plus the staged diff later committed unchanged as `35bb7bc`. The machine lost
  power during wheel velocity, after choice and block had completed.
- wheel velocity and movement state: `runs/20260929T192827Z_phase3_confirmation_resume`
  (`35bb7bc`, clean tree). It used the same config except `name` and `targets`,
  and its split hashes equal the interrupted run's.
- block on the fixed split (adjacent pairs, scored per fold):
  `runs/20260930T082034Z_phase3_confirmation_block_pairs`. It ran on `35bb7bc`
  plus the staged fix, which is committed with this entry. It used the same
  config except `name` and `targets`.
- block on the first-version split: `runs/20260929T114437Z_phase3_confirmation/block`.
  **Invalid**: that split scores AUROC ~0 on noise (entry below).

**Gate verdicts, as the contract printed them:**
- choice: model_with_task does NOT beat null_trialstruct (median Δauroc +0.001,
  wins 6/10 sessions, Wilcoxon p = 0.0742). **NOT PASSED.**
- wheel_velocity: model_with_task does NOT beat null_trialstruct (median Δr2
  +0.023, wins 6/10 sessions, Wilcoxon p = 0.461). **NOT PASSED.**
- movement_state: model_with_task beats null_trialstruct (median Δauroc +0.315,
  wins 10/10 sessions, Wilcoxon p = 0.000977). **PASSED.**
- block, first-version split: invalid, no verdict.
- block, fixed split: model_with_task does NOT beat null_trialstruct (median
  Δauroc -0.001, wins 4/10 sessions, Wilcoxon p = 0.839). **NOT PASSED.**

**Spikes-alone verdicts (model vs null_trialstruct):**
- choice: not beaten (Δauroc -0.231, wins 1/10, p = 0.997).
- wheel_velocity: not beaten (Δr2 -0.019, wins 4/10, p = 0.577).
- movement_state: beaten (Δauroc +0.317, wins 10/10, p = 0.000977).
- block, fixed split: not beaten (Δauroc -0.432, wins 0/10, p = 1).

**Passes within-session:** movement state only.

**Not used for the gate:** `runs/20260929T223355Z_phase3_first_table`, the first
table's 10 sessions re-run at `35bb7bc`. Those sessions were already seen, and its
block rows use the first-version split, so they are invalid.

The full tables and every verdict line are in each run's `report.txt`.

**Status:** gate not passed. Phase 3 is parked (2026-09-30) for a change of
direction.

### 2026-09-30 — Leave-one-block-out for block, first version: inverted a decoder with no signal

**What was tried:** B1 from the gate investigation (`docs/DECISIONS.md`,
2026-09-29), as first built: each biased block held out once as a fold's test set,
and each session's held-out predictions pooled over its folds before scoring.
Run on both session sets at `35bb7bc`:
- confirmation set: `runs/20260929T114437Z_phase3_confirmation/block`;
- first table, re-run: `runs/20260929T223355Z_phase3_first_table/block`.

**Result:** `model` scored AUROC 0.000 on the confirmation set (median, q25 and q75
all 0.000; balanced accuracy 0.073) and 0.000 on the first table (q75 0.001).
`baseline_ridge`, `baseline_rrr` and `ceiling_within` were the same or close, and
both nulls sat far below chance: `null_shuffle` 0.179 and `null_pseudosession`
0.200 on the confirmation set. **These block rows are not measurements.** They are
superseded by the re-run on the fixed split.

**Why:** holding a block out shifts the training set's class balance against the
held-out label.
- In all 10 confirmation sessions, the fraction of label 1 among a fold's training
  trials was lower when a label-1 block was held out (e.g. `0a018f12`: 0.544 vs
  0.683). This describes targets only; no model was fit to find it.
- A decoder with little signal predicts close to its training base rate. Every
  held-out block then gets a score on the wrong side of the others, and pooling
  the folds turns that into a ranking: AUROC ~0.
- Synthetic check (pure-noise features, IBL-generated block labels, the real
  `LogisticDecoder`, predictions pooled over folds): AUROC 0.000, 0.024, 0.000,
  0.000, 0.000 in 5 sessions.

**Second attempt, also failed:** holding out adjacent pairs of blocks (one 0.2,
one 0.8) but still pooling. On pure noise through the real split, decoders and
contract (`tests/test_lobo_no_signal.py`), the medians over 6 sessions were
`model` 0.401, `model_with_task` 0.407, `null_shuffle` 0.341 and
`null_pseudosession` 0.340. The two blocks of a pair differ in length (20–100
trials), so a fold's training balance still tracks its test set's composition,
and pooled predictions still carry each fold's offset. Shifted and pseudo labels
make some test sets even more one-sided, so the nulls were hit hardest.

**What fixed it:** pairs, and scoring each fold on its own, with a session's
metric the mean over its folds (a fold's offset is then shared by all its
samples and can't rank them). On the same noise test: `model` 0.489,
`model_with_task` 0.532, `null_shuffle` 0.519, `null_pseudosession` 0.499.

**Lessons:**
- Pooling predictions from folds whose training sets differ in class balance
  does not give a valid AUROC.
- The split's tests checked partitions, gaps and bookkeeping but never what a
  decoder with no signal scores. Any fold-based scheme now needs that test,
  through the real contract, before it runs on real data.

### 2026-09-29 — Phase 3 first six-row table: decision gate not passed

**Run:** `runs/20260929T070943Z_phase3_first_table`, git SHA `81edcfe` (clean),
config `configs/runs/phase3_first_table.yaml`.

**Verdict lines (model vs null_trialstruct), as the contract printed them:**
- choice: model does NOT beat null_trialstruct (median Δauroc -0.071, wins 4/10 sessions, Wilcoxon p = 0.968): this is not decoding.
- block: model does NOT beat null_trialstruct (median Δauroc -0.307, wins 0/9 sessions, Wilcoxon p = 1): this is not decoding.
- wheel_velocity: model does NOT beat null_trialstruct (median Δr2 +0.053, wins 6/10 sessions, Wilcoxon p = 0.577): this is not decoding.
- movement_state: model beats null_trialstruct (median Δauroc +0.281, wins 10/10 sessions, Wilcoxon p = 0.000977).

The full six-row tables and the other verdict lines are in each target's
`report.txt` in the run directory.

**Status:** gate not passed. The investigation's gate call, on the confirmation set, is the 2026-09-30 entry above.

### 2026-09-27 — Reproducing one NEDS decoding number on one session (Phase 0 task 1)

**What was tried:** Cloned NEDS (`external/NEDS`, gitignored) and ran its own
pipeline on held-out test session `d23a44ef-1402-4ed7-97f5-47e9a7a504d9`, on
this Mac: `prepare_data.py` → `create_dataset.py` → `train.py` for 1 epoch as a
smoke test, to measure cost before a full run.

**Hypothesis:** The NEDS paper reports a decoding number for an identifiable
held-out session, and running the released code on that session with default
settings would land close to it.

**Result:** The pipeline runs end to end, but **a numeric reproduction is not
possible as specified.**
- Single-session results are published **only as averages and distributions
  over the 10 held-out sessions**. Figure 2B's per-session scatter plot doesn't
  identify sessions.
- Those numbers come from a random hyperparameter search over 50 models.
- No linear or reduced-rank baseline code is released.
- Getting the pipeline to run at all took four distinct local patches (native
  build pin, `ibllib` API change, macOS `spawn` loop, empty Hugging Face org),
  all in `docs/DECISIONS.md`.
- A fresh ONE download serves data revisions that postdate the paper (motion
  energy from May–June 2025), so even a perfect code match wouldn't guarantee
  identical inputs.
- Full single-session training would take roughly 11–14 h on this machine's
  GPU (MPS).

**Why it failed (best guess):** The roadmap task assumed per-session published
numbers, and they don't exist. It isn't a flaw in NEDS's science.

**Do not retry unless:** per-session NEDS numbers with eids become available
(e.g. from the authors), or you have a CUDA GPU and budget for the 10-session,
50-model-search setup. Otherwise compare in Phase 3/4: run our baselines on
NEDS's 10 test eids (`external/NEDS/data/test_eids.txt`) and compare against
the paper's 10-session averages, using balanced accuracy explicitly (see
`docs/PRIOR_ART.md` §C).

---

## Template

### YYYY-MM-DD — Short title

**What was tried:**

**Hypothesis:**

**Result:**

**Why it failed (best guess):**

**Do not retry unless:**
