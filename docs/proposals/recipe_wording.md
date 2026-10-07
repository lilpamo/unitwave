# Recipe wording, for the user's approval (step 9)

Drafted 2026-10-03, before any recipe file or UI is built (step 9, "wording first").
Once approved, each recipe becomes a file under `configs/recipes/`, and this text is
what the Recipes panel shows.

**Words in [brackets]** come from the session's task definition: the IBL value is
shown, and another task shows its own. Every number (windows, counts, α) is read from
the configs named, never typed into a recipe.

**Every step has six parts:**
- **Question:** what this step answers.
- **Why this answers it:** the reason, in plain language.
- **Analysis:** what runs, and with which parameters.
- **Null and correction:** what the result is tested against, and how many tests it
  is corrected over.
- **Needs:** what must exist first.
- **How to read it:** what the result shows, and what it does not.

**Offered only where it fits.** A recipe lists what it needs from the task and the
data. If a session lacks any of it, the panel still lists the recipe, greyed out, with
the reason.

**Every result is exploratory** unless it ran on sessions named as held out before it
was run (the analysis log, designed separately).

---

## Recipe 1. Which units respond to the [stimulus] beyond the movement that follows it?

**Needs from the task:** a declared stimulus event and movement event (`movement`), so
reaction times exist. **Offered:** IBL ([stimulus onset], [first movement]) and
MC_Maze ([go cue], [movement onset]). **Not offered:**
- **Steinmetz:** its trials hold the response time, not movement onset;
- **Allen Visual Coding:** passive viewing, no movement events.

### Step 1. Which units change their rate after the [stimulus]?

- **Question:** which units fire differently in the 0.3 s after the [stimulus] than
  in the 0.2 s before it?
- **Why this answers it:** a unit that responds to the [stimulus] should change its
  rate soon after it, on most trials.
- **Analysis:** the responsiveness test at [stimulus onset], on the trials the trial
  filters keep. The statistic is the mean over trials of (rate 0 to 0.3 s after −
  rate 0.2 s before) (configs/analysis.yaml).
- **Null and correction:** the unit's spike train is circularly shifted against the
  event times, by at least 10 s. This keeps the unit's own firing pattern and breaks
  only its timing relative to the [stimulus]. The test is two-sided, with
  Benjamini–Hochberg across the units tested (α = 0.05).
- **Needs:** QC-passing units; trials with a [stimulus onset] time. Trials without one
  are excluded and counted.
- **How to read it:** a responsive unit changes its rate (up or down) after the
  [stimulus], more than shifted copies of its own train do.
  - **It does not show the [stimulus] caused the change:** the animal usually starts
    moving within a few hundred milliseconds, inside the 0.3 s window. Steps 2 and 3
    separate the two.
  - **Not responsive is not the same as unaffected:** a weak or rare response can be
    missed, and fewer trials mean less power.

### Step 2. Which units follow the movement rather than the [stimulus]?

- **Question:** is a unit's rate locked to each trial's own [first movement]?
- **Why this answers it:** reaction times vary from trial to trial. A unit driven by
  the movement fires at the movement's time, wherever it falls after the [stimulus].
  Re-pairing trials with other reaction times blurs that unit. A unit driven by the
  [stimulus] blurs the same either way.
- **Analysis:** the movement-locking test. The statistic is the mean over trials of
  (rate 0 to 0.2 s after the [first movement] − rate 0.2 s before it)
  (configs/movement.yaml).
- **Null and correction:** reaction times permuted 10,000 times within [signed
  contrast] strata, so each trial keeps its [stimulus] and gets a reaction time from
  trials like it (seed 0). Two-sided, with Benjamini–Hochberg across the units tested
  (α = 0.05).
- **Needs:** trials with both a [stimulus onset] and a [first movement] time. Trials
  missing either are excluded and counted.
- **How to read it:** a locked unit is sharper at the true movement times than at
  re-paired ones.
  - **It does not show the unit ignores the [stimulus]:** a unit can be both locked
    and stimulus-responsive.
  - **It can over-call:** if reaction time tracks the animal's state (how engaged it
    is, say), a unit following that state can be labelled locked.
  - **Not locked is not the same as unrelated to movement:** only the [first
    movement] is tested.

### Step 3. Which units still respond when no movement falls in the window?

- **Question:** on trials where the animal moves only after the 0.3 s window ends,
  does the unit still change its rate after the [stimulus]?
- **Why this answers it:** on those trials, no [first movement] falls inside the
  window. A rate change there can't come from that movement.
- **Analysis:** the step 1 test, on the movement-free trials only: those whose
  [first movement] comes more than 0.3 s after the [stimulus]. A trial without a
  movement time isn't known to be movement-free, so it is excluded.
- **Null and correction:** as step 1 (shift null, two-sided, Benjamini–Hochberg
  across the units tested).
- **Needs:** step 1's trials, with a [first movement] time. The panel shows how many
  trials are left, and why the rest were excluded.
- **How to read it:** a unit responsive on movement-free trials responds before any
  [first movement].
  - **It does not rule out movement entirely:** smaller movements than the one
    detected (whisking, posture, a wheel twitch below threshold), and preparing to
    move, can still fall in the window.
  - **These trials are not a random sample:** late movements are more common in some
    conditions (on IBL, low contrast) and when the animal is less engaged. A
    difference from step 1 can come from which trials were kept, not only from the
    movement.
  - **Fewer trials, less power:** a unit responsive in step 1 but not here isn't
    shown to be movement-driven.

---

## Recipe 2. Does activity carry [choice] information beyond the [stimulus]?

**Needs from the task:**
- a two-level comparison on [choice], whose null permutes labels within the [stimulus]
  strata (`permute_within`);
- a decoding target on [choice], for step 3.

**Offered:** IBL Brain Wide Map sessions ([choice], within [signed contrast]) and
Steinmetz ([choice], within [contrasts]). On an IBL Phy folder, steps 1 and 2 run, and
step 3 says decoding needs a Brain Wide Map session. **Not offered:** Allen and
MC_Maze, which have no choice.

### Step 1. Do the trials split by [choice] look different?

- **Question:** around the [stimulus], do trials with a [left] and a [right] choice
  show different rates?
- **Why this answers it:** it is the plain look, before any test: a split PSTH shows
  when, and how much, the two kinds of trials differ.
- **Analysis:** the raster and PSTH of one unit, split by [choice], aligned to
  [stimulus onset], on the page's window and bins, with each group's trial count.
- **Null and correction:** none. This step is descriptive and makes no claim.
- **Needs:** trials with a [choice]. Trials without one ([no-go]) are excluded and
  counted.
- **How to read it:** a gap between the two curves shows a difference, not its cause.
  - **On correct trials the [choice] follows the [stimulus]:** a unit that only
    responds to the [stimulus] side shows a gap here too. Step 2 separates the two.
  - **Units picked by eye from this view are selected on the data shown:** a test on
    them afterwards isn't independent.

### Step 2. Which units separate [choice] at a fixed [stimulus]?

- **Question:** among trials with the same [stimulus], does the unit's rate tell the
  [choice] apart?
- **Why this answers it:** shuffling the [choice] only among trials with the same
  [stimulus] keeps the [stimulus] as it was. What's left to separate is the [choice].
- **Analysis:** the selectivity test for [choice], at [stimulus onset]. The statistic
  is the AUROC between [left] and [right] trials, on each trial's rate 0 to 0.3 s
  after the event (configs/selectivity.yaml).
- **Null and correction:** [choice] labels permuted 10,000 times within [signed
  contrast] strata (zero contrast split by side; seed 0). Two-sided, with
  Benjamini–Hochberg across the units tested (α = 0.05). Each [choice] needs at least
  10 trials, or the test is refused.
- **Needs:** trials with a [choice] and a [stimulus]. Others are excluded and
  counted.
- **How to read it:** a selective unit tells the [choice] apart beyond what the
  [stimulus] alone explains.
  - **It does not separate the [choice] from the movement that reports it:** the
    [choice] is the direction the animal turns. Within the window, a unit may follow
    the turn, not the decision.
  - **It does not show when the information appears:** one window is tested.
  - **Trials are treated as interchangeable within a [stimulus]:** slow drifts in a
    unit's rate are not accounted for.

### Step 3. Can the [choice] be read out beyond what the task alone predicts?

- **Question:** do the session's units predict the [choice] better than the task's
  own variables do, with no spikes?
- **Why this answers it:** the no-spikes baseline already knows the current
  [stimulus], the last 10 trials' [stimuli, choices and outcomes], and on IBL the block
  prior. Spikes that beat it carry [choice] information beyond all of these.
- **Analysis:** the decoding run for [choice]:
  - its window: the 100 ms before the [first movement];
  - the model: a logistic decoder on the QC-passing units' binned counts, trained on
    the early 80% of trials and tested on the late 20%, with a 2 s gap
    (configs/studio_decoding.yaml);
  - its rows, in the contract's order: the shuffle null, the no-spikes baseline, the
    linear baseline, the model and the within-session ceiling, with balanced accuracy
    and AUROC. On one session the linear baseline, the model and the ceiling are the
    same decoder, and the multi-session baseline isn't run (the table says so).
- **Null and correction:**
  - **Against the shuffle null:** targets circularly shifted 100 times within the
    session.
  - **Against the no-spikes baseline:** the gap is checked with a paired bootstrap
    over the test trials, 2,000 draws.
  - **Correction:** Benjamini–Hochberg across the comparisons (α = 0.05).
  - **The verdict line:** says plainly whether the model beat both.
- **Needs:** a decoding target on [choice] (IBL: Brain Wide Map sessions only), and
  enough trials of each [choice] in both halves.
- **How to read it:** beating the no-spikes baseline means the spikes add [choice]
  information to what the task variables give.
  - **It does not say which units carry it,** or in which region.
  - **It is one session:** whether it holds in other sessions is a separate question.
  - **It doesn't separate decision from movement:** the window ends at the movement,
    so its preparation can carry the [choice].
  - **Not beating the baseline is not proof of no information:** a linear readout
    over 100 ms can miss it.

---

## Recipe 3. How do brain regions differ in their responses to the [stimulus]?

**Needs from the data:**
- brain regions named in the Allen mouse atlas;
- a [stimulus] event;
- at least 5 sessions sharing regions (configs/summary.yaml).

**Offered:** IBL Brain Wide Map sets, Steinmetz files and Allen Visual Coding files.
**Not offered:** MC_Maze, a macaque Utah-array dataset with no atlas regions.

### Step 1. Choose the sessions.

- **Question:** which sessions will be compared, decided before seeing any region
  result?
- **Why this answers it:** the region test treats each session as one observation.
  Choosing sessions after seeing results would bias it.
- **Analysis:** an IBL set saved on the homepage, or a list of NWB files with their
  layout. The run records each session's file hash and the trial filter.
- **Null and correction:** none. Choosing makes no claim.
- **Needs:** sessions with brain regions. A dataset without them is refused, saying
  why.
- **How to read it:** a region gets a verdict only if it has units in at least 5 of
  the chosen sessions. The panel lists which regions would.
  - **It does not show the sessions are representative:** probes sample regions
    unevenly.

### Step 2. Label every session's units, then compare regions.

- **Question:** does a region hold more, or fewer, [stimulus]-responsive units than its
  sessions would predict?
- **Why this answers it:** every unit gets the step 1 label of Recipe 1, on its own
  session. A region is then compared with the other units recorded alongside it, in
  each session.
- **Analysis:** a region summary, run on the command line, because every session is
  loaded and tested (minutes per session):
  - the responsiveness test at [stimulus onset], as in Recipe 1, step 1, with each
    session's own unit QC;
  - regions at the Beryl level of the Allen atlas.
- **Null and correction:**
  - **Per region:** its labelled units, summed over sessions, against the exact null in
    which region labels are permuted within each session. No seed is needed.
  - **The test:** two-sided, with Benjamini–Hochberg across the regions with a verdict
    (α = 0.05).
  - **The session is the unit of inference:** a session with many responsive units
    can't make a region look enriched just by contributing many units.
- **Needs:** step 1's sessions. A session that can't be labelled is left out, and the
  reason is shown.
- **How to read it:** "more" means the region's units are labelled more often than
  their own sessions' other units. The flatmap shows the excess, and the spread plot
  shows one dot per session.
  - **It compares how often units are labelled, not how strongly they respond.**
  - **It does not show the region causes anything,** nor why its units respond.
  - **Units are sampled where the probes went:** a region's units may come from one
    part of it.
  - **A refused region is not a region without a difference:** it has too few
    sessions to test.

---

**Wording choices to confirm:**
- **Recipe 2, step 3's window** is IBL's choice window, before the first movement. On
  Steinmetz it is the 100 ms before the response (the wheel reaching threshold), from
  its task definition.
- **Recipe 3 uses responsiveness only.** A selective or movement-locked label could be
  offered as a variant later.

---

**As built (step 9a, 2026-10-03).** The recipe files under `configs/recipes/` follow
this wording, with four differences:
- **Recipe 2, step 1, Needs:** "Trials without one ([no-go]) are excluded and counted"
  became "The rest are excluded and counted: [the condition's own excluded text]".
  IBL's says "no-go trials (choice 0)", which doesn't fit inside parentheses.
- **Recipe 2, step 2, Null:** "(zero contrast split by side)" is dropped. The strata
  are named as the task names them ("signed contrast", "contrasts"), the same name the
  test's own null line uses.
- **Recipe 2, step 3, How to read it:** "a linear readout over 100 ms" became "over a
  short window", since the window comes from each task.
- **Recipe 2, step 3, the button:** it starts the run in the Decoding panel, where
  its progress, verdict line and table already appear, rather than in the Recipes
  panel.
