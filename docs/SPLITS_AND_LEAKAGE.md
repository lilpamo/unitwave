# Splits and leakage

The single most likely way this project produces a wrong result.

## Why neural decoding leaks so easily

Neural and behavioural time series are strongly autocorrelated on the timescale
of hundreds of milliseconds to seconds. Two windows 200 ms apart are nearly the
same sample. A random time-point split therefore puts near-duplicates of each
test window in the training set, and decoding accuracy inflates dramatically —
often from "modest" to "suspiciously excellent."

Four further routes, in rough order of how often they catch people:

1. **Trial structure.** IBL trials have fixed phase ordering. A model given
   enough temporal context can infer trial phase, and from phase infer the
   likely behaviour, with no neural information. `null_trialstruct` exists to
   catch this.
2. **Normalization across the split.** Z-scoring a whole session before
   splitting leaks test statistics into training.
3. **Session identity as a shortcut.** With session embeddings, a multi-session
   model can memorize per-session behavioural marginals. Held-out-session
   evaluation catches it; held-out-time-within-session does not.
4. **Hyperparameter selection on the test split.** Selecting bin width, filter
   parameters, or QC thresholds by test performance is leakage through you.
   Freeze them in Phase 2 and record them as ADRs.

## Split types

All produced by `splits/registry.py`, serialized as JSON with a hash covering the
split definition, the session manifest version, and `PREPROC_VERSION`.

| Name | Train | Test | Answers |
|---|---|---|---|
| `within_session` | early blocks of session S | late blocks of S, gap ≥ context | Ceiling. How much information is there at all? |
| `held_out_session` | sessions of animal A except S | session S | Does it transfer across recordings of the same animal? |
| `held_out_animal` | animals ≠ A | all sessions of animal A | **The headline test.** |
| `held_out_lab` | labs ≠ L | all sessions of lab L | Rig and pipeline shift. Few groups — report humbly. |
| `held_out_region` | sessions whose insertions miss region R | sessions covering R | Anatomical generalization. Definition below. |
| `held_out_config` | standard probe configs | non-standard insertions / unit counts | Configuration shift. |

**Held-out region needs care.** Insertions cover multiple regions, so "sessions
containing R" is not cleanly separable. Use the stricter definition: a session is
in the test set if ≥20% of its QC-passing units are in R, and in the training set
only if it contains **zero** units in R. Sessions in neither bucket are dropped.
This costs data and is the only honest version.

## Rules, enforced by tests

**Blocking.** Temporal splits use contiguous blocks, never interleaved folds.

**Gap.** Train and test blocks are separated by a gap of at least
`max(context_length, 2 s)`. Behavioural autocorrelation outlives the neural
context window.

**Trial-level targets split on trials.** Choice and block are per-trial, so a
trial must lie wholly on one side. Splitting mid-trial duplicates the label
across the boundary.

**Calibration split is disjoint at session level.** Conformal prediction and
temperature scaling require a calibration set independent of training. For
cross-animal evaluation it must also be independent of the *test animals*, or
your coverage guarantee is circular. This means a three-way animal-level
partition: train animals / calibration animals / test animals.

**Normalization statistics travel with the split file.** `preprocess/normalize.py`
takes a split object, not a session. There is no API that lets you normalize
without declaring a split.

**Exchangeability is broken across animals.** Split conformal prediction assumes
exchangeability between calibration and test. Across animals it does not hold.
Report coverage, expect it to miss nominal, and present the miss as a result.
Reporting conformal coverage across animals as if the guarantee held would be
the most serious error available to this project.

## The guard API

Every training and evaluation entry point begins with:

```python
from unitwave.splits.guards import assert_split_valid

split = load_split(cfg.split_path)
assert_split_valid(split, context_bins=cfg.context_bins)
```

`assert_split_valid` checks: no shared bin indices; gap satisfied; disjoint
subject sets for animal-level splits; disjoint session sets for session-level
splits; calibration disjoint from both; trial integrity; no empty partition;
manifest and preproc versions match those recorded in the split file.

It raises. It does not warn.

## Smell tests to run on every new result

- Does it beat `null_shuffle`? If not, there is no signal.
- Does it beat `null_trialstruct`? If not, you decoded the task, not the brain.
- Is held-out-animal performance suspiciously close to within-session? Suspect
  leakage before celebrating transfer.
- Does per-session performance correlate with unit count more strongly than with
  anything else? Then you measured yield, not decoding.
- Did performance improve right after you touched `preprocess/` or `splits/`?
  Assume leakage and bisect.
