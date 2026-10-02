"""Step 3: d23a44ef gives identical numbers and captions before and after task
definitions. The golden record was made with the code at 7490130, before them
(tests/golden/task_golden.py).

Numbers and captions must be identical. The page's JSON documents may gain fields
the task definitions add (ADDED below, by name), and nothing else: every field the
record has must be there, unchanged."""

import json

import numpy as np
import pytest
from golden.task_golden import OUT, collect, studio

# Fields step 3 adds to the page's JSON: a trial filter's kind, and the single-trial
# header's task, per-condition levels and flags.
ADDED = {"kind", "task", "conditions", "flags"}


@pytest.fixture(scope="module")
def now():
    try:
        return collect(studio())
    except (OSError, ValueError) as e:
        pytest.skip(f"d23a44ef not available: {e}")


def test_every_number_is_identical_to_the_record_made_before_task_definitions(now):
    arrays, _ = now
    gold = np.load(OUT.with_suffix(".npz"))
    assert sorted(arrays) == sorted(gold.files)
    for key in gold.files:
        np.testing.assert_array_equal(arrays[key], gold[key], err_msg=key)


def _json(text):
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, (dict, list)) else None


def _unchanged(gold, now, where: str, added: set) -> None:
    """Every field of gold is in now, equal; fields only in now are collected."""
    if isinstance(gold, dict):
        assert isinstance(now, dict), where
        for key, value in gold.items():
            assert key in now, f"{where}: {key} is gone"
            _unchanged(value, now[key], f"{where}/{key}", added)
        added |= set(now) - set(gold)
    elif isinstance(gold, list):
        assert isinstance(now, list) and len(now) == len(gold), where
        for i, (g, n) in enumerate(zip(gold, now)):
            _unchanged(g, n, f"{where}/{i}", added)
    else:
        assert now == gold, where


def test_every_caption_and_label_is_identical_too(now):
    _, texts = now
    gold = json.loads(OUT.with_suffix(".json").read_text())
    assert sorted(texts) == sorted(gold)
    added: set = set()
    for key, value in gold.items():
        old, new = _json(value), _json(texts[key])
        if old is None:
            assert texts[key] == value, key
        else:
            _unchanged(old, new, key, added)
    assert added <= ADDED, f"unexpected new fields: {sorted(added - ADDED)}"
