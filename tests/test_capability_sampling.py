"""Deterministic subject-stratified MMLU sampling + parser status
(capability_eval, Integrity Repair A). CPU-only; no datasets download."""

import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from capability_eval import (  # noqa: E402
    LETTERS, parse_letter, stratified_allocation, stratified_sample_indices,
)


def test_allocation_exact_n_and_proportional():
    counts = {"algebra": 500, "law": 300, "virology": 200}
    alloc = stratified_allocation(counts, 100)
    assert sum(alloc.values()) == 100
    assert alloc == {"algebra": 50, "law": 30, "virology": 20}


def test_allocation_largest_remainder_and_caps():
    counts = {"a": 3, "b": 3, "c": 3}
    alloc = stratified_allocation(counts, 8)
    assert sum(alloc.values()) == 8
    assert all(alloc[s] <= counts[s] for s in counts)
    # tiny subjects are never over-allocated even when others are exhausted
    counts = {"big": 100, "tiny": 1}
    alloc = stratified_allocation(counts, 100)
    assert alloc["tiny"] <= 1 and sum(alloc.values()) == 100


def test_allocation_rejects_oversampling():
    with pytest.raises(ValueError, match="cannot sample"):
        stratified_allocation({"a": 2}, 3)


def test_stratified_sample_deterministic_and_stratified():
    subjects = (["algebra"] * 50 + ["law"] * 30 + ["virology"] * 20) * 4
    idx_a, alloc_a = stratified_sample_indices(subjects, 40, seed=42)
    idx_b, alloc_b = stratified_sample_indices(subjects, 40, seed=42)
    assert idx_a == idx_b and alloc_a == alloc_b  # deterministic
    assert len(idx_a) == len(set(idx_a)) == 40
    by_subject = Counter(subjects[i] for i in idx_a)
    assert by_subject == {"algebra": 20, "law": 12, "virology": 8}

    idx_c, _ = stratified_sample_indices(subjects, 40, seed=43)
    assert idx_c != idx_a  # seed actually matters


def test_stratified_sample_indices_valid():
    subjects = ["s1"] * 5 + ["s2"] * 5
    indices, _ = stratified_sample_indices(subjects, 6, seed=0)
    assert all(0 <= i < len(subjects) for i in indices)
    assert indices == sorted(indices)


@pytest.mark.parametrize("completion,expected", [
    ("A", (0, "parsed")),
    (" B) is right", (1, "parsed")),
    ("The answer is C", (2, "parsed")),
    # uppercase-only search, identical to the accepted run's parser
    ("b lowercase only", (-1, "unparsed")),
    ("no letter here", (-1, "unparsed")),
    ("", (-1, "unparsed")),
])
def test_parse_letter_status(completion, expected):
    pred, status = parse_letter(completion)
    assert (pred, status) == expected
    if status == "parsed":
        assert LETTERS[pred] in completion
