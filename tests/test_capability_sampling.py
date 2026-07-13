"""Deterministic subject-stratified MMLU sampling + parser status
(capability_eval, Integrity Repair A). CPU-only; no datasets download."""

import sys
from collections import Counter
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from capability_eval import (  # noqa: E402
    GATE0C_CAPABILITY_PROTOCOL_NAME, LETTERS, exact_text_slice,
    forced_choice_loglikelihoods, parse_letter, stratified_allocation,
    stratified_sample_indices, subject_cluster_bootstrap_delta,
    summarize_generation_details, validate_gate0c_capability_binding,
    validate_gate0c_capability_protocol,
    verify_dataset_fingerprint,
)
from src.utils.evidence_run import EvidenceRunError  # noqa: E402


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
    # audited parser (QA Q5): uppercase-only, no permissive first-letter scan
    ("b lowercase only", (None, "unparsed")),
    ("no letter here", (None, "unparsed")),
    ("", (None, "unparsed")),
])
def test_parse_letter_status(completion, expected):
    pred, status, _candidates = parse_letter(completion)
    assert (pred, status) == expected
    if status == "parsed":
        assert LETTERS[pred] in completion


@pytest.mark.parametrize("completion,candidates", [
    ("Answer: A or C", ("A", "C")),
    ("Answer: C. Answer: C.", ("C",)),
    ("Answer: A. Final answer: D.", ("A", "D")),
    ("A or C", ("A", "C")),
    ("C) first\nC) repeated", ("C",)),
])
def test_parse_letter_rejects_multiple_or_conflicting_evidence(completion,
                                                                candidates):
    pred, status, observed = parse_letter(completion)
    assert pred is None
    assert status == "ambiguous"
    assert observed == candidates


@pytest.mark.parametrize("completion", ["C", "(C)", "Answer: C",
                                         "The answer is C."])
def test_parse_letter_accepts_single_explicit_candidate(completion):
    pred, status, candidates = parse_letter(completion)
    assert (pred, status, candidates) == (2, "parsed", ("C",))


class _StableTokenizer:
    def __call__(self, text, add_special_tokens=False):
        table = {"P": [1], "PA": [1, 2, 3]}
        return {"input_ids": table[text]}

    def convert_ids_to_tokens(self, ids):
        return [f"t{i}" for i in ids]


class _BoundaryMergingTokenizer(_StableTokenizer):
    def __call__(self, text, add_special_tokens=False):
        table = {"P": [1, 2], "PA": [1, 9]}
        return {"input_ids": table[text]}


class _FixedModel:
    def __call__(self, input_ids, use_cache=False):
        seq = input_ids.shape[1]
        logits = torch.zeros((1, seq, 16), dtype=torch.float32)
        return type("Output", (), {"logits": logits})()


def test_joint_candidate_tokenization_persists_multitoken_suffix():
    result = forced_choice_loglikelihoods(
        _FixedModel(), _StableTokenizer(), "P", "cpu", labels=("A",))
    assert result["A"]["token_ids"] == [2, 3]
    assert result["A"]["full_token_ids"] == [1, 2, 3]
    assert len(result["A"]["per_token_logprobs"]) == 2


def test_candidate_tokenization_fails_closed_on_boundary_merge():
    with pytest.raises(ValueError, match="changes the prompt"):
        forced_choice_loglikelihoods(
            _FixedModel(), _BoundaryMergingTokenizer(), "P", "cpu",
            labels=("A",))


def test_exact_wikitext_slice_and_short_corpus_abort():
    rows = ["a" * 210_000, "b" * 210_000]
    observed = exact_text_slice(rows, char_start=200_000, char_end=400_000)
    joined = "\n".join(rows)
    assert observed == joined[200_000:400_000]
    with pytest.raises(ValueError, match="requires at least 400000"):
        exact_text_slice(["short"], char_start=200_000, char_end=400_000)


def test_accepted_dataset_fingerprint_is_exact():
    verify_dataset_fingerprint("mmlu", "fp", "fp", accepted=True)
    with pytest.raises(EvidenceRunError, match="requires --expected"):
        verify_dataset_fingerprint("mmlu", "fp", None, accepted=True)
    with pytest.raises(EvidenceRunError, match="mismatch"):
        verify_dataset_fingerprint("mmlu", "other", "fp", accepted=True)


def test_frozen_gate0c_capability_protocol():
    report = validate_gate0c_capability_protocol(
        protocol=GATE0C_CAPABILITY_PROTOCOL_NAME,
        model="google/gemma-2-9b-it", n_mmlu=800, seed=946441184,
        wikitext_char_start=200_000, wikitext_char_end=400_000,
        skip_generation_readout=False)
    assert report["conditions"] == (
        "baseline", "positive_writers_k2", "suppressors_k4", "targeted_k6")
    with pytest.raises(EvidenceRunError, match="protocol mismatch"):
        validate_gate0c_capability_protocol(
            protocol=GATE0C_CAPABILITY_PROTOCOL_NAME,
            model="google/gemma-2-9b-it", n_mmlu=400, seed=946441184,
            wikitext_char_start=200_000, wikitext_char_end=400_000,
            skip_generation_readout=False)


def test_frozen_capability_binding_uses_current_direction_and_sets():
    report = validate_gate0c_capability_binding(
        component_set="gemma2_9b_it_resid_2026-07-12",
        direction_path=("results/controlled_directions_gemma2_9b_it/"
                        "direction_M_resid_block20.npy"),
        resolved_sets={
            "positive_writers": "L19MLP,L20MLP",
            "suppressors": "L18H13,L20H10,L19H12,L17H7"},
        repo_root=ROOT)
    assert report["direction_sha256"].startswith("1b6d692e")
    with pytest.raises(EvidenceRunError, match="binding mismatch"):
        validate_gate0c_capability_binding(
            component_set="gemma2_9b_it_resid_2026-07-12",
            direction_path=("results/controlled_directions_gemma2_9b_it/"
                            "direction_M_resid_block20.npy"),
            resolved_sets={"positive_writers": "L1MLP",
                           "suppressors": "L18H13"}, repo_root=ROOT)


def test_generation_summary_separates_all_item_and_parsed_only_policies():
    details = [
        {"parser_status": "parsed", "correct": True},
        {"parser_status": "parsed", "correct": False},
        {"parser_status": "ambiguous", "correct": None},
        {"parser_status": "unparsed", "correct": None},
    ]
    report = summarize_generation_details(details)
    assert report["parse_rate"] == 0.5
    assert report["all_items"]["accuracy"] == 0.25
    assert report["parsed_only"]["accuracy"] == 0.5


def test_subject_cluster_bootstrap_is_deterministic_and_reports_signs():
    args = ([1, 0, 1, 0], [1, 1, 0, 0], ["a", "a", "b", "b"])
    a = subject_cluster_bootstrap_delta(*args, seed=7, n_boot=200)
    b = subject_cluster_bootstrap_delta(*args, seed=7, n_boot=200)
    assert a == b
    assert a["per_subject"]["a"]["sign"] == "positive"
    assert a["per_subject"]["b"]["sign"] == "negative"
    assert "NOT an equivalence/no-cost result" in a["interpretation"]
