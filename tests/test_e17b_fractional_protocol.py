"""Pure acceptance tests for the frozen Gate-0C fractional null protocol."""

import contextlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.e17b_null_audit import (  # noqa: E402
    GATE0C_NULL_PROTOCOL, NullProtocolError, analyze_fractional_protocol,
    derive_child_seeds, isotropic_unit_direction, plus_one_rank_p,
    run_fractional_protocol, signed_full_ablation_effect,
    validate_gate0c_binding, validate_null_protocol,
)


def frozen_validate(n=39):
    return validate_null_protocol(
        "accepted", n, GATE0C_NULL_PROTOCOL["master_seed"],
        GATE0C_NULL_PROTOCOL["fractions"], GATE0C_NULL_PROTOCOL["readouts"],
        GATE0C_NULL_PROTOCOL["cells"])


def test_seed_stream_is_deterministic_unique_and_prefix_invariant():
    seed = GATE0C_NULL_PROTOCOL["master_seed"]
    a = derive_child_seeds(seed, 39)
    b = derive_child_seeds(seed, 100)
    assert a == b[:39]
    assert len(a) == len(set(a)) == 39


def test_random_direction_is_float32_unit_and_hash_stable():
    child = derive_child_seeds(GATE0C_NULL_PROTOCOL["master_seed"], 1)[0]
    a, ra = isotropic_unit_direction(child, 64)
    b, rb = isotropic_unit_direction(child, 64)
    assert a.dtype == np.float32
    assert np.isclose(np.linalg.norm(a), 1.0, atol=1e-6)
    assert np.array_equal(a, b)
    assert ra["unit_direction_sha256"] == rb["unit_direction_sha256"]


def test_accepted_protocol_requires_exact_39_and_rejects_omission_or_smoke():
    assert frozen_validate()["smoke"] is False
    with pytest.raises(NullProtocolError, match="explicit"):
        frozen_validate(None)
    with pytest.raises(NullProtocolError, match="exactly 39"):
        frozen_validate(3)
    smoke = validate_null_protocol(
        "exploratory", None, 1, (0.0, 1.0), ("raw_ab_dual",),
        ("M_confirm", "T_confirm"))
    assert smoke["smoke"] is True and "NON-EVIDENTIAL SMOKE" in smoke["evidence_note"]


def test_signed_statistic_and_plus_one_p():
    assert signed_full_ablation_effect([1, 2], [0, 1]) == 1.0
    assert plus_one_rank_p(1.0, [0.0] * 39) == 1 / 40
    assert plus_one_rank_p(0.0, [0.0] * 39) == 1.0


def test_orchestrator_scores_same_cells_and_readouts_for_every_direction():
    state = {"label": "baseline", "fraction": 0.0}
    calls = []

    def score(readout, cell):
        calls.append((state["label"], state["fraction"], readout, cell))
        return {"scores": [state["fraction"]]}

    @contextlib.contextmanager
    def ablator(label, fraction):
        old = dict(state)
        state.update(label=str(label), fraction=fraction)
        try:
            yield
        finally:
            state.update(old)

    result = run_fractional_protocol(
        score_fn=score, ablator_factory=ablator,
        cells=("M_confirm", "T_confirm"),
        readouts=("raw_ab_dual", "continuation"),
        fractions=(0.0, 0.5, 1.0),
        random_directions=[("r0", np.ones(2), {"seed": 1})])
    assert set(result["random"]["r0"]["scores"]) == {
        "raw_ab_dual", "continuation"}
    assert all(set(result["random"]["r0"]["scores"][r]) == {
        "M_confirm", "T_confirm"} for r in ("raw_ab_dual", "continuation"))
    assert len(calls) == 4 * (1 + 2 + 1)  # baseline + nonzero fractions + null


def synthetic_scores():
    readouts = GATE0C_NULL_PROTOCOL["readouts"]
    cells = GATE0C_NULL_PROTOCOL["cells"]
    base = {r: {c: {"scores": [1.0] * 4} for c in cells} for r in readouts}
    targets = {}
    for fraction in GATE0C_NULL_PROTOCOL["fractions"]:
        targets[str(fraction)] = {
            r: {
                "M_confirm": {"scores": [1.0 - fraction] * 4},
                "T_confirm": {"scores": [1.0 - 0.2 * fraction] * 4},
            } for r in readouts}
    random = {
        f"r{i}": {"scores": {
            r: {c: {"scores": [0.9] * 4} for c in cells} for r in readouts}}
        for i in range(39)}
    return {"baseline": base, "target_fractions": targets, "random": random}


def test_analysis_represents_family_lofo_rank_dose_and_selectivity_gates():
    families = {"M_confirm": ["a", "a", "b", "b"],
                "T_confirm": ["c", "c", "d", "d"]}
    report = analyze_fractional_protocol(
        synthetic_scores(), families, GATE0C_NULL_PROTOCOL["readouts"],
        n_boot=200)
    assert report["all_required_gates_pass"] is True
    for readout in GATE0C_NULL_PROTOCOL["readouts"]:
        cell = report["readouts"][readout]["M_confirm"]
        assert cell["plus_one_p_one_sided"] == 1 / 40
        assert all(cell["gates"].values())
        assert report["readouts"][readout]["descriptive_selectivity"][
            "abs_T_over_abs_M"] == pytest.approx(0.2)


def test_only_raw_is_primary_and_dose_tolerance_is_three_percent_of_full():
    scores = synthetic_scores()
    # Chat sign failure is reported but cannot replace or veto the raw gate.
    for key in scores["target_fractions"]:
        fraction = float(key)
        scores["target_fractions"][key]["chat_ab_dual"]["M_confirm"] = {
            "scores": [1.0 + fraction] * 4}
    # A 0.01 downward step in signed effect is within 3% of full Z=1.
    scores["target_fractions"]["0.25"]["raw_ab_dual"]["M_confirm"] = {
        "scores": [0.8] * 4}
    scores["target_fractions"]["0.5"]["raw_ab_dual"]["M_confirm"] = {
        "scores": [0.81] * 4}
    families = {"M_confirm": ["a", "a", "b", "b"],
                "T_confirm": ["c", "c", "d", "d"]}
    report = analyze_fractional_protocol(
        scores, families, GATE0C_NULL_PROTOCOL["readouts"], n_boot=100)
    assert report["all_required_gates_pass"] is True
    assert report["readouts"]["chat_ab_dual"]["format_sign_transfer"] is False

    # Increase the reversal beyond .03 * |Z_full|: raw primary must fail.
    scores["target_fractions"]["0.5"]["raw_ab_dual"]["M_confirm"] = {
        "scores": [0.85] * 4}
    report = analyze_fractional_protocol(
        scores, families, GATE0C_NULL_PROTOCOL["readouts"], n_boot=100)
    assert report["all_required_gates_pass"] is False


def test_gate0c_binding_freezes_model_direction_block_and_cell_bytes():
    report = validate_gate0c_binding(
        protocol_name=GATE0C_NULL_PROTOCOL["name"],
        model=GATE0C_NULL_PROTOCOL["model"],
        component_set=GATE0C_NULL_PROTOCOL["component_set"],
        direction_path=GATE0C_NULL_PROTOCOL["direction"]["path"],
        block=20, repo_root=ROOT)
    assert report["direction_sha256"] == GATE0C_NULL_PROTOCOL["direction"]["sha256"]
    with pytest.raises(NullProtocolError, match="binding mismatch"):
        validate_gate0c_binding(
            protocol_name=GATE0C_NULL_PROTOCOL["name"],
            model="meta-llama/Llama-3.1-8B-Instruct",
            component_set=GATE0C_NULL_PROTOCOL["component_set"],
            direction_path=GATE0C_NULL_PROTOCOL["direction"]["path"],
            block=20, repo_root=ROOT)
