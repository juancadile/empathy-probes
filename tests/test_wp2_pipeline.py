import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "analysis"))

from extract_gate2_activations import (  # noqa: E402
    candidate_blocks,
    load_records,
    quote_token_index,
    validate_human_gate,
)
from select_wp2_representation import (  # noqa: E402
    choose_candidate,
    family_contrasts,
    source_balanced_folds,
)


def test_gate2_extraction_plan_counts_and_blocks():
    assert candidate_blocks(42) == [13, 19, 25, 20]
    dev = load_records("dev")
    confirm = load_records("confirm")
    assert len(dev) == len(confirm) == 1568
    assert sum(row["readout_role"] == "quote_boundary" for row in dev) == 128
    assert all("confirm" not in row["partition"].lower() for row in dev)
    assert all("confirm" in row["partition"].lower() for row in confirm)


def test_quote_boundary_uses_final_quote_character():
    prompt = 'Context. The exact message is: "need help." End.'
    quote = "need help."
    offsets = [(0, 7), (7, 31), (31, 36), (36, 41), (41, 42), (42, 47)]
    assert quote_token_index(offsets, quote, prompt) == 4


def test_real_extraction_gate_fails_closed(tmp_path):
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({
        "schema": "empathy-action-probes/gate2-v2-human-gate/1",
        "gate2_v2": {"passed_human_gate": False},
    }))
    with pytest.raises(ValueError, match="not open"):
        validate_human_gate(gate)


def test_family_contrasts_average_variants_within_family():
    rows, vectors = [], []
    for family, offset in (("f1", 0.0), ("f2", 10.0)):
        for variant in ("v0", "v1"):
            rows.extend([
                {"family_id": family, "variant_id": variant, "arm_id": "pos"},
                {"family_id": family, "variant_id": variant, "arm_id": "neg"},
            ])
            vectors.extend([[offset + 3.0, 1.0], [offset + 1.0, 1.0]])
    contrasts = family_contrasts(np.asarray(vectors), rows, "pos", "neg")
    np.testing.assert_allclose(contrasts, [[2.0, 0.0], [2.0, 0.0]])


def test_source_balanced_folds_never_split_family():
    rows = []
    for source in ("a", "b"):
        for family_index in range(8):
            for variant in range(2):
                rows.append({
                    "family_id": f"{source}{family_index}",
                    "source": source,
                    "variant_id": f"v{variant}",
                })
    folds = source_balanced_folds(rows, 4, 123)
    assert set(folds.values()) == {0, 1, 2, 3}
    for source in ("a", "b"):
        assert sorted(folds[f"{source}{i}"] for i in range(8)) == [0, 0, 1, 1, 2, 2, 3, 3]


def test_lexicographic_selection_prefers_quiet_then_low_dimension():
    configs = [
        {"kind": "a", "block": 19, "role": "prompt_final", "dim": 4, "alpha": 1.0},
        {"kind": "b", "block": 20, "role": "prompt_final", "dim": 1, "alpha": None},
        {"kind": "c", "block": 19, "role": "prompt_final", "dim": 2, "alpha": 0.1},
    ]
    quiet = {name: 0.55 for name in (
        "T_new", "D_new", "P_new", "G_new", "B_new", "Spos_new", "O_new", "Ctext_new"
    )}
    scored = [
        {"config": configs[0], "valid": True,
         "metrics": {"target_auroc": 0.90, "quiet_auroc": {**quiet, "T_new": 0.70}}},
        {"config": configs[1], "valid": True,
         "metrics": {"target_auroc": 0.885, "quiet_auroc": quiet}},
        {"config": configs[2], "valid": True,
         "metrics": {"target_auroc": 0.885, "quiet_auroc": quiet}},
    ]
    assert choose_candidate(scored, [13, 19, 25, 20])["config"] == configs[1]
