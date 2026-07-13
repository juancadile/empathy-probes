"""e17b fractional-ablation result schema: per-pair persistence and
backward-compatible summaries (Integrity Repair A). CPU-only, no model."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from e17b_null_audit import (  # noqa: E402
    EVAL_SETS, cell_provenance, clustered_delta, validate_e17b_artifact,
)
from weight_orthogonalization import load_pairs  # noqa: E402


COND = [1.0, 2.0, 3.0, 4.0]
BASE = [0.5, 1.0, 1.5, 2.0]
FAMS = ["a", "a", "b", "b"]


def test_clustered_delta_backward_compatible_summary():
    record = clustered_delta(COND, BASE, FAMS, seed=0)
    assert set(record) == {"mean", "ci95"}  # legacy shape untouched
    assert record["mean"] == 1.25
    assert record["ci95"][0] <= record["mean"] <= record["ci95"][1]


def test_clustered_delta_detail_reconstructs_summary():
    record = clustered_delta(COND, BASE, FAMS, seed=0, detail=True)
    assert record["per_pair_scores"] == COND
    deltas = np.asarray(record["per_pair_delta"])
    assert np.allclose(deltas, np.asarray(COND) - np.asarray(BASE))
    # every reported number is recomputable from the same record
    assert np.isclose(record["mean"], deltas.mean())


def test_clustered_delta_detail_and_summary_agree():
    plain = clustered_delta(COND, BASE, FAMS, seed=0)
    detailed = clustered_delta(COND, BASE, FAMS, seed=0, detail=True)
    assert plain["mean"] == detailed["mean"]
    assert plain["ci95"] == detailed["ci95"]  # same bootstrap stream


def test_cell_provenance_is_self_contained():
    pair_sets = {n: load_pairs(ROOT / p) for n, p in EVAL_SETS.items()}
    fams = {n: [p["scenario_id"] for p in pairs]
            for n, pairs in pair_sets.items()}
    prov = cell_provenance(pair_sets, fams,
                           eval_sets={n: ROOT / p for n, p in EVAL_SETS.items()})
    for name in ("M_confirm", "T_confirm"):
        assert prov[name]["sha256"]
        assert prov[name]["n_rows"] == len(pair_sets[name])
        assert prov[name]["families"] == fams[name]
    # the repaired T_confirm artifact: 40 rows, 40 unique families x variants
    assert prov["T_confirm"]["n_rows"] == 40
    assert len(set(prov["T_confirm"]["families"])) == 5


def _accepted_artifact(verdict):
    return {
        "model": "google/gemma-2-9b-it",
        "block": 20,
        "run_contract": {"run_mode": "accepted"},
        "component_sets": {},
        "direction_file": {},
        "cells": {},
        "provenance": {},
        "test3_fractional_ablation": {
            "gate0c_analysis": {"all_required_gates_pass": verdict}},
    }


@pytest.mark.parametrize("verdict", [True, False])
def test_accepted_artifact_persists_both_scientific_outcomes(verdict):
    # Acceptance is an integrity/completeness property, not a favorable-result
    # filter. Failed preregistered gates must remain publishable artifacts.
    validate_e17b_artifact(_accepted_artifact(verdict))


def test_accepted_artifact_requires_explicit_gate_verdict():
    payload = _accepted_artifact(False)
    del payload["test3_fractional_ablation"]["gate0c_analysis"]
    with pytest.raises(ValueError, match="lacks Gate-0C analysis"):
        validate_e17b_artifact(payload)
