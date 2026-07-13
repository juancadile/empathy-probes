import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.gate0b_task_control import (
    PROTOCOL, analyze_gate0b, family_summary, validate_artifact,
)


def _scores(m_writer=-3.0, t_writer=-0.5):
    cells = PROTOCOL["cells"]
    out = {}
    for condition in ("baseline", "positive_writers_k2", "suppressors_k4"):
        out[condition] = {}
        for readout in PROTOCOL["readouts"]:
            out[condition][readout] = {}
            for cell, spec in cells.items():
                base = [0.0] * spec["n_rows"]
                delta = 0.0
                if condition == "positive_writers_k2":
                    delta = m_writer if cell == "M_confirm" else t_writer
                elif condition == "suppressors_k4":
                    delta = 1.0
                out[condition][readout][cell] = {
                    "scores": [x + delta for x in base]}
    return out


def _families():
    return {
        name: [f"{name}_{i // 8}" for i in range(spec["n_rows"])]
        for name, spec in PROTOCOL["cells"].items()
    }


def test_family_summary_uses_equal_family_means():
    result = family_summary([2, 2, 4, 4], [1, 1, 1, 1],
                            ["a", "a", "b", "b"], seed=3, n_boot=100)
    assert result["mean"] == 2.0
    assert result["family_effects"] == {"a": 1.0, "b": 3.0}
    assert result["per_pair_delta"] == [1.0, 1.0, 3.0, 3.0]


@pytest.mark.parametrize("m,t,verdict", [(-3.0, -0.5, True),
                                         (-3.0, -1.0, False)])
def test_frozen_writer_ratio_gate(m, t, verdict):
    report = analyze_gate0b(_scores(m, t), _families())
    assert report["all_required_gates_pass"] is verdict
    for readout in PROTOCOL["primary_readouts"]:
        assert report["writer_selectivity"][readout]["pass"] is verdict


def test_validator_preserves_pass_and_failure():
    scores = _scores()
    base = {
        "schema": "empathy-action-probes/gate0b-task-control/1",
        "protocol": {"name": PROTOCOL["name"]},
        "run_contract": {"run_mode": "accepted"},
        "component_sets": {}, "direction": {}, "cells": {},
        "scores": scores, "provenance": {},
    }
    for verdict in (True, False):
        artifact = copy.deepcopy(base)
        artifact["analysis"] = {"all_required_gates_pass": verdict}
        validate_artifact(artifact)


def test_validator_rejects_incomplete_scores():
    scores = _scores()
    scores["baseline"]["continuation"]["M_confirm"]["scores"].pop()
    artifact = {
        "schema": "empathy-action-probes/gate0b-task-control/1",
        "protocol": {"name": PROTOCOL["name"]},
        "run_contract": {"run_mode": "accepted"},
        "component_sets": {}, "direction": {}, "cells": {},
        "scores": scores, "analysis": {"all_required_gates_pass": False},
        "provenance": {},
    }
    with pytest.raises(ValueError, match="incomplete scores"):
        validate_artifact(artifact)
