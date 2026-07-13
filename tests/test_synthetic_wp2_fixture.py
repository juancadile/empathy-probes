import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "analysis"))

import build_synthetic_wp2_fixture as fixture  # noqa: E402
from select_wp2_representation import Dataset, control_rows, target_rows  # noqa: E402


def test_synthetic_fixture_matches_real_development_shape(tmp_path):
    out = tmp_path / "fixture"
    plan = fixture.build(out)
    data = Dataset.load(out)
    assert plan["target_model_loaded"] is False
    assert data.activations.shape == (1568, 4, fixture.HIDDEN_SIZE)
    assert data.blocks.tolist() == [13, 19, 25, 20]


def test_synthetic_target_and_nuisances_are_separated(tmp_path):
    out = tmp_path / "fixture"
    fixture.build(out)
    data = Dataset.load(out)
    target, rows = target_rows(data, 19, "prompt_final")
    current = np.mean([x[0] for x, row in zip(target, rows) if row["arm_id"] == "current_actual"])
    archived = np.mean([x[0] for x, row in zip(target, rows) if row["arm_id"] == "archived_actual"])
    assert current - archived > 2.0
    task, task_rows = control_rows(data, 19, "T_new")
    by_arm = {
        arm: np.mean([x[0] for x, row in zip(task, task_rows) if row["arm_id"] == arm])
        for arm in ("interrupt", "persist")
    }
    assert abs(by_arm["interrupt"] - by_arm["persist"]) < 0.1


def test_synthetic_builder_refuses_overwrite(tmp_path):
    out = tmp_path / "fixture"
    fixture.build(out)
    try:
        fixture.build(out)
    except ValueError as exc:
        assert "refusing to overwrite" in str(exc)
    else:
        raise AssertionError("synthetic fixture was overwritten")
