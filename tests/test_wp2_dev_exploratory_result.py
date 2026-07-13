import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/wp2_dev_exploratory_20260713"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exploratory_result_is_bound_and_stops_before_confirmation():
    manifest = json.loads((RESULT / "MANIFEST.json").read_text())
    for relative, expected in manifest["artifacts"].items():
        assert sha256(RESULT / relative) == expected
    selection = json.loads((RESULT / "selection/selection.json").read_text())
    plan = json.loads((RESULT / "activations/extraction_plan.json").read_text())
    assert plan["phase"] == "dev"
    assert plan["authorization_mode"] == "exploratory_dev"
    assert plan["claim_authorized"] is False
    assert plan["human_gate"] is None
    assert plan["selection_lock"] is None
    assert selection["selection"]["outcome"] == "no-representation"
    assert selection["selection"]["chosen_config"] is None
    assert selection["selection"]["frozen_representation"] is None


def test_exploratory_gate_failure_recomputes_from_frozen_metrics():
    selection = json.loads((RESULT / "selection/selection.json").read_text())["selection"]
    quiet = selection["nested_quiet_auroc"]
    recomputed = (
        selection["nested_target_auroc"] >= 0.75
        and all(abs(value - 0.5) <= 0.10 for value in quiet.values())
    )
    assert recomputed is False
    assert selection["development_gate_passed"] is False
    assert {name for name, value in quiet.items() if abs(value - 0.5) > 0.10} == {
        "B_new", "Spos_new", "O_new", "Ctext_new"
    }
