import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/wp2_broadened_dev_allblocks_20260713"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_result():
    return json.loads((RESULT / "selection/selection.json").read_text())


def passes_development_screen(item):
    metrics = item["metrics"]
    return (
        item["valid"]
        and metrics["target_auroc"] >= 0.75
        and all(abs(value - 0.5) <= 0.10 for value in metrics["quiet_auroc"].values())
    )


def test_broadened_result_is_hash_bound_and_discovery_only():
    manifest = json.loads((RESULT / "MANIFEST.json").read_text())
    for relative, expected in manifest["local_artifacts"].items():
        assert sha256(RESULT / relative) == expected
    result = load_result()
    assert result["claim_authorized"] is False
    assert result["confirmation_authorized"] is False
    assert result["candidate_blocks"] == list(range(42))
    assert result["candidate_count"] == 1764
    assert result["screen"]["outcome"] == "no-candidate"
    assert result["screen"]["development_screen_passed"] is False


def test_nested_selection_fails_and_does_not_converge_on_a_site():
    result = load_result()
    quiet = result["nested_outer"]["metrics"]["quiet_auroc"]
    failed = {name for name, value in quiet.items() if abs(value - 0.5) > 0.10}
    assert failed == {"T_new", "B_new", "Spos_new"}
    chosen = [fold["chosen_config"] for fold in result["nested_outer"]["folds"]]
    assert [item["block"] for item in chosen] == [28, 15, 2, 7]
    assert [item["role"] for item in chosen] == [
        "quote_boundary", "prompt_final", "quote_boundary", "prompt_final"
    ]
    assert len({(item["block"], item["role"]) for item in chosen}) == 4


def test_full_development_passes_do_not_override_nested_selection():
    result = load_result()
    apparent_passes = [
        item for item in result["full_development_cv"] if passes_development_screen(item)
    ]
    assert len(apparent_passes) == 62
    assert result["screen"]["best_full_development_diagnostic"] == {
        "kind": "residualized",
        "block": 3,
        "role": "quote_boundary",
        "dim": 1,
        "nuisance_dim": 8,
        "alpha": None,
    }
    assert result["screen"]["development_screen_passed"] is False
