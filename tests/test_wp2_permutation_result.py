import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/wp2_permutation_calibration_20260713"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_all_locked_permutations_are_preserved_and_hash_bound():
    manifest = json.loads((RESULT / "MANIFEST.json").read_text())
    files = sorted(RESULT.glob("permutation_[0-9][0-9].json"))
    assert len(files) == manifest["raw_permutation_count"] == 32
    combined = hashlib.sha256()
    for index, path in enumerate(files):
        item = json.loads(path.read_text())
        assert item["permutation_index"] == index
        assert item["claim_authorized"] is False
        assert item["confirmation_authorized"] is False
        combined.update(path.read_bytes())
    assert combined.hexdigest() == manifest["raw_permutations_concatenated_sha256"]
    assert sha256(RESULT / "permutation_calibration.json") == manifest[
        "canonical_aggregate_sha256"
    ]


def test_canonical_result_uses_fold_convergence_not_confounded_pass_inference():
    result = json.loads((RESULT / "permutation_calibration.json").read_text())
    assert result["claim_authorized"] is False
    assert result["confirmation_authorized"] is False
    assert result["permutation_count"] == 32
    assert result["observed"]["mean_pairwise_block_distance"] == 86 / 6
    primary = result["null"]["primary_fold_convergence"]
    assert primary["statistic"] == "mean_pairwise_absolute_block_distance"
    assert primary["median"] == 12.083333333333332
    assert primary["observed_lower_tail_p_plus_one"] == 18 / 33
    pass_counts = result["null"]["apparent_pass_count"]
    assert pass_counts["inferential_use"] == (
        "none: target permutation destroys decodability"
    )
    assert "observed_upper_tail_p_plus_one" not in pass_counts


def test_observed_selection_is_no_more_site_stable_than_null():
    result = json.loads((RESULT / "permutation_calibration.json").read_text())
    assert result["observed"]["unique_site_count"] == 4
    assert result["null"]["unique_site_count"]["values"].count(4) == 28
    assert result["null"]["unique_site_count"]["observed_lower_tail_p_plus_one"] == 1.0
    assert result["observed"]["block_span"] == 26
    assert result["null"]["block_span"]["median"] == 24.0
