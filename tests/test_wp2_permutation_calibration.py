import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/analysis"))

from calibrate_wp2_permutations import (  # noqa: E402
    mean_pairwise_block_distance,
    observation_target_families,
    permuted_dataset,
    plus_one_tail,
    swap_assignment,
)
from select_wp2_representation import Dataset  # noqa: E402


def test_permutation_lock_is_discovery_only_and_fixes_32_runs():
    lock = json.loads(
        (ROOT / "notes/WP2_PERMUTATION_CALIBRATION_LOCK_2026-07-13.json").read_text()
    )
    assert lock["permutation_indices"] == list(range(32))
    assert lock["claim_authorized"] is False
    assert lock["confirmation_authorized"] is False


def test_assignment_is_deterministic_mixed_and_index_specific():
    families = [f"family-{index}" for index in range(16)]
    first = swap_assignment(families, 2459917311, 0)
    assert first == swap_assignment(families, 2459917311, 0)
    assert 0 < len(first) < 16
    assert first != swap_assignment(families, 2459917311, 1)


def test_permutation_only_exchanges_target_arms_within_selected_family():
    rows = [
        {"dataset": "wp3", "cell": "observation", "family_id": "f1",
         "arm_id": "current_actual"},
        {"dataset": "wp3", "cell": "observation", "family_id": "f1",
         "arm_id": "archived_actual"},
        {"dataset": "wp3", "cell": "observation", "family_id": "f2",
         "arm_id": "current_actual"},
        {"dataset": "wp1", "cell": "observation", "family_id": "f1",
         "arm_id": "current_actual"},
    ]
    data = Dataset(None, None, rows)
    changed = permuted_dataset(data, {"f1"})
    assert [row["arm_id"] for row in changed.rows] == [
        "archived_actual", "current_actual", "current_actual", "current_actual"
    ]
    assert [row["arm_id"] for row in data.rows] == [
        "current_actual", "archived_actual", "current_actual", "current_actual"
    ]


def test_observation_target_families_requires_exactly_sixteen():
    rows = [
        {"dataset": "wp3", "cell": "observation", "family_id": f"f{index}",
         "arm_id": arm}
        for index in range(16)
        for arm in ("current_actual", "archived_actual")
    ]
    assert observation_target_families(rows) == sorted(f"f{index}" for index in range(16))


def test_plus_one_tail_has_fixed_monte_carlo_resolution():
    values = list(range(32))
    assert plus_one_tail(values, 32, "upper") == 1 / 33
    assert plus_one_tail(values, -1, "lower") == 1 / 33


def test_fold_convergence_uses_all_six_block_pairs():
    sites = [{"block": block, "role": "prompt_final"} for block in (28, 15, 2, 7)]
    assert mean_pairwise_block_distance(sites) == 86 / 6
