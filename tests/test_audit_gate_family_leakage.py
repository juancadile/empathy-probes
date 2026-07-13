import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data_generation"))
import audit_gate_family_leakage as leakage  # noqa: E402


def test_length_bound_rejects_impossible_jaccard():
    assert leakage.possible_by_length(90, 100, 0.85)
    assert not leakage.possible_by_length(10, 100, 0.85)


def test_compare_finds_cross_and_within_pool_duplicates():
    phrase = "finish the quarterly report before the client review meeting"
    new = [
        {"family_id": "a", "field": "active_objective", "text": phrase},
        {"family_id": "b", "field": "active_objective", "text": phrase},
    ]
    old = [{"path": "old.jsonl", "record_index": 0, "field": "prompt",
            "text": phrase}]
    cross, within = leakage.compare(new, old)
    assert len(cross) == 2
    assert len(within) == 1
    assert cross[0]["jaccard"] == 1.0
