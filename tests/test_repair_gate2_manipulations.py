import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "evaluation"))
import repair_gate2_manipulations as repair  # noqa: E402


def test_merge_repairs_preserves_order_and_superseded_attempt():
    first = [
        {"request_id": "a", "raw_output": "bad", "rating": None,
         "parse_error": "cut", "kind": "wp1"},
        {"request_id": "b", "raw_output": "ok", "rating": {},
         "parse_error": None, "kind": "wp1"},
    ]
    fixed = [{"request_id": "a", "raw_output": "fixed", "rating": {},
              "parse_error": None, "kind": "wp1"}]
    merged = repair.merge_repairs(first, fixed)
    assert [record["request_id"] for record in merged] == ["a", "b"]
    assert merged[0]["superseded_attempt"]["raw_output"] == "bad"
    assert merged[1]["raw_output"] == "ok"
