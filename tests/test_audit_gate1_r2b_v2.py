import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "evaluation"))
import audit_gate1_r2b_v2 as audit  # noqa: E402


def test_primary_judge_is_pinned_and_independent():
    assert audit.MODEL == "Qwen/Qwen3-14B"
    assert len(audit.REVISION) == 40
    assert audit.SEED == 3010918835


def test_qwen_prompt_contains_no_literal_midpoint_answer():
    rows = audit.load_jsonl(
        ROOT / "data/gate_families/gate1_r2b_v2/r2b_families.jsonl")
    requests = audit.qwen_r2b_requests(rows)
    assert len(requests) == 320
    assert all('"current_need_1_to_5":3' not in item["prompt"]
               for item in requests)
    assert all("Do not copy a default or midpoint response" in item["prompt"]
               for item in requests)


def test_degeneracy_gate_rejects_template_copying_and_constant_ratings():
    copied = [{"raw_output": "same", "rating": {
        "current_need_1_to_5": 3, "interruption_cost_1_to_5": 3}} for _ in range(4)]
    assert audit.degeneracy_reason(copied) == "all raw outputs are byte-identical"
    constant = [{"raw_output": str(index), "rating": {
        "current_need_1_to_5": 3, "interruption_cost_1_to_5": 3}}
        for index in range(4)]
    assert "constant" in audit.degeneracy_reason(constant)
    varied = [{"raw_output": str(index), "rating": {
        "current_need_1_to_5": 1 + index, "interruption_cost_1_to_5": 2 + index}}
        for index in range(3)]
    assert audit.degeneracy_reason(varied) is None
