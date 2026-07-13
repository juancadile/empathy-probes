import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "evaluation"))

from frontier_judge_sensitivity_batch import (  # noqa: E402
    classify_r2b_part,
    gate2_requests,
    r2b_requests,
)


def test_r2b_sensitivity_reuses_exact_160_blinded_packet_prompts():
    requests = r2b_requests()
    assert len(requests) == 160
    assert len({item["audit_id"] for item in requests}) == 160
    assert len({item["custom_id"] for item in requests}) == 160
    assert all("Do not copy a default or midpoint response" in item["prompt"]
               for item in requests)


def test_gate2_sensitivity_reuses_all_opaque_pairwise_requests():
    requests = gate2_requests()
    assert len(requests) == 848
    assert len({item["request_id"] for item in requests}) == 848
    assert len({item["custom_id"] for item in requests}) == 848
    assert {item["expected_higher"] for item in requests} == {"A", "B"}


def test_precommitted_r2b_branches_are_three_way():
    assert classify_r2b_part(0.0, [-0.2, 0.2]) == "equivalently_flat"
    assert classify_r2b_part(1.0, [0.4, 1.5]) == "positively_rising"
    assert classify_r2b_part(0.4, [-0.1, 0.9]) == "inconclusive"
