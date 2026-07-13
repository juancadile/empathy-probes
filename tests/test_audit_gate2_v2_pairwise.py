import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "evaluation"))
import audit_gate2_v2_pairwise as audit  # noqa: E402


def requests():
    data = ROOT / "data/gate_families/gate2_v2"
    return audit.requests_for(
        audit.load_jsonl(data / "wp1_families.jsonl"),
        audit.load_jsonl(data / "wp3_families.jsonl"))


def test_pairwise_request_matrix_is_complete_and_unique():
    items = requests()
    assert len(items) == 848
    assert len({item["request_id"] for item in items}) == len(items)
    assert {item["partition"] for item in items} == {
        "WP1-v2-dev", "WP1-v2-confirm", "WP3-v2-dev",
        "WP3-v2-confirm", "B8-v2-confirm"}


def test_pairwise_order_is_opaque_and_reproducible():
    first = requests()
    second = requests()
    assert [(item["request_id"], item["expected_higher"]) for item in first] == [
        (item["request_id"], item["expected_higher"]) for item in second]
    counts = {label: sum(item["expected_higher"] == label for item in first)
              for label in ("A", "B")}
    assert abs(counts["A"] - counts["B"]) < 80
    assert all("ARM A:" in item["prompt"] and "ARM B:" in item["prompt"]
               for item in first)


def test_analysis_requires_fourteen_of_sixteen_per_check():
    records = []
    for index in range(16):
        records.append({
            "kind": "gate2_wp1_v2", "partition": "WP1-v2-confirm",
            "label": "T_new", "check": "task_persistence",
            "expected_higher": "A", "rating": {
                "higher": "A" if index < 14 else "B",
                "confidence_1_to_5": 5},
        })
    report = audit.analyze(records)
    assert report["passed_pairwise_screen"]
    assert report["checks"][0]["required_correct"] == 14
    records[13]["rating"]["higher"] = "B"
    assert not audit.analyze(records)["passed_pairwise_screen"]


def test_generated_prompts_request_strict_json_only():
    item = requests()[0]
    assert "Return only one JSON object" in item["prompt"]
    assert '"higher":"A or B or TIE"' not in item["prompt"]
    assert json.loads('{"higher":"A","confidence_1_to_5":5}')["higher"] == "A"


def test_pairwise_degeneracy_gate_rejects_template_copying_and_one_choice():
    copied = [{"raw_output": "same", "rating": {
        "higher": "A", "confidence_1_to_5": 5}} for _ in range(4)]
    assert audit.degeneracy_reason(copied) == "all raw outputs are byte-identical"
    one_choice = [{"raw_output": str(index), "rating": {
        "higher": "A", "confidence_1_to_5": 5}} for index in range(4)]
    assert "choices are identical" in audit.degeneracy_reason(one_choice)
    varied = [{"raw_output": str(index), "rating": {
        "higher": "A" if index % 2 else "B", "confidence_1_to_5": 5}}
        for index in range(4)]
    assert audit.degeneracy_reason(varied) is None
