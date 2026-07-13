import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "evaluation"))
import audit_gate1_manipulations as audit  # noqa: E402


def test_parse_json_accepts_fenced_and_embedded_objects():
    assert audit.parse_json('```json\n{"x": 1}\n```') == {"x": 1}
    assert audit.parse_json('answer: {"x": 2}') == {"x": 2}


def test_bootstrap_interval_is_deterministic():
    values = np.array([1.0, 2.0, 3.0])
    assert audit.bootstrap_interval(values, .90, 42, 1000) == \
        audit.bootstrap_interval(values, .90, 42, 1000)


def test_writer_analysis_requires_correct_need_and_branch_labels():
    records = []
    for family_index in range(16):
        for condition in ("current_need", "task_control"):
            for order, expected in (("positive_first", "A"),
                                    ("negative_first", "B")):
                records.append({
                    "family_id": f"f{family_index}", "partition": "p",
                    "source": "s", "domain": "d", "condition": condition,
                    "expected_interrupt_option": expected,
                    "expected_persist_option": "B" if expected == "A" else "A",
                    "expected_person_in_current_need": condition == "current_need",
                    "rating": {"interrupt_option": expected,
                               "persist_option": "B" if expected == "A" else "A",
                               "person_in_current_need": condition == "current_need"},
                })
    report = audit.analyze_writer(records)
    assert report["passed_machine_gate"]
    assert report["partitions"]["p"]["branch_majority_correct"] == 16
