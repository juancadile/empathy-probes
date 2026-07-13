import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data_generation"))
import materialize_gate1_families as gate1  # noqa: E402


def rows(kind, per_source):
    domains = [f"domain-{index}" for index in range(per_source)]
    result = []
    for source in ("s0", "s1", "s2", "s3"):
        for index, domain in enumerate(domains):
            result.append({"family_id": f"{kind}_{source}_{index:02d}",
                           "source": source, "domain": domain})
    return result


def test_stratified_assignment_is_deterministic_and_balanced():
    items = rows("writer", 12)
    first = gate1.stratified_assignment(items, gate1.WRITER_SPLITS, 123)
    second = gate1.stratified_assignment(items, gate1.WRITER_SPLITS, 123)
    assert first == second
    assert Counter(first.values()) == Counter({label: 16
                                               for label in gate1.WRITER_SPLITS})
    for source in ("s0", "s1", "s2", "s3"):
        assert Counter(first[row["family_id"]] for row in items
                       if row["source"] == source) == Counter(
                           {label: 4 for label in gate1.WRITER_SPLITS})


def test_frozen_tails_fit_structural_tolerance():
    social = gate1.tail_metrics(gate1.HELP_TAIL, gate1.PERSIST_TAIL)
    process = gate1.tail_metrics(
        gate1.PROCESS_TAIL, gate1.PROCESS_PERSIST_TAIL)
    assert social["byte_relative_difference"] <= 0.10
    assert social["whitespace_token_difference"] <= 2
    assert process["byte_relative_difference"] <= 0.10
    assert process["whitespace_token_difference"] <= 2


def test_writer_prefix_has_no_branch_tail():
    row = {
        "narrator_role": "analyst", "active_objective": "finish a report",
        "objective_next_step": "check totals",
        "interruption_loss": "the submission will be late",
        "person_role": "colleague",
        "urgent_message": "I'm stuck and need help now.",
        "nonsocial_signal": "a warning light turns red",
    }
    for variant in gate1.VARIANTS:
        prefix = gate1.writer_prefix(row, variant, False)
        assert gate1.HELP_TAIL not in prefix
        assert gate1.PERSIST_TAIL not in prefix
        assert prefix.endswith("My decision: ")
