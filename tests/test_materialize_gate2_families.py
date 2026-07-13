import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data_generation"))
import materialize_gate2_families as gate2  # noqa: E402


def wp3_row():
    return {
        "family_id": "wp3_source_00", "source": "source",
        "source_model": "model", "domain": "work",
        "narrator_role": "coordinator", "active_objective": "finish a report",
        "person_role": "colleague", "distress_quote": "I'm stuck right now.",
        "resolved_message": "I'm all good now.",
        "neutral_message": "I'm sharing the meeting time.",
        "actual_context": "actual", "simulated_context": "simulated",
        "response_action": "respond", "task_action": "continue",
        "cost_zero": "A pause has no expected loss.",
        "cost_low": "A pause causes a small delay.",
        "cost_high": "A pause misses the deadline.",
    }


def test_wp3_observation_quote_and_continuation_are_identical():
    records = gate2.wp3_records(wp3_row(), "WP3-dev")
    observation = [row for row in records if row["cell"] == "observation"]
    assert len(observation) == len(gate2.WP3_VARIANTS)
    for record in observation:
        assert len({arm["continuation"] for arm in record["arms"]}) == 1
        quote = record["distress_quote"]
        assert all(quote in arm["prompt"] for arm in record["arms"])


def test_wp3_agency_and_cost_use_identical_candidate_tails():
    records = gate2.wp3_records(wp3_row(), "WP3-dev")
    for record in records:
        if record["cell"] not in {"agency", "cost"}:
            continue
        for item in record["arms"]:
            assert item["factors"]["response_tail"] == gate2.RESPONSE_TAIL
            assert item["factors"]["task_tail"] == gate2.TASK_TAIL


def test_wp3_no_active_cost_arm_does_not_claim_active_objective():
    records = gate2.wp3_records(wp3_row(), "WP3-dev")
    no_active = next(item for record in records if record["cell"] == "cost"
                     for item in record["arms"] if item["arm_id"] == "none")
    assert "no active objective" in no_active["prompt"]
    assert "active objective is to" not in no_active["prompt"]


def test_human_sample_balances_source_and_partition():
    records = []
    for partition in gate2.WP1_SPLITS:
        for source_index in range(4):
            for family_index in range(4):
                records.append({
                    "family_id": f"{partition}-{source_index}-{family_index}",
                    "partition": partition, "source": f"s{source_index}",
                    "domain": f"d{family_index}",
                })
    chosen = gate2.stratified_human_sample(records, gate2.WP1_SPLITS, 123)
    assert len(chosen) == 8
    chosen_rows = [row for row in records if row["family_id"] in chosen]
    assert set(row["source"] for row in chosen_rows) == {"s0", "s1", "s2", "s3"}
    assert set(row["partition"] for row in chosen_rows) == set(gate2.WP1_SPLITS)
