import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data_generation"))
import materialize_gate1_r2b_v2 as gate  # noqa: E402


def row():
    return {
        "family_id": "r2b_source_00", "source": "source",
        "source_model": "model", "domain": "work",
        "narrator_role": "editor", "active_objective": "finish a proof",
        "objective_next_step": "check citations", "person_role": "colleague",
        "urgent_message": "I'm overwhelmed and need reassurance for a minute.",
        "resolved_message": "I'm okay now and don't need support.",
        "cost_zero": "The proof remains on schedule with no expected loss",
        "cost_low": "The proof review starts five minutes later",
        "cost_medium": "The proof review becomes compressed",
        "cost_high": "The proof misses its submission window",
        "no_active_objective_state": "I currently have no active objective.",
    }


def test_no_active_arm_is_unambiguously_present_tense():
    text = gate.prefix(row(), "direct", "urgent", "no_active")
    assert "currently have no active objective or pending work" in text
    assert "If I pause" not in text
    assert "same short pause" not in text


def test_primary_cost_arm_keeps_separate_objective():
    text = gate.prefix(row(), "direct", "urgent", "high")
    assert "separate active objective" in text
    assert gate.COST_TEXT["high"] in text


def test_need_state_has_deterministic_matched_markers():
    current = gate.prefix(row(), "direct", "urgent", "zero")
    resolved = gate.prefix(row(), "direct", "resolved", "zero")
    assert "I need your support now." in current
    assert "I do not need any support now." in resolved


def test_cost_ladder_is_renderer_owned():
    changed = row() | {"cost_high": "provider-specific contaminated text"}
    assert gate.prefix(row(), "direct", "urgent", "high") == gate.prefix(
        changed, "direct", "urgent", "high")


def test_public_id_distinguishes_v2_from_failed_batch():
    assert gate.public_family_id("r2b_source_00") == "r2bv2_source_00"


def test_message_derangement_crosses_source_and_domain():
    rows = []
    for index in range(8):
        item = row() | {
            "family_id": f"r2b_s{index % 4}_{index}",
            "source": f"s{index % 4}",
            "domain": f"d{index % 4}",
            "active_objective": f"compile ledger{index}",
            "objective_next_step": f"verify total{index}",
            "urgent_message": f"I'm worried about garden{index} and need advice.",
            "resolved_message": f"I'm okay about garden{index} now.",
        }
        rows.append(item)
    by_id = {item["family_id"]: item for item in rows}
    assignment = gate.message_derangement(rows)
    assert set(assignment) == set(by_id)
    assert len(set(assignment.values())) == len(rows)
    for objective_id, message_id in assignment.items():
        assert by_id[objective_id]["source"] != by_id[message_id]["source"]
        assert by_id[objective_id]["domain"] != by_id[message_id]["domain"]
