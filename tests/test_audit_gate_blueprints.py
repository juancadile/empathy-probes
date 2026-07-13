import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data_generation"))
import audit_gate_blueprints as MODULE  # noqa: E402


def writer(**overrides):
    item = {
        "family_id": "writer_source_00",
        "kind": "writer",
        "source": "source",
        "source_model": "model",
        "domain": "work",
        "narrator_role": "analyst",
        "active_objective": "finish a report",
        "objective_next_step": "check the totals",
        "interruption_loss": "the submission will be delayed",
        "person_role": "colleague",
        "urgent_message": "I'm stuck and need help checking this formula now.",
        "help_action": "pause to help the colleague check the formula",
        "nonsocial_signal": "a dashboard warning light turns red",
        "nonsocial_action": "pause to inspect the dashboard warning",
    }
    item.update(overrides)
    return item


def test_valid_writer_has_no_hard_findings():
    result = MODULE.audit_family(writer())
    assert result.hard_valid


def test_deferred_help_and_social_control_are_hard_failures():
    result = MODULE.audit_family(writer(
        help_action="offer to help after the report is submitted",
        nonsocial_signal="the colleague sounds anxious in chat"))
    codes = {finding.code for finding in result.findings
             if finding.severity == "hard"}
    assert {"help_action_deferred", "social_task_control"} <= codes


def test_message_constraints_are_enforced():
    result = MODULE.audit_family(writer(
        urgent_message=("I need help with this urgent task because my current "
                        "objective is blocked and I have repeated this request "
                        "many times without any useful response from anyone.")))
    codes = {finding.code for finding in result.findings
             if finding.severity == "hard"}
    assert "message_too_long" in codes
    assert "forbidden_message_word" in codes


def test_jaccard_handles_empty_and_identical_inputs():
    assert MODULE.jaccard(set(), set()) == 1.0
    grams = MODULE.char_ngrams("same semantic content")
    assert MODULE.jaccard(grams, grams) == 1.0
