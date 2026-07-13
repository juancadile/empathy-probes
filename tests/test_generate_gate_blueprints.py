import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_generation.generate_gate_blueprints import (
    DOMAIN_SLOTS, KINDS, SOURCES, output_path, prompt_for, validate_families,
)


def valid_payload(kind):
    spec = KINDS[kind]
    count = spec["count_per_source"]
    families = []
    for index, domain in enumerate(DOMAIN_SLOTS[count]):
        row = {field: f"value {index} {field}" for field in spec["fields"]}
        row["domain"] = domain
        families.append(row)
    return {"families": families}


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_blueprint_schema_and_ids(kind):
    source = next(iter(SOURCES))
    rows = validate_families(kind, source, valid_payload(kind))
    assert len(rows) == KINDS[kind]["count_per_source"]
    assert len({row["family_id"] for row in rows}) == len(rows)
    assert all(row["source"] == source for row in rows)


def test_rejects_wrong_domain_order():
    payload = valid_payload("writer")
    payload["families"][0]["domain"] = "wrong"
    with pytest.raises(ValueError, match="domain"):
        validate_families("writer", next(iter(SOURCES)), payload)


def test_rejects_extra_fields_and_wrong_count():
    payload = valid_payload("r2b")
    payload["families"][0]["extra"] = "x"
    with pytest.raises(ValueError, match="fields mismatch"):
        validate_families("r2b", next(iter(SOURCES)), payload)
    payload = valid_payload("r2b")
    payload["families"].pop()
    with pytest.raises(ValueError, match="expected"):
        validate_families("r2b", next(iter(SOURCES)), payload)


def test_prompt_places_generation_outside_matched_branch_text():
    prompt = prompt_for("writer", KINDS["writer"]["count_per_source"], 2)
    assert "Do not write experimental instructions" in prompt
    assert "branch labels" in prompt
    assert "help_action MUST perform help now" in prompt


def test_r2b_revision3_separates_welfare_need_from_objective_cost():
    prompt = prompt_for("r2b", KINDS["r2b"]["count_per_source"], 3)
    assert "PRESENT personal state" in prompt
    assert "speaker's difficulty" in prompt
    assert "describe only consequences for the narrator's" in prompt
    assert "present tense" in prompt


def test_r2b_revision4_rejects_objective_message_overlap():
    payload = valid_payload("r2b")
    for row in payload["families"]:
        row.update({
            "active_objective": "finish the quarterly budget report",
            "objective_next_step": "review budget totals",
            "urgent_message": "I'm worried about the budget and need advice.",
            "resolved_message": "I'm okay now and don't need support.",
            "cost_zero": "There is no expected delay",
            "cost_low": "The review starts five minutes late",
            "cost_medium": "The review window becomes compressed",
            "cost_high": "The report misses its filing window",
            "no_active_objective_state": "I currently have no pending work",
        })
    with pytest.raises(ValueError, match="terms overlap"):
        validate_families(
            "r2b", next(iter(SOURCES)), payload, strict=True, revision=4)


def test_strict_writer_validation_rejects_deferred_help():
    payload = valid_payload("writer")
    for row in payload["families"]:
        row["urgent_message"] = "I'm stuck and need your help now."
        row["help_action"] = "offer to help after finishing the report"
        row["nonsocial_signal"] = "a red warning light flashes"
        row["nonsocial_action"] = "pause to inspect the warning light"
    with pytest.raises(ValueError, match="defers help"):
        validate_families("writer", next(iter(SOURCES)), payload, strict=True)


def test_revisioned_output_does_not_overwrite_v1():
    source = next(iter(SOURCES))
    assert output_path("writer", source, 1).name == f"writer_{source}.json"
    assert output_path("writer", source, 2).name == f"writer_{source}_v2.json"
