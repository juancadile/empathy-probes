import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_generation.generate_gate_blueprints import (
    DOMAIN_SLOTS, KINDS, SOURCES, prompt_for, validate_families,
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
    prompt = prompt_for("writer", KINDS["writer"]["count_per_source"])
    assert "Do not write experimental instructions" in prompt
    assert "branch labels" in prompt
