import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "evaluation"))
import audit_gate2_manipulations as audit  # noqa: E402


def test_requests_flatten_only_direct_variant():
    record = {
        "variant_id": "v0_direct", "family_id": "f", "partition": "p",
        "source": "s", "domain": "d", "contrast": "D_new",
        "arms": [{"arm_id": "warm", "prompt": "warm text", "factors": {}},
                 {"arm_id": "terse", "prompt": "terse text", "factors": {}}],
    }
    requests = audit.requests_for([record], "wp1")
    assert len(requests) == 2
    assert {item["arm_id"] for item in requests} == {"warm", "terse"}


def test_validate_requires_every_dimension():
    record = {"rating": {key: 3 for key in audit.RATING_KEYS}}
    record["rating"].update({"active_objective": True, "confidence_1_to_5": 5})
    assert audit.validate(record) == []
    del record["rating"][audit.RATING_KEYS[0]]
    assert audit.RATING_KEYS[0] in audit.validate(record)
