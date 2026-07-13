import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data_generation"))

import build_e22b2_axis as e22  # noqa: E402


def blueprint(index, source, partition):
    return {
        "schema": "empathy-action-probes/e22b2-blueprint/1",
        "family_id": f"fresh-e22-{index}",
        "source": source,
        "source_model": f"model-{source}",
        "domain": f"domain-{index}",
        "partition": partition,
        "scenario": f"P1 and P2 each requested support conversation number {index}.",
        "p1_need_clause": "P1 has a moderate ongoing difficulty.",
        "need_clauses": {
            "low": "P2 has less substantial difficulty than P1.",
            "equal": "P2 has the same degree of difficulty as P1.",
            "high": "P2 has more substantial difficulty than P1.",
        },
        "immediacy_clauses": {
            "can_wait": "P2's response window remains open for several hours.",
            "respond_now": "P2's response window closes after this turn.",
        },
        "variants": [
            {"variant_id": f"v{variant}", "opening": f"Version {variant}.",
             "bridge": "Consider both situations separately."}
            for variant in range(4)
        ],
    }


def fixtures():
    return [
        blueprint(0, "a", "E22b2-dev"),
        blueprint(1, "a", "E22b2-confirm"),
        blueprint(2, "b", "E22b2-dev"),
        blueprint(3, "b", "E22b2-confirm"),
        blueprint(4, "c", "E22b2-dev"),
        blueprint(5, "c", "E22b2-confirm"),
    ]


def test_blueprints_render_exact_cartesian_product():
    rows = fixtures()
    assert e22.validate_blueprints(rows, min_families=6)["passed"]
    rendered = e22.render(rows, min_families=6)
    assert len(rendered) == 6 * 4 * 3 * 2
    assert e22.validate_rendered(rendered, 6)["passed"]


def test_axis_clauses_and_decision_are_byte_stable():
    rendered = e22.render(fixtures(), min_families=6)
    cells = [row for row in rendered if row["family_id"] == "fresh-e22-0"
             and row["variant_id"] == "v0"]
    for need in e22.NEED_LEVELS:
        assert len({row["need_clause"] for row in cells if row["need_level"] == need}) == 1
    for immediacy in e22.IMMEDIACY_LEVELS:
        assert len({row["immediacy_clause"] for row in cells
                    if row["immediacy_level"] == immediacy}) == 1
    assert {row["decision_clause"] for row in cells} == {e22.DECISION_CLAUSE}


def test_need_clause_cannot_leak_immediacy():
    rows = fixtures()
    rows[0]["need_clauses"]["high"] += " It is urgent now."
    result = e22.validate_blueprints(rows, min_families=6)
    assert not result["passed"]
    assert any("leaks immediacy" in error for error in result["errors"])


def test_immediacy_clause_cannot_leak_need():
    rows = fixtures()
    rows[0]["immediacy_clauses"]["respond_now"] += " The situation is severe."
    result = e22.validate_blueprints(rows, min_families=6)
    assert not result["passed"]
    assert any("leaks need" in error for error in result["errors"])


def test_prior_family_ids_are_rejected():
    rows = fixtures()
    rows[0]["family_id"] = "mm_tutor"
    result = e22.validate_blueprints(rows, min_families=6)
    assert not result["passed"]
    assert any("overlap prior moral axis" in error for error in result["errors"])
