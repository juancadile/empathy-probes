import sys
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data_generation"))
import materialize_gate2_v2 as gate2  # noqa: E402


def rendered():
    wp1_rows, _ = gate2.load_kind("wp1", gate2.BLUEPRINT_REVISION)
    wp3_rows, _ = gate2.load_kind("wp3", gate2.BLUEPRINT_REVISION)
    wp1_rows = gate2.apply_neutral_overrides("wp1", wp1_rows)
    wp3_rows = gate2.apply_neutral_overrides("wp3", wp3_rows)
    wp1_assignment = gate2.stratified_assignment(
        wp1_rows, gate2.WP1_SPLITS, gate2.WP1_SPLIT_SEED)
    wp3_assignment = gate2.stratified_assignment(
        wp3_rows, gate2.WP3_SPLITS, gate2.WP3_SPLIT_SEED)
    wp1_messages = gate2.message_derangement(
        "wp1", wp1_rows, gate2.WP1_MESSAGE_SEED)
    wp3_messages = gate2.message_derangement(
        "wp3", wp3_rows, gate2.WP3_MESSAGE_SEED)
    return (
        gate2.wp1_records(wp1_rows, wp1_assignment, wp1_messages),
        gate2.wp3_records(wp3_rows, wp3_assignment, wp3_messages),
    )


def test_full_v2_materialization_passes_structural_validation():
    wp1, wp3 = rendered()
    result = gate2.validate(wp1, wp3)
    assert result["passed"], result["errors"]
    assert result["wp1_families"] == 32
    assert result["wp3_families"] == 48


def test_every_message_assignment_is_cross_source_domain_and_content():
    for records in rendered():
        by_family = {}
        for record in records:
            by_family.setdefault(record["family_id"], record)
        for record in by_family.values():
            assert record["source"] != record["message_source"]
            assert record["domain"] != record["message_domain"]
            assert record["objective_message_term_overlap"] == []


def test_wp1_exact_controls_are_renderer_owned():
    wp1, _ = rendered()
    direct = [row for row in wp1 if row["variant_id"] == "v0_direct"]
    family = direct[0]["family_id"]
    rows = {row["contrast"]: row for row in direct
            if row["family_id"] == family}

    l_new = rows["L_new"]
    assert len(l_new["arms"]) == 4
    assert all(l_new["distress_quote"] in arm["prompt"]
               for arm in l_new["arms"])
    assert any("fictional person in a test service" in arm["prompt"]
               for arm in l_new["arms"])

    b_new = {arm["arm_id"]: arm for arm in rows["B_new"]["arms"]}
    assert "separate active objective" in b_new["active_zero_cost"]["prompt"]
    assert "currently have no active objective or pending work" in (
        b_new["no_active_objective"]["prompt"])
    assert rows["D_new"]["arms"][0]["prompt"].split(": ", 1)[-1] == (
        rows["D_new"]["arms"][1]["prompt"].split(": ", 1)[-1])


def test_wp3_observation_is_choice_free_and_quote_matched():
    _, wp3 = rendered()
    observation = next(row for row in wp3
                       if row["variant_id"] == "v0_direct"
                       and row["cell"] == "observation")
    assert {arm["continuation"] for arm in observation["arms"]} == {
        gate2.FIXED_TASK_CONTINUATION}
    assert all(observation["distress_quote"] in arm["prompt"]
               for arm in observation["arms"])
    assert all("selected action" not in arm["prompt"]
               and "choose" not in arm["prompt"]
               for arm in observation["arms"])


def test_wp3_resolved_and_neutral_controls_do_not_claim_unresolved_need():
    _, wp3 = rendered()
    controls = next(row for row in wp3
                    if row["variant_id"] == "v0_direct"
                    and row["cell"] == "resolved_neutral_controls")
    arms = {arm["arm_id"]: arm["prompt"] for arm in controls["arms"]}
    assert "explicitly resolved" in arms["resolved_current"]
    assert "no support is needed now" in arms["resolved_current"]
    assert "neutral social fact" in arms["neutral_current"]
    assert "remains unresolved" not in arms["resolved_current"]
    assert "remains unresolved" not in arms["neutral_current"]


def test_cost_ladder_is_fixed_and_ordered_in_every_family():
    wp1, wp3 = rendered()
    for record in wp1:
        if record["contrast"] != "Ctext_new":
            continue
        arms = {arm["arm_id"]: arm["prompt"] for arm in record["arms"]}
        assert set(arms) == set(gate2.COST_TEXT)
        for level, text in gate2.COST_TEXT.items():
            assert text in arms[level]
    for record in wp3:
        if record["cell"] != "cost":
            continue
        arms = {arm["arm_id"]: arm["prompt"] for arm in record["arms"]}
        for level, text in gate2.COST_TEXT.items():
            assert text in arms[level]
        assert "currently have no active objective or pending work" in arms["none"]


def test_neutral_messages_contain_no_need_or_distress_markers():
    forbidden = re.compile(
        r"\b(worr\w*|struggl\w*|troubl\w*|confus\w*|unsure|uncertain|"
        r"help|support|stuck|overwhelm\w*|anxious|frustrat\w*|need|request|"
        r"problem|difficult\w*|excited|thrilled)\b", re.IGNORECASE)
    wp1_rows, _ = gate2.load_kind("wp1", gate2.BLUEPRINT_REVISION)
    wp3_rows, _ = gate2.load_kind("wp3", gate2.BLUEPRINT_REVISION)
    wp1_rows = gate2.apply_neutral_overrides("wp1", wp1_rows)
    wp3_rows = gate2.apply_neutral_overrides("wp3", wp3_rows)
    assert all(not forbidden.search(row["neutral_social_message"])
               for row in wp1_rows)
    assert all(not forbidden.search(row["neutral_message"])
               for row in wp3_rows)
