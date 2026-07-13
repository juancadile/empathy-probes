"""Manipulation-pretest runner: deterministic blinded extraction, strict
rating parse, retry/error persistence, summaries, historical-artifact
protection, and offline dry-run export (Integrity Repair A). No network."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import src.evaluation.manipulation_pretest as mp  # noqa: E402


def test_need_battery_extraction_matches_historical_design():
    items, sources = mp.extract_items("need_v2_2")
    assert len(items) == 80  # 4 arms x 10 families x 2 ack variants
    by_arm = {}
    for item in items:
        by_arm.setdefault(item["arm"], []).append(item)
    assert {arm: len(v) for arm, v in by_arm.items()} == {
        "urgent": 20, "mild": 20, "resolved": 20, "excited": 20}
    assert len({i["item_id"] for i in items}) == 80
    assert all(i["text"].strip() for i in items)
    assert all(i["family"].startswith(("cx_",)) for i in items)
    assert len(sources) == 4 and all(s["sha256"] for s in sources.values())


def test_moral_battery_extraction():
    items, sources = mp.extract_items("moral_v2_2")
    assert len(items) == 60  # 3 arms x 10 families x 2 ack variants
    arms = {i["arm"] for i in items}
    assert arms == {"lower", "equal", "higher"}
    assert len(sources) == 1  # single file, arm from relative_need field


def test_extraction_is_deterministic():
    a, _ = mp.extract_items("need_v2_2")
    b, _ = mp.extract_items("need_v2_2")
    assert a == b


def test_extraction_rejects_ambiguous_pinning(tmp_path, monkeypatch):
    cell = tmp_path / "cell.jsonl"
    rows = [
        {"scenario_id": "f1", "ack_variant": 0, "closing_variant": 0,
         "cost_rank": 0, "shared_prefix": "text A", "cell": "X"},
        {"scenario_id": "f1", "ack_variant": 0, "closing_variant": 0,
         "cost_rank": 0, "shared_prefix": "text B", "cell": "X"},
    ]
    cell.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setitem(mp.BATTERIES, "synthetic", {
        "arms": {"only": "cell.jsonl"}, "arm_field": None,
        "pin": {"cost_rank": 0, "closing_variant": 0},
        "unit_fields": ("scenario_id", "ack_variant"),
        "rated_field": "shared_prefix", "questions": ("need_now",),
    })
    with pytest.raises(mp.PretestExtractionError, match="distinct rated texts"):
        mp.extract_items("synthetic", repo_root=tmp_path)
    monkeypatch.setitem(mp.BATTERIES["synthetic"], "pin", {"cost_rank": 99})
    with pytest.raises(mp.PretestExtractionError, match="no rows"):
        mp.extract_items("synthetic", repo_root=tmp_path)


def test_rendered_prompt_is_blind_and_versioned():
    items, _ = mp.extract_items("need_v2_2")
    item = items[0]
    rendered = mp.render_rating_prompt(item["text"], "need_now",
                                       mp.DEFAULT_PROMPT_VERSION,
                                       target=item["target"]["descriptor"])
    spec = mp.PRETEST_PROMPTS[mp.DEFAULT_PROMPT_VERSION]
    q = spec["questions"]["need_now"]
    # byte-exact equality IS the blindness proof: nothing beyond the rated
    # text and the versioned question/scale (with the target descriptor
    # substituted) enters the judge input. (Arm LABELS like "excited" can
    # legitimately occur inside stimulus prose, so blindness is about
    # injected identity, not vocabulary.)
    assert rendered == spec["template"].format(
        text=item["text"],
        question=q["question"].format(target=item["target"]["descriptor"]),
        scale=q["scale"], low=q["low"], high=q["high"])
    for leak in (item["family"], item["item_id"]):  # cx_..-style ids
        assert leak not in rendered


@pytest.mark.parametrize("text,expected", [
    ("4", (4, "parsed")),
    (" 5. ", (5, "parsed")),
    ('"1"', (1, "parsed")),
    ("3 (moderate)", (None, "not_an_integer")),
    ("six", (None, "not_an_integer")),
    ("9", (None, "out_of_scale")),
    ("0", (None, "out_of_scale")),
    ("", (None, "not_an_integer")),
    (None, (None, "no_text")),
])
def test_parse_rating_strict(text, expected):
    assert mp.parse_rating(text, 1, 5) == expected


def ok_response(text):
    return (200, {"id": "msg_1", "model": "judge-x", "stop_reason": "end_turn",
                  "usage": {"output_tokens": 1},
                  "content": [{"type": "text", "text": text}]}, None)


def test_rate_item_success_no_wasted_calls():
    calls = []

    def transport(body):
        calls.append(body)
        return ok_response("4")

    verdict = mp.rate_item(transport, "judge-x", "prompt", 1, 5,
                           sleep_fn=lambda s: None)
    assert verdict["rating"] == 4
    assert verdict["rating_status"] == "rated"
    assert len(calls) == 1
    assert verdict["attempts"][0]["raw_response"]["id"] == "msg_1"


def test_rate_item_failure_stays_unknown():
    verdict = mp.rate_item(lambda body: ok_response("between 3 and 4"),
                           "judge-x", "prompt", 1, 5, max_attempts=2,
                           sleep_fn=lambda s: None)
    assert verdict["rating"] is None
    assert verdict["rating_status"] == "error_or_parse_failure"
    assert len(verdict["attempts"]) == 2
    assert all(a["parse_status"] == "not_an_integer"
               for a in verdict["attempts"])


def test_summarize_family_arm_and_unknowns():
    items = []
    for arm, ratings in [("urgent", [5, 4]), ("resolved", [2, None])]:
        for i, rating in enumerate(ratings):
            items.append({
                "arm": arm, "family": f"fam{i}",
                "ratings": {"need_now": {"rating": rating}},
            })
    summary = mp.summarize(items, ("need_now",))["need_now"]
    assert summary["urgent"]["mean"] == 4.5
    assert summary["urgent"]["n"] == 2 and summary["urgent"]["n_unknown"] == 0
    assert summary["resolved"]["n"] == 1
    assert summary["resolved"]["n_unknown"] == 1
    assert summary["urgent"]["per_family_mean"] == {"fam0": 5.0, "fam1": 4.0}


def test_summarize_arm_with_all_unknown():
    items = [{"arm": "x", "family": "f",
              "ratings": {"q": {"rating": None}}}]
    summary = mp.summarize(items, ("q",))["q"]
    assert summary["x"]["mean"] is None and summary["x"]["n_unknown"] == 1


def test_resolve_out_path_protects_historical_artifacts(tmp_path):
    for protected in mp.PROTECTED_OUTPUTS:
        with pytest.raises(SystemExit, match="historical pretest"):
            mp.resolve_out_path(protected)
    existing = tmp_path / "exists.json"
    existing.write_text("{}")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        mp.resolve_out_path(str(existing))
    fresh = mp.resolve_out_path(str(tmp_path / "new.json"))
    assert fresh == tmp_path / "new.json"


def test_main_dry_run_exports_exact_inputs(tmp_path):
    out = tmp_path / "need_dryrun.json"
    assert mp.main(["--battery", "need_v2_2", "--dry-run",
                    "--out", str(out), "--shuffle-seed", "3"]) == 0
    export = json.loads(out.read_text())
    assert export["mode"] == "dry_run_export"
    assert export["judge"]["model"] is None and export["judge"]["api"] is None
    assert len(export["items"]) == 80
    assert all(i["rendered_judge_inputs"]["need_now"] for i in export["items"])
    assert len(export["presentation"]["order_item_ids"]) == 80
    # shuffled order differs from extraction order (seeded, deterministic)
    extraction_order = [i["item_id"] for i in export["items"]]
    assert export["presentation"]["order_item_ids"] != extraction_order


def test_main_live_requires_pinned_judge(tmp_path, capsys):
    # live mode refuses an omitted provider ...
    with pytest.raises(SystemExit):
        mp.main(["--battery", "moral_v2_2",
                 "--out", str(tmp_path / "moral_v2.json")])
    assert "--provider is required" in capsys.readouterr().err
    # ... and an omitted judge model (QA Q2)
    with pytest.raises(SystemExit):
        mp.main(["--battery", "moral_v2_2", "--provider", "openai",
                 "--out", str(tmp_path / "moral_v2b.json")])
    assert "--judge-model is required" in capsys.readouterr().err


def test_moral_target_snapshots_name_p2_across_families_and_questions():
    items, _ = mp.extract_items("moral_v2_2")
    by_family = {item["family"]: item for item in items}
    for family in ("mm_tutor", "mm_moderator"):
        item = by_family[family]
        descriptor = item["target"]["descriptor"]
        for question in ("need_now", "respond_now"):
            rendered = mp.render_rating_prompt(
                item["text"], question, mp.DEFAULT_PROMPT_VERSION,
                target=descriptor)
            assert descriptor in rendered
            assert "OTHER person" not in rendered
        assert item["target"]["role"] == "P2_new_arrival"
        assert item["target"]["manifest_sha256"] == mp.target_manifest_hash(
            item["target"]["manifest_version"])


def test_target_manifest_requires_exact_one_to_one_coverage(monkeypatch):
    version = mp.BATTERIES["moral_v2_2"]["target"]["manifest_version"]
    manifest = mp.MORAL_TARGET_MANIFESTS[version]
    families = set(manifest["descriptors"])
    broken = {**manifest, "descriptors": dict(manifest["descriptors"])}
    broken["descriptors"].pop("mm_tutor")
    monkeypatch.setitem(mp.MORAL_TARGET_MANIFESTS, version, broken)
    with pytest.raises(mp.PretestExtractionError, match="one-to-one"):
        mp.validate_target_coverage("moral_v2_2", families)


def test_need_v2_wording_is_byte_identical_to_historical_fixed_target():
    item = mp.extract_items("need_v2_2")[0][0]
    old = mp.render_rating_prompt(
        item["text"], "need_now", "pretest_rating_v1_2026-07-13")
    new = mp.render_rating_prompt(
        item["text"], "need_now", mp.DEFAULT_PROMPT_VERSION,
        target=item["target"]["descriptor"])
    assert new == old


def test_explicit_provider_dry_run_records_adapter_metadata(tmp_path):
    out = tmp_path / "need_openai.json"
    assert mp.main([
        "--battery", "need_v2_2", "--dry-run", "--provider", "openai",
        "--judge-model", "gpt-4.1-2025-04-14", "--out", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert payload["judge"]["api"] == "openai-chat-completions"
    assert payload["judge"]["request_metadata"]["provider"] == "openai"


def test_accepted_pretest_rejects_floating_judge_alias(tmp_path, capsys):
    with pytest.raises(SystemExit):
        mp.main([
            "--battery", "need_v2_2", "--provider", "openai",
            "--judge-model", "gpt-4.1", "--run-mode", "accepted",
            "--out", str(tmp_path / "out.json")])
    assert "snapshot-shaped" in capsys.readouterr().err
