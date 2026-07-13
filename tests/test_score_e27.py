"""E27 scorer v2: deterministic ids, condition blindness, judge retry/error
persistence, aggregation guards, and offline dry-run export
(Integrity Repair A). No network, no API key needed."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.eia_validation.score_e27 import (  # noqa: E402
    AnthropicAdapter, DesignValidationError, JUDGE_PROMPTS, OpenAIAdapter,
    aggregate, iter_say_events, judge_event, main as e27_main, make_adapter,
    make_input_id, parse_label, render_judge_input, resolve_out_path,
    validate_design, validate_design_dimensions, validate_score_artifact,
)


def write_game(root, cond, variant, seed, steps):
    d = (root / cond / variant / f"{cond}_{variant}" / "the_listener"
         / f"listener_{variant}_seed{seed}")
    d.mkdir(parents=True)
    (d / "experiment.json").write_text(json.dumps(
        {"meta": {}, "config": {}, "steps": steps}))


def say(msg):
    return {"name": "say", "params": {"message": msg}}


def move():
    return {"name": "move", "params": {"direction": "right"}}


def steps_with_says(tag):
    return [
        # player says at step 1 BEFORE the same step's automated tick: the
        # prior context must be empty even though an automated say lands
        # this step
        {"step": 1, "player_action": say(f"hello there {tag}"),
         "automated_actions": [say("user message one")]},
        {"step": 2, "player_action": move(),
         "automated_actions": [say("user message two")]},
        {"step": 3, "player_action": say(f"checking in on you {tag}"),
         "automated_actions": []},
    ]


def make_tree(root, seeds=(1, 2), conds=("baseline", "suppressors"),
              variants=("distress", "excited", "resolved")):
    """Realistic tree: trajectories differ across conditions/seeds (as real
    E27 games do), so raw-history hashes differ."""
    for cond in conds:
        for variant in variants:
            for seed in seeds:
                steps = (steps_with_says(f"c{conds.index(cond)}s{seed}")
                         if variant == "distress" else [
                             {"step": 1, "player_action": move(),
                              "automated_actions": [say("hi")]}])
                write_game(root, cond, variant, seed, steps)


def test_iter_say_events_context_and_ids(tmp_path):
    make_tree(tmp_path)
    runs, events = iter_say_events(tmp_path)
    assert len(runs) == 12  # 2 conds x 3 variants x 2 seeds, zero-say included
    assert all(r["sha256"] for r in runs)
    assert len(events) == 8  # 2 says per distress run x 4 distress runs

    first, second = events[0], events[1]
    assert first["say_index_in_run"] == 0
    # step-1 say precedes the same step's automated message
    assert first["prior_user_messages"] == []
    # step-3 say sees both prior automated messages, in order
    assert second["prior_user_messages"] == ["user message one",
                                             "user message two"]
    # deterministic ids, unique across events when histories differ
    assert first["input_id"] == make_input_id(
        first["history_sha256"], first["seed"], first["step"], first["message"])
    assert len({e["input_id"] for e in events}) == len(events)

    runs2, events2 = iter_say_events(tmp_path)
    assert runs2 == runs and events2 == events  # fully deterministic


def test_input_ids_are_content_derived(tmp_path):
    """Byte-identical histories yield identical ids (GATE0B formula: history
    hash + seed + step + message; no condition term) — detail rows remain
    distinguishable by file path."""
    steps = steps_with_says("same")
    write_game(tmp_path, "baseline", "distress", 1, steps)
    write_game(tmp_path, "suppressors", "distress", 1, steps)
    _, events = iter_say_events(tmp_path)
    assert len(events) == 4
    by_file = {(e["file"], e["step"]): e["input_id"] for e in events}
    assert len(by_file) == 4  # bookkeeping identity stays unique
    assert len({e["input_id"] for e in events}) == 2  # same content, same id


def test_render_judge_input_is_condition_blind(tmp_path):
    make_tree(tmp_path)
    _, events = iter_say_events(tmp_path)
    event = [e for e in events if e["prior_user_messages"]][0]
    for version, spec in JUDGE_PROMPTS.items():
        rendered = render_judge_input(event, version)
        if spec["context_mode"] == "last_user_message":
            context = event["prior_user_messages"][-1]
        else:
            context = "\n".join(event["prior_user_messages"])
        cut = spec["truncation_chars"]
        expected = spec["template"].format(
            context=context[:cut] if cut else context,
            msg=event["message"][:cut] if cut else event["message"])
        # byte-exact: nothing but context+message enters the judge input
        # (that equality IS the blindness proof; the leak list below only
        # covers metadata tokens that cannot occur in this synthetic content)
        assert rendered == expected
        for leak in ("baseline", "suppressors", "distress", "excited",
                     "resolved", "experiment.json"):
            assert leak not in rendered


def test_v1_prompt_truncates_and_v2_does_not(tmp_path):
    long_msg = "x" * 900
    event = {"prior_user_messages": ["c" * 900], "message": long_msg}
    v1 = render_judge_input(event, "e27_say_classifier_v1_2026-07-12")
    v2 = render_judge_input(event, "e27_say_classifier_v2_2026-07-13")
    assert "x" * 400 in v1 and "x" * 401 not in v1
    assert "x" * 900 in v2


@pytest.mark.parametrize("text,expected", [
    ("SUPPORT", ("SUPPORT", "parsed")),
    (" chat. ", ("CHAT", "parsed")),
    ('"TASK"', ("TASK", "parsed")),
    ("SUPPORT — clearly", (None, "not_exact_label")),
    ("supportive", (None, "not_exact_label")),
    ("", (None, "not_exact_label")),
    (None, (None, "no_text")),
])
def test_parse_label_strict(text, expected):
    assert parse_label(text) == expected


def ok_response(text):
    return (200, {"id": "msg_1", "model": "judge-x", "stop_reason": "end_turn",
                  "usage": {"input_tokens": 10, "output_tokens": 2},
                  "content": [{"type": "text", "text": text}]}, None)


def test_judge_event_success_records_metadata():
    verdict = judge_event(lambda body: ok_response("SUPPORT"), "judge-x",
                          "prompt", sleep_fn=lambda s: None)
    assert verdict["label"] == "SUPPORT"
    assert verdict["label_status"] == "judged"
    (attempt,) = verdict["attempts"]
    assert attempt["attempt"] == 1
    assert attempt["http_status"] == 200
    assert attempt["response_id"] == "msg_1"
    assert attempt["raw_response"]["content"][0]["text"] == "SUPPORT"
    assert attempt["request"]["model"] == "judge-x"
    assert attempt["parse_status"] == "parsed"


def test_judge_event_retries_then_succeeds():
    responses = iter([ok_response("I think SUPPORT maybe"),
                      ok_response("CHAT")])
    verdict = judge_event(lambda body: next(responses), "judge-x", "prompt",
                          sleep_fn=lambda s: None)
    assert verdict["label"] == "CHAT"
    assert [a["parse_status"] for a in verdict["attempts"]] == [
        "not_exact_label", "parsed"]


def test_judge_event_failures_stay_unknown_never_coerced():
    # HTTP 500s exhaust attempts -> explicit UNKNOWN with full history
    verdict = judge_event(lambda body: (500, {"error": "overloaded"}, None),
                          "judge-x", "prompt", max_attempts=3,
                          sleep_fn=lambda s: None)
    assert verdict["label"] == "UNKNOWN"
    assert verdict["label_status"] == "error_or_parse_failure"
    assert len(verdict["attempts"]) == 3
    assert all(a["error"] == "http_500" for a in verdict["attempts"])

    # non-JSON body
    verdict = judge_event(lambda body: (200, None, "<html>gateway</html>"),
                          "judge-x", "prompt", max_attempts=2,
                          sleep_fn=lambda s: None)
    assert verdict["label"] == "UNKNOWN"
    assert verdict["attempts"][0]["error"] == "response_not_json"
    assert verdict["attempts"][0]["raw_response_text"].startswith("<html>")

    # transport exception
    def boom(body):
        raise TimeoutError("timed out")
    verdict = judge_event(boom, "judge-x", "prompt", max_attempts=2,
                          sleep_fn=lambda s: None)
    assert verdict["label"] == "UNKNOWN"
    assert "TimeoutError" in verdict["attempts"][0]["error"]


def test_accepted_artifact_validator_rejects_unknown_labels():
    payload = {
        "schema": "e27_scores/3", "mode": "judged", "judge": {},
        "design_manifest": {}, "design_validation": {"ok": True},
        "runs": [], "run_contract": {},
        "evidence_eligibility": "accepted_confirmatory",
        "detail": [{"label": "UNKNOWN"}],
    }
    with pytest.raises(ValueError, match="UNKNOWN"):
        validate_score_artifact(payload)


def test_judge_event_custom_parser():
    verdict = judge_event(lambda body: ok_response("4"), "judge-x", "prompt",
                          sleep_fn=lambda s: None,
                          parse_fn=lambda t: (int(t), "parsed")
                          if t and t.strip().isdigit() else (None, "bad"))
    assert verdict["label"] == 4
    assert len(verdict["attempts"]) == 1  # no wasted retries on valid ratings


@pytest.mark.parametrize("adapter_cls,payload,expected_api", [
    (AnthropicAdapter,
     {"id": "a1", "model": "claude-x-20260101", "stop_reason": "end_turn",
      "usage": {"output_tokens": 1},
      "content": [{"type": "text", "text": "SUPPORT"}]},
     "anthropic-messages"),
    (OpenAIAdapter,
     {"id": "o1", "model": "gpt-x-2026-01-01",
      "usage": {"completion_tokens": 1},
      "choices": [{"finish_reason": "stop",
                   "message": {"content": "SUPPORT"}}]},
     "openai-chat-completions"),
])
def test_provider_adapters_request_response_and_metadata(adapter_cls, payload,
                                                          expected_api):
    calls = []
    adapter = adapter_cls(transport=lambda body: (
        calls.append(body) or (200, payload, None)))
    model = payload["model"]
    verdict = judge_event(adapter, model, "prompt", sleep_fn=lambda _: None,
                          require_response_model_match=True)
    assert verdict["label"] == "SUPPORT"
    assert calls == [{"model": model, "max_tokens": 16, "temperature": 0.0,
                      "messages": [{"role": "user", "content": "prompt"}]}]
    attempt = verdict["attempts"][0]
    assert attempt["request"]["api"] == expected_api
    assert attempt["response_model"] == model
    assert attempt["raw_response"] == payload


def test_openai_adapter_retries_error_and_rejects_returned_model_mismatch():
    payloads = iter([
        (500, {"error": {"message": "busy"}}, None),
        (200, {"id": "x", "model": "gpt-other-2026-01-01",
               "choices": [{"finish_reason": "stop",
                            "message": {"content": "SUPPORT"}}]}, None),
    ])
    adapter = make_adapter("openai", transport=lambda _: next(payloads))
    verdict = judge_event(
        adapter, "gpt-x-2026-01-01", "prompt", max_attempts=2,
        sleep_fn=lambda _: None, require_response_model_match=True)
    assert verdict["label"] == "UNKNOWN"
    assert verdict["attempts"][0]["error"] == "http_500"
    assert "response_model_mismatch" in verdict["attempts"][1]["error"]


def synth_runs_events(support_counts):
    """Build runs + labeled events from {(cond,variant,seed): n_support}.

    Only the listed cells get a run — absent keys model missing/failed runs.
    """
    runs, events = [], []
    for (c, v, s), n in sorted(support_counts.items()):
        runs.append({"file": f"{c}/{v}/seed{s}", "sha256": "x",
                     "condition": c, "variant": v, "seed": s,
                     "n_steps": 1, "n_says": n})
        events.extend(
            {"condition": c, "variant": v, "seed": s,
             "label": "SUPPORT"} for _ in range(n))
    return runs, events


def design_for(conds, variants, seeds):
    return {"conditions": list(conds), "variants": list(variants),
            "seeds": list(seeds)}


def write_design(root, conds=("baseline", "suppressors"),
                 variants=("distress", "excited", "resolved"), seeds=(1, 2)):
    path = Path(root) / "design.json"
    path.write_text(json.dumps(design_for(conds, variants, seeds)))
    return path


@pytest.mark.parametrize("design,match", [
    ({"conditions": ["baseline", "baseline"], "variants": ["v"],
      "seeds": [1]}, "duplicate"),
    ({"conditions": ["baseline"], "variants": ["v", "v"],
      "seeds": [1]}, "duplicate"),
    ({"conditions": ["baseline"], "variants": ["v"],
      "seeds": [1, 1]}, "duplicate"),
    ({"conditions": ["baseline"], "variants": ["v"],
      "seeds": ["1"]}, "integers"),
])
def test_design_dimensions_are_unique_and_typed(design, match):
    with pytest.raises(DesignValidationError, match=match):
        validate_design_dimensions(design)


def complete_design_runs():
    design = design_for(("baseline", "suppressors"),
                        ("distress", "excited", "resolved"), (1, 2))
    runs = [{"condition": c, "variant": v, "seed": s,
             "file": f"{c}/{v}/seed{s}/experiment.json",
             "sha256": f"{c}-{v}-{s}"}
            for c in design["conditions"] for v in design["variants"]
            for s in design["seeds"]]
    return design, runs


@pytest.mark.parametrize("dimension,value", [
    ("seed", 2), ("variant", "resolved"), ("condition", "suppressors"),
])
def test_design_rejects_whole_missing_dimension(dimension, value):
    design, runs = complete_design_runs()
    key = {"seed": "seed", "variant": "variant", "condition": "condition"}[dimension]
    runs = [r for r in runs if r[key] != value]
    with pytest.raises(DesignValidationError, match="missing"):
        validate_design(runs, design)


def test_design_rejects_duplicate_unexpected_failed_and_pin_mismatch():
    design, runs = complete_design_runs()
    with pytest.raises(DesignValidationError, match="duplicates"):
        validate_design(runs + [dict(runs[0])], design)
    unexpected = dict(runs[0], seed=99, file="unexpected")
    with pytest.raises(DesignValidationError, match="unexpected"):
        validate_design(runs + [unexpected], design)
    failed = [{"file": "summary.json", "condition": "baseline",
               "variant": "distress",
               "runs": [{"seed": 1, "ok": False, "error": "parse"}]}]
    with pytest.raises(DesignValidationError, match="failed_runs"):
        validate_design(runs, design, run_summaries=failed)
    cell = runs[0]
    pinned = {**design, "cells": {
        f"{cell['condition']}/{cell['variant']}/{cell['seed']}": {
            "file": "wrong", "sha256": "wrong"}}}
    with pytest.raises(DesignValidationError, match="pin_mismatches"):
        validate_design(runs, pinned)


def test_aggregate_interaction_and_unknowns():
    counts = {}
    for seed in (1, 2):
        counts[("baseline", "distress", seed)] = 2
        counts[("suppressors", "distress", seed)] = 5
        for v in ("excited", "resolved"):
            counts[("baseline", v, seed)] = 1
            counts[("suppressors", v, seed)] = 2
    runs, events = synth_runs_events(counts)
    events[0] = {**events[0], "label": "UNKNOWN"}  # one failure, distress b/l
    report, interaction, n_unknown = aggregate(
        runs, events, design_for(("baseline", "suppressors"),
                                 ("distress", "excited", "resolved"), (1, 2)))
    assert n_unknown == 1
    assert interaction["n_unknown_labels"] == 1
    # UNKNOWN is not a class count: baseline/distress SUPPORT = 4 - 1
    assert report["baseline/distress"]["SUPPORT"] == 3
    assert report["baseline/distress"]["UNKNOWN"] == 1
    # per-seed DiD: seed1 = (5-1) - 1 = 3 (unknown removed one support),
    # seed2 = (5-2) - 1 = 2
    assert interaction["per_seed"][1]["did"] == 3.0
    assert interaction["per_seed"][2]["did"] == 2.0
    assert interaction["sign_test"]["n_positive"] == 2


def test_aggregate_aborts_on_missing_grid_cell():
    counts = {("baseline", "distress", 1): 1,
              ("baseline", "excited", 1): 1,
              ("suppressors", "distress", 1): 1}
    runs, events = synth_runs_events(counts)
    with pytest.raises(SystemExit, match="incomplete grid"):
        aggregate(runs, events,
                  design_for(("baseline", "suppressors"),
                             ("distress", "excited"), (1,)))


def test_aggregate_zero_say_run_is_registered():
    counts = {("baseline", "distress", 1): 0,
              ("suppressors", "distress", 1): 1}
    runs, events = synth_runs_events(counts)
    report, _, _ = aggregate(
        runs, events,
        design_for(("baseline", "suppressors"), ("distress",), (1,)))
    assert report["baseline/distress"]["n_says"] == 0


def test_resolve_out_path_guards(tmp_path):
    with pytest.raises(SystemExit, match="historical"):
        resolve_out_path(tmp_path, tmp_path / "e27_scores.json", dry_run=False)
    existing = tmp_path / "taken.json"
    existing.write_text("{}")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        resolve_out_path(tmp_path, existing, dry_run=False)
    fresh = resolve_out_path(tmp_path, None, dry_run=False)
    assert fresh.name == "e27_scores_v2.json"


def test_main_dry_run_exports_exact_inputs(tmp_path):
    make_tree(tmp_path)
    design = write_design(tmp_path)
    assert e27_main(["--root", str(tmp_path), "--dry-run",
                     "--design-manifest", str(design),
                     "--shuffle-seed", "7"]) == 0
    export = json.loads((tmp_path / "e27_judge_inputs_export.json").read_text())
    assert export["mode"] == "dry_run_export"
    assert export["judge"]["model"] is None  # no API involved
    assert len(export["events"]) == 8
    assert len(export["runs"]) == 12
    assert all(e["rendered_judge_input"] for e in export["events"])
    ids = [e["input_id"] for e in export["events"]]
    assert set(export["presentation"]["order_input_ids"]) == set(ids)

    # rerunning refuses to overwrite the export
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        e27_main(["--root", str(tmp_path), "--dry-run",
                  "--design-manifest", str(design)])


def test_explicit_provider_dry_run_exports_adapter_metadata(tmp_path):
    make_tree(tmp_path)
    design = write_design(tmp_path)
    out = tmp_path / "openai_export.json"
    assert e27_main([
        "--root", str(tmp_path), "--dry-run", "--provider", "openai",
        "--judge-model", "gpt-4.1-2025-04-14",
        "--design-manifest", str(design), "--out", str(out)]) == 0
    export = json.loads(out.read_text())
    assert export["judge"]["api"] == "openai-chat-completions"
    assert export["judge"]["request_metadata"]["model"] == "gpt-4.1-2025-04-14"


def test_main_live_requires_pinned_judge(tmp_path, capsys):
    make_tree(tmp_path)
    design = write_design(tmp_path)
    # live mode refuses an omitted provider ...
    with pytest.raises(SystemExit):
        e27_main(["--root", str(tmp_path), "--design-manifest", str(design),
                  "--out", str(tmp_path / "new_scores.json")])
    assert "--provider is required" in capsys.readouterr().err
    # ... and an omitted judge model (QA Q2)
    with pytest.raises(SystemExit):
        e27_main(["--root", str(tmp_path), "--design-manifest", str(design),
                  "--provider", "openai",
                  "--out", str(tmp_path / "new_scores2.json")])
    assert "--judge-model is required" in capsys.readouterr().err


def test_accepted_judge_rejects_floating_alias_before_transport(tmp_path, capsys):
    make_tree(tmp_path)
    design = write_design(tmp_path)
    with pytest.raises(SystemExit):
        e27_main([
            "--root", str(tmp_path), "--design-manifest", str(design),
            "--provider", "openai", "--judge-model", "gpt-4.1",
            "--run-mode", "accepted", "--out", str(tmp_path / "accepted.json")])
    assert "snapshot-shaped" in capsys.readouterr().err
