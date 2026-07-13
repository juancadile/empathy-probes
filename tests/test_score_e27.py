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
    JUDGE_PROMPTS, aggregate, iter_say_events, judge_event, main as e27_main,
    make_input_id, parse_label, render_judge_input, resolve_out_path,
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


def test_judge_event_custom_parser():
    verdict = judge_event(lambda body: ok_response("4"), "judge-x", "prompt",
                          sleep_fn=lambda s: None,
                          parse_fn=lambda t: (int(t), "parsed")
                          if t and t.strip().isdigit() else (None, "bad"))
    assert verdict["label"] == 4
    assert len(verdict["attempts"]) == 1  # no wasted retries on valid ratings


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
    report, interaction, n_unknown = aggregate(runs, events)
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
        aggregate(runs, events)


def test_aggregate_zero_say_run_is_registered():
    counts = {("baseline", "distress", 1): 0,
              ("suppressors", "distress", 1): 1}
    runs, events = synth_runs_events(counts)
    report, _, _ = aggregate(runs, events)
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
    assert e27_main(["--root", str(tmp_path), "--dry-run",
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
        e27_main(["--root", str(tmp_path), "--dry-run"])


def test_main_live_requires_pinned_judge(tmp_path, capsys):
    make_tree(tmp_path)
    with pytest.raises(SystemExit):
        e27_main(["--root", str(tmp_path),
                  "--out", str(tmp_path / "new_scores.json")])
    assert "--judge-model is required" in capsys.readouterr().err
