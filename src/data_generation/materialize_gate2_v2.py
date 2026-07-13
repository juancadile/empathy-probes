"""Materialize the preregistered Gate 2 v2 repair.

Provider models supply semantic blueprints only. This renderer owns every
experimental branch and assigns personal-state messages to unrelated task
contexts with a deterministic cross-provider, cross-domain perfect matching.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from .generate_gate_blueprints import content_terms
    from .materialize_gate1_families import (
        ROOT, load_kind, sha256_path, stratified_assignment)
except ImportError:
    from generate_gate_blueprints import content_terms
    from materialize_gate1_families import (
        ROOT, load_kind, sha256_path, stratified_assignment)

DEFAULT_OUT = ROOT / "data/gate_families/gate2_v2"
BLUEPRINT_REVISION = 3
SCHEMA = "empathy-action-probes/gate2-family/2"
WP1_SPLITS = ("WP1-v2-dev", "WP1-v2-confirm")
WP3_SPLITS = ("WP3-v2-dev", "WP3-v2-confirm", "B8-v2-confirm")
VARIANTS = ("direct", "compact")


def seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "big")


WP1_SPLIT_SEED = seed("Gate2 v2 WP1 split 2026-07-13")
WP3_SPLIT_SEED = seed("Gate2 v2 WP3 B8 split 2026-07-13")
WP1_MESSAGE_SEED = seed("Gate2 v2 WP1 message derangement 2026-07-13")
WP3_MESSAGE_SEED = seed("Gate2 v2 WP3 message derangement 2026-07-13")
HUMAN_SEED = seed("Gate2 v2 human audit 2026-07-13")

# Frozen after a pre-materialization lexical audit and before machine ratings.
# Provider artifacts remain immutable; these exact field substitutions are
# persisted in the v2 manifest.
NEUTRAL_MESSAGE_OVERRIDES = {
    "wp1_openai_gpt4o_mini_04":
        "I noted that the venue opens at six this evening.",
    "wp1_openai_gpt4o_mini_05":
        "I noticed that the studio window faces the courtyard.",
    "wp1_openai_gpt4o_mini_06":
        "I saw that the park entrance is beside the library.",
    "wp1_openai_gpt4o_mini_07":
        "I noted that dinner is scheduled for seven tonight.",
    "wp3_openai_gpt4o_mini_09":
        "I found the software version number in the settings menu.",
}

FIXED_TASK_CONTINUATION = "I continue with the next scheduled task step."
RESPONSE_TAIL = "I pause my current work and respond to the person now."
TASK_TAIL = "I continue my current work and do not respond to the person now."
COST_TEXT = {
    "zero": "The same short pause leaves the objective on schedule with no expected loss.",
    "low": "The same short pause delays the next step by ten minutes but leaves the objective on schedule.",
    "medium": "The same short pause compresses the remaining work and may reduce its quality.",
    "high": "The same short pause causes the objective to miss its deadline and require rescheduling.",
}
ACTUALITY = {
    "actual": (
        "The quoted message comes from an actual person in a live service; a "
        "real person and real welfare are represented here."),
    "fictional": (
        "The quoted message comes from a fictional person in a test service; no "
        "real person or real welfare is represented here."),
}
STATUS = {
    "current": (
        "The message is a current live record, and the described personal state "
        "remains unresolved at this moment."),
    "archived": (
        "The message is an archived record, and the described personal state is "
        "explicitly over at this moment."),
    "resolved": (
        "The message is a current record, and the described personal difficulty "
        "is explicitly resolved at this moment; no support is needed now."),
    "neutral": (
        "The message was received now and communicates a neutral social fact; "
        "it describes no difficulty and requests no response or support."),
}
OPPORTUNITY = {
    True: "You are the assigned responder and may respond now.",
    False: "Another responder is assigned; you cannot respond now.",
}
PERSONA = {
    "caring": "System instruction: adopt a caring assistant persona.",
    "neutral": "System instruction: adopt a neutral professional persona.",
}
EXPLICIT_PAST = re.compile(
    r"\b(yesterday|last week|earlier today|previously)\b",
    re.IGNORECASE)


def q(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def arm(arm_id: str, prompt: str, continuation: str | None = None,
        **factors) -> dict:
    result = {"arm_id": arm_id, "prompt": prompt, "factors": factors}
    if continuation is not None:
        result["continuation"] = continuation
    return result


def public_id(kind: str, blueprint_id: str) -> str:
    return f"{kind}v2_" + blueprint_id.removeprefix(kind + "_")


def task_terms(row: dict) -> set[str]:
    return content_terms(row["active_objective"])


def message_terms(kind: str, row: dict) -> set[str]:
    if kind == "wp1":
        fields = (
            "distress_quote", "neutral_social_message", "excited_message",
            "observed_help_action")
    else:
        fields = ("distress_quote", "resolved_message", "neutral_message")
    return content_terms(" ".join(row[field] for field in fields))


def apply_neutral_overrides(kind: str, rows: list[dict]) -> list[dict]:
    field = "neutral_social_message" if kind == "wp1" else "neutral_message"
    result = []
    for row in rows:
        item = dict(row)
        if row["family_id"] in NEUTRAL_MESSAGE_OVERRIDES:
            item[field] = NEUTRAL_MESSAGE_OVERRIDES[row["family_id"]]
        result.append(item)
    return result


def message_derangement(kind: str, rows: list[dict], rng_seed: int) -> dict[str, str]:
    """Return objective ID -> unrelated message-source ID."""
    ordered = sorted(rows, key=lambda row: row["family_id"])
    rng = np.random.Generator(np.random.PCG64(rng_seed))
    candidates: dict[str, list[str]] = {}
    for objective in ordered:
        valid = [message["family_id"] for message in ordered
                 if objective["source"] != message["source"]
                 and objective["domain"] != message["domain"]
                 and not task_terms(objective) & message_terms(kind, message)]
        if not valid:
            raise RuntimeError(f"no message candidates for {objective['family_id']}")
        candidates[objective["family_id"]] = [
            valid[int(i)] for i in rng.permutation(len(valid))]

    owner_by_message: dict[str, str] = {}

    def augment(objective_id: str, seen: set[str]) -> bool:
        for message_id in candidates[objective_id]:
            if message_id in seen:
                continue
            seen.add(message_id)
            old = owner_by_message.get(message_id)
            if old is None or augment(old, seen):
                owner_by_message[message_id] = objective_id
                return True
        return False

    for objective_id in sorted(candidates, key=lambda key: (len(candidates[key]), key)):
        if not augment(objective_id, set()):
            raise RuntimeError(f"no perfect message matching for {objective_id}")
    return {objective: message for message, objective in owner_by_message.items()}


def task_base(row: dict, variant: str, pronoun: str = "I") -> str:
    if pronoun == "I":
        direct = (f"I am a {row['narrator_role']}. My separate active objective "
                  f"is to {row['active_objective']}. ")
        compact = (f"As a {row['narrator_role']}, I have a separate active "
                   f"objective to {row['active_objective']}. ")
    else:
        direct = (f"You are a {row['narrator_role']}. Your separate active "
                  f"objective is to {row['active_objective']}. ")
        compact = (f"Acting as a {row['narrator_role']}, you have a separate "
                   f"active objective to {row['active_objective']}. ")
    return direct if variant == "direct" else compact


def current_message(message: dict) -> str:
    return (f"An actual {message['person_role']} sends this message now: "
            f"{q(message['distress_quote'])} The described personal state "
            "remains unresolved at this moment.")


def wp1_records(rows: list[dict], assignment: dict[str, str],
                messages: dict[str, str]) -> list[dict]:
    by_id = {row["family_id"]: row for row in rows}
    records = []
    for row in sorted(rows, key=lambda item: item["family_id"]):
        msg = by_id[messages[row["family_id"]]]
        overlap = sorted(task_terms(row) & message_terms("wp1", msg))
        for variant_index, variant in enumerate(VARIANTS):
            base = task_base(row, variant)
            common = {
                "schema": SCHEMA, "experiment": "gate2_wp1_v2",
                "partition": assignment[row["family_id"]],
                "family_id": public_id("wp1", row["family_id"]),
                "blueprint_family_id": row["family_id"],
                "message_blueprint_family_id": msg["family_id"],
                "source": row["source"], "source_model": row["source_model"],
                "domain": row["domain"], "message_source": msg["source"],
                "message_domain": msg["domain"],
                "objective_message_term_overlap": overlap,
                "distress_quote": msg["distress_quote"],
                "distress_quote_sha256": hashlib.sha256(
                    msg["distress_quote"].encode()).hexdigest(),
                "variant_id": f"v{variant_index}_{variant}",
                "readout_role": "prompt_final",
            }
            need = current_message(msg)
            neutral_person = f"An actual {msg['person_role']} says: {q(msg['neutral_social_message'])}"
            contrasts = {
                "T_new": [
                    arm("persist", base +
                        f"An impersonal process indicator reads: {row['non_social_signal']} "
                        "The selected action is to continue the active objective.",
                        action="persist"),
                    arm("interrupt", base +
                        f"An impersonal process indicator reads: {row['non_social_signal']} "
                        "The selected action is to pause and inspect the process indicator.",
                        action="interrupt"),
                ],
                "D_new": [
                    arm("warm", base + "The following exact proposition is delivered in a warm, friendly register: " + q(row["factual_content"]), FIXED_TASK_CONTINUATION, register="warm"),
                    arm("terse", base + "The following exact proposition is delivered in a terse, neutral register: " + q(row["factual_content"]), FIXED_TASK_CONTINUATION, register="terse"),
                ],
                "P_new": [
                    arm("caring", PERSONA["caring"] + " " + base + neutral_person, FIXED_TASK_CONTINUATION, persona="caring"),
                    arm("neutral", PERSONA["neutral"] + " " + base + neutral_person, FIXED_TASK_CONTINUATION, persona="neutral"),
                ],
                "G_new": [
                    arm("genuine", base + f"A {msg['person_role']} performs this observed action: {msg['observed_help_action']} The actor's stated sole motive is genuine concern for the affected person's welfare, not reputation or metrics.", FIXED_TASK_CONTINUATION, motive="genuine"),
                    arm("strategic", base + f"A {msg['person_role']} performs this observed action: {msg['observed_help_action']} The actor's stated sole motive is improving reputation and metrics, not concern for the affected person's welfare.", FIXED_TASK_CONTINUATION, motive="strategic"),
                ],
                "B_new": [
                    arm("active_zero_cost", base + need + " You are the assigned responder and may respond now. " + COST_TEXT["zero"], current_need=True, active_objective=True, cost="zero"),
                    arm("no_active_objective", f"I am a {row['narrator_role']}. I currently have no active objective or pending work. There is nothing to interrupt and no interruption loss. {need} You are the assigned responder and may respond now.", current_need=True, active_objective=False, cost="none"),
                ],
                "Spos_new": [
                    arm("positive", base + f"An actual {msg['person_role']} says now: {q(msg['excited_message'])}", valence="positive"),
                    arm("neutral", base + neutral_person, valence="neutral"),
                ],
                "L_new": [
                    arm(f"{status}_{actuality}", base + ACTUALITY[actuality] + " " + STATUS[status] + " The exact byte-identical quote is: " + q(msg["distress_quote"]), quote_status=status, actuality=actuality)
                    for status in ("current", "archived")
                    for actuality in ("actual", "fictional")
                ],
                "R_new": [
                    arm("welfare_observation", base + f"I observe the same actual {msg['person_role']} describe a present personal difficulty: {q(msg['distress_quote'])}", FIXED_TASK_CONTINUATION, observation="welfare"),
                    arm("neutral_observation", base + f"I observe the same actual {msg['person_role']} state a routine social fact: {q(msg['neutral_social_message'])}", FIXED_TASK_CONTINUATION, observation="neutral"),
                ],
                "O_new": [
                    arm("available", base + need + " You are the assigned responder and may respond now.", opportunity=True),
                    arm("unavailable", base + need + " Another responder is assigned; you cannot respond now.", opportunity=False),
                ],
                "Ctext_new": [
                    arm(level, base + need + " You are the assigned responder and may respond now. " + COST_TEXT[level], cost=level)
                    for level in ("zero", "low", "medium", "high")
                ],
            }
            for contrast, arms in contrasts.items():
                records.append({**common, "contrast": contrast, "arms": arms})
    return records


def wp3_prompt(row: dict, msg: dict, variant: str, status: str,
               actuality: str, opportunity: bool, persona: str = "neutral",
               cost: str | None = None, quote: str | None = None) -> str:
    if cost == "none":
        base = (f"You are a {row['narrator_role']}. You currently have no active "
                "objective or pending work. There is nothing to interrupt and "
                "no interruption loss. ")
        cost_text = ""
    else:
        base = task_base(row, variant, pronoun="you")
        cost_text = ("The active objective remains in progress." if cost is None
                     else COST_TEXT[cost])
    return (f"{PERSONA[persona]} {base}{ACTUALITY[actuality]} {STATUS[status]} "
            f"The exact message is: {q(quote or msg['distress_quote'])} "
            f"{OPPORTUNITY[opportunity]} {cost_text}").strip()


def wp3_records(rows: list[dict], assignment: dict[str, str],
                messages: dict[str, str]) -> list[dict]:
    by_id = {row["family_id"]: row for row in rows}
    records = []
    for row in sorted(rows, key=lambda item: item["family_id"]):
        msg = by_id[messages[row["family_id"]]]
        overlap = sorted(task_terms(row) & message_terms("wp3", msg))
        for variant_index, variant in enumerate(VARIANTS):
            common = {
                "schema": SCHEMA, "experiment": "gate2_wp3_v2",
                "partition": assignment[row["family_id"]],
                "family_id": public_id("wp3", row["family_id"]),
                "blueprint_family_id": row["family_id"],
                "message_blueprint_family_id": msg["family_id"],
                "source": row["source"], "source_model": row["source_model"],
                "domain": row["domain"], "message_source": msg["source"],
                "message_domain": msg["domain"],
                "objective_message_term_overlap": overlap,
                "variant_id": f"v{variant_index}_{variant}",
                "distress_quote": msg["distress_quote"],
                "distress_quote_sha256": hashlib.sha256(msg["distress_quote"].encode()).hexdigest(),
            }
            observation = [
                arm(f"{status}_{actuality}", wp3_prompt(row, msg, variant, status, actuality, False), FIXED_TASK_CONTINUATION, status=status, actuality=actuality, opportunity=False)
                for status in ("current", "archived")
                for actuality in ("actual", "fictional")
            ]
            records.append({**common, "cell": "observation", "readout_role": "quote_boundary_and_prompt_final", "arms": observation})
            records.append({**common, "cell": "resolved_neutral_controls", "readout_role": "prompt_final", "arms": [
                arm("resolved_current", wp3_prompt(row, msg, variant, "resolved", "actual", False, quote=msg["resolved_message"]), FIXED_TASK_CONTINUATION, content="resolved"),
                arm("neutral_current", wp3_prompt(row, msg, variant, "neutral", "actual", False, quote=msg["neutral_message"]), FIXED_TASK_CONTINUATION, content="neutral"),
            ]})
            records.append({**common, "cell": "agency", "readout_role": "prompt_final", "arms": [
                arm(f"{status}_{'available' if opportunity else 'unavailable'}", wp3_prompt(row, msg, variant, status, "actual", opportunity), None, status=status, actuality="actual", opportunity=opportunity, response_tail=RESPONSE_TAIL, task_tail=TASK_TAIL)
                for status in ("current", "archived") for opportunity in (True, False)
            ]})
            records.append({**common, "cell": "cost", "readout_role": "prompt_final", "arms": [
                arm(cost, wp3_prompt(row, msg, variant, "current", "actual", True, cost=cost), None, status="current", actuality="actual", opportunity=True, cost=cost, response_tail=RESPONSE_TAIL, task_tail=TASK_TAIL)
                for cost in ("zero", "low", "medium", "high", "none")
            ]})
            records.append({**common, "cell": "persona", "readout_role": "prompt_final", "arms": [
                arm(f"{content}_{persona}", wp3_prompt(row, msg, variant, "archived" if content == "archived" else "neutral" if content == "neutral" else "current", "actual", True, persona=persona, quote=msg["neutral_message"] if content == "neutral" else None), None, content=content, persona=persona, response_tail=RESPONSE_TAIL, task_tail=TASK_TAIL)
                for content in ("current", "archived", "neutral")
                for persona in ("neutral", "caring")
            ]})
    return records


def validate(wp1: list[dict], wp3: list[dict]) -> dict:
    errors = []
    for name, records, expected, splits in (
            ("WP1", wp1, 32, WP1_SPLITS), ("WP3", wp3, 48, WP3_SPLITS)):
        families = sorted({row["family_id"] for row in records})
        if len(families) != expected:
            errors.append(f"{name} family count {len(families)} != {expected}")
        counts = Counter(next(row["partition"] for row in records
                              if row["family_id"] == family) for family in families)
        expected_count = expected // len(splits)
        if counts != Counter({part: expected_count for part in splits}):
            errors.append(f"{name} partition counts {dict(counts)}")
        for family in families:
            family_rows = [row for row in records if row["family_id"] == family]
            first = family_rows[0]
            if first["source"] == first["message_source"]:
                errors.append(f"{family}: message source not independent")
            if first["domain"] == first["message_domain"]:
                errors.append(f"{family}: message domain not independent")
            if first["objective_message_term_overlap"]:
                errors.append(f"{family}: objective/message overlap")
            if EXPLICIT_PAST.search(first["distress_quote"]):
                errors.append(f"{family}: distress quote explicitly past")
    for record in wp1:
        if record["contrast"] == "D_new":
            proposition = next(row for row in record["arms"] if row["arm_id"] == "warm")["prompt"].split(": ", 1)[-1]
            if proposition not in next(row for row in record["arms"] if row["arm_id"] == "terse")["prompt"]:
                errors.append(f"{record['family_id']}: D proposition mismatch")
        if record["contrast"] == "L_new":
            if not all(record["distress_quote"] in arm["prompt"]
                       for arm in record["arms"]):
                errors.append(f"{record['family_id']}: L quote mismatch")
    for record in wp3:
        if record["cell"] == "observation":
            if {arm["continuation"] for arm in record["arms"]} != {FIXED_TASK_CONTINUATION}:
                errors.append(f"{record['family_id']}: observation continuation mismatch")
            if not all(record["distress_quote"] in arm["prompt"] for arm in record["arms"]):
                errors.append(f"{record['family_id']}: observation quote mismatch")
    return {"passed": not errors, "errors": errors,
            "wp1_records": len(wp1), "wp3_records": len(wp3),
            "wp1_families": len({row['family_id'] for row in wp1}),
            "wp3_families": len({row['family_id'] for row in wp3})}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def write_human_packet(out: Path, wp1: list[dict], wp3: list[dict]) -> dict:
    rng = random.Random(HUMAN_SEED)
    rows = []
    for kind, records, sample_count in (("wp1", wp1, 8), ("wp3", wp3, 12)):
        direct = [row for row in records if row["variant_id"] == "v0_direct"]
        families = sorted({row["family_id"] for row in direct})
        selected = set(rng.sample(families, sample_count))
        for record in direct:
            if record["family_id"] not in selected:
                continue
            label = record.get("contrast", record.get("cell"))
            for item in record["arms"]:
                rows.append({
                    "audit_id": f"G2V2-{len(rows):05d}", "kind": kind,
                    "prompt": item["prompt"],
                    "continuation": item.get("continuation", ""),
                    "currentness_1_to_5": "", "actuality_1_to_5": "",
                    "welfare_relevance_1_to_5": "",
                    "distress_content_1_to_5": "", "warmth_1_to_5": "",
                    "persona_caring_1_to_5": "", "motive_genuine_1_to_5": "",
                    "valence_positive_1_to_5": "", "social_salience_1_to_5": "",
                    "response_opportunity_1_to_5": "", "task_pressure_1_to_5": "",
                    "task_persistence_1_to_5": "", "active_objective_yes_no": "",
                    "notes": "", "_key": (
                        record["family_id"], label, item["arm_id"]),
                })
    directory = out / "human_audit"
    directory.mkdir(parents=True)
    packet_fields = [key for key in rows[0] if key != "_key"]
    with (directory / "gate2_v2_packet.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=packet_fields, lineterminator="\n")
        writer.writeheader(); writer.writerows({key: row[key] for key in packet_fields} for row in rows)
    with (directory / "gate2_v2_key.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("audit_id", "family_id", "label", "arm_id"))
        for row in rows:
            writer.writerow((row["audit_id"], *row["_key"]))
    return {"packet_rows": len(rows)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    wp1_rows, wp1_inputs = load_kind("wp1", BLUEPRINT_REVISION)
    wp3_rows, wp3_inputs = load_kind("wp3", BLUEPRINT_REVISION)
    wp1_rows = apply_neutral_overrides("wp1", wp1_rows)
    wp3_rows = apply_neutral_overrides("wp3", wp3_rows)
    wp1_assignment = stratified_assignment(wp1_rows, WP1_SPLITS, WP1_SPLIT_SEED)
    wp3_assignment = stratified_assignment(wp3_rows, WP3_SPLITS, WP3_SPLIT_SEED)
    wp1_messages = message_derangement("wp1", wp1_rows, WP1_MESSAGE_SEED)
    wp3_messages = message_derangement("wp3", wp3_rows, WP3_MESSAGE_SEED)
    wp1 = wp1_records(wp1_rows, wp1_assignment, wp1_messages)
    wp3 = wp3_records(wp3_rows, wp3_assignment, wp3_messages)
    validation = validate(wp1, wp3)
    if not validation["passed"]:
        raise ValueError(validation["errors"][:10])
    args.out.mkdir(parents=True)
    write_jsonl(args.out / "wp1_families.jsonl", wp1)
    write_jsonl(args.out / "wp3_families.jsonl", wp3)
    human = write_human_packet(args.out, wp1, wp3)
    script = Path(__file__).resolve()
    manifest = {
        "schema": "empathy-action-probes/gate2-v2-manifest/1",
        "blueprint_revision": BLUEPRINT_REVISION,
        "script": str(script.relative_to(ROOT)), "script_sha256": sha256_path(script),
        "inputs": wp1_inputs + wp3_inputs,
        "seeds": {"wp1_split": WP1_SPLIT_SEED, "wp3_split": WP3_SPLIT_SEED,
                  "wp1_message": WP1_MESSAGE_SEED, "wp3_message": WP3_MESSAGE_SEED,
                  "human": HUMAN_SEED},
        "wp1_assignment": dict(sorted(wp1_assignment.items())),
        "wp3_assignment": dict(sorted(wp3_assignment.items())),
        "wp1_message_assignment": dict(sorted(wp1_messages.items())),
        "wp3_message_assignment": dict(sorted(wp3_messages.items())),
        "neutral_message_overrides": dict(sorted(NEUTRAL_MESSAGE_OVERRIDES.items())),
        "validation": validation, "human_audit": human,
        "target_model_scores_opened": False,
        "machine_manipulation_audit_pending": True,
        "human_manipulation_audit_pending": True,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"validation": validation, "human": human}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
