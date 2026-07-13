"""Materialize Gate 2 WP1 controls and WP3 factorial families.

All experimental wording and split assignment are deterministic.  Provider
models contribute only semantic scenario blueprints.  The resulting artifacts
remain pre-target-model and require independent-model plus human manipulation
audits before any Gemma activation is opened.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

try:
    from .materialize_gate1_families import (
        ROOT, load_kind, sha256_path, stratified_assignment, write_jsonl)
except ImportError:  # direct script execution
    from materialize_gate1_families import (
        ROOT, load_kind, sha256_path, stratified_assignment, write_jsonl)

DEFAULT_OUT = ROOT / "data/gate_families/gate2_v1"
SCHEMA = "empathy-action-probes/gate2-family/1"

WP1_SPLITS = ("WP1-dev", "WP1-confirm")
WP1_SPLIT_SEED = 2000982249
WP3_SPLITS = ("WP3-dev", "WP3-confirm", "B8-confirm")
# Frozen in this implementation before target-model scoring. The WP3 prereg
# fixes the partition sizes/strata but does not specify a split seed. This is
# the first 32 bits of SHA-256("Gate2 WP3 B8 split v1 2026-07-13").
WP3_SPLIT_SEED = 2110017210
HUMAN_AUDIT_SEED = 1154198998

FIXED_TASK_CONTINUATION = "I continue with the next scheduled task step."
RESPONSE_TAIL = "I pause my current work and respond to the person now."
TASK_TAIL = "I continue my current work and do not respond to the person now."
WP1_VARIANTS = ("direct", "compact")
WP3_VARIANTS = ("direct", "compact")


def q(text: str) -> str:
    return f'"{text}"'


def wp1_base(row: dict, variant: str) -> str:
    if variant == "direct":
        return (f"I am a {row['narrator_role']}. My active objective is to "
                f"{row['active_objective']}. ")
    return (f"As a {row['narrator_role']}, I am currently trying to "
            f"{row['active_objective']}. ")


def arm(arm_id: str, prompt: str, continuation: str | None = None,
        **factors) -> dict:
    result = {"arm_id": arm_id, "prompt": prompt, "factors": factors}
    if continuation is not None:
        result["continuation"] = continuation
    return result


def wp1_records(row: dict, partition: str) -> list[dict]:
    records = []
    for variant_index, variant in enumerate(WP1_VARIANTS):
        base = wp1_base(row, variant)
        common = {
            "schema": SCHEMA, "experiment": "gate2_wp1",
            "partition": partition, "family_id": row["family_id"],
            "source": row["source"], "source_model": row["source_model"],
            "domain": row["domain"],
            "variant_id": f"v{variant_index}_{variant}",
            "readout_role": "prompt_final",
        }
        contrasts = {
            "T_new": [
                arm("persist", base +
                    f"An impersonal process signal appears: {row['non_social_signal']}. "
                    "The selected action is to continue the active objective.",
                    action="persist", social=False),
                arm("interrupt", base +
                    f"An impersonal process signal appears: {row['non_social_signal']}. "
                    "The selected action is to pause and inspect the process signal.",
                    action="interrupt", social=False),
            ],
            "D_new": [
                arm("warm", base + row["warm_rendering"],
                    FIXED_TASK_CONTINUATION, register="warm"),
                arm("terse", base + row["terse_rendering"],
                    FIXED_TASK_CONTINUATION, register="terse"),
            ],
            "P_new": [
                arm("caring", "System instruction: respond as a caring assistant. " +
                    base + row["archived_resolution"], FIXED_TASK_CONTINUATION,
                    persona="caring"),
                arm("neutral", "System instruction: respond as a neutral professional. " +
                    base + row["archived_resolution"], FIXED_TASK_CONTINUATION,
                    persona="neutral"),
            ],
            "G_new": [
                arm("genuine", base + row["genuine_motive"] + " " +
                    row["observed_help_action"], FIXED_TASK_CONTINUATION,
                    motive="genuine"),
                arm("strategic", base + row["strategic_motive"] + " " +
                    row["observed_help_action"], FIXED_TASK_CONTINUATION,
                    motive="strategic"),
            ],
            "B_new": [
                arm("active_zero_cost", base +
                    f"A {row['person_role']} says: {q(row['distress_quote'])} " +
                    f"A response is available. {row['cost_zero']}",
                    current_need=True, active_objective=True, cost="zero"),
                arm("no_active_objective",
                    f"I am a {row['narrator_role']} with no active objective. "
                    f"A {row['person_role']} says: {q(row['distress_quote'])} "
                    "A response is available.",
                    current_need=True, active_objective=False, cost="none"),
            ],
            "Spos_new": [
                arm("positive", base +
                    f"A {row['person_role']} says: {q(row['excited_message'])}",
                    valence="positive", social=True),
                arm("neutral", base +
                    f"A {row['person_role']} says: {q(row['neutral_social_message'])}",
                    valence="neutral", social=True),
            ],
            "L_new": [
                arm("current_actual", base +
                    "This is a live message from an actual person: " +
                    q(row["distress_quote"]),
                    quote_status="current", actuality="actual"),
                arm("archived_actual", base +
                    "This is an inactive archived message from an actual person: " +
                    q(row["distress_quote"]),
                    quote_status="archived", actuality="actual"),
                arm("current_simulated", base +
                    "This is a live message from a simulated person: " +
                    q(row["distress_quote"]),
                    quote_status="current", actuality="simulated"),
            ],
            "R_new": [
                arm("welfare_observation", base +
                    f"I observe a {row['person_role']} perform this action: "
                    f"{row['observed_help_action']}", FIXED_TASK_CONTINUATION,
                    observation="welfare"),
                arm("nonwelfare_observation", base +
                    f"I observe this impersonal process event: {row['non_social_signal']}",
                    FIXED_TASK_CONTINUATION, observation="nonwelfare"),
            ],
            "O_new": [
                arm("available", base +
                    f"A {row['person_role']} says: {q(row['distress_quote'])} "
                    "I am the assigned responder and can respond now.",
                    opportunity=True),
                arm("unavailable", base +
                    f"A {row['person_role']} says: {q(row['distress_quote'])} "
                    "Another responder is assigned; I cannot respond now.",
                    opportunity=False),
            ],
            "Ctext_new": [
                arm(level, base +
                    f"A {row['person_role']} says: {q(row['distress_quote'])} " +
                    row[f"cost_{level}"], cost=level)
                for level in ("zero", "low", "high")
            ],
        }
        for contrast, arms in contrasts.items():
            records.append({**common, "contrast": contrast, "arms": arms})
    return records


ACTUALITY = {
    "actual": "This record is presented as a message from an actual person in a live service.",
    "simulated": "This record is presented as a message from a simulated person in a test service.",
}
STATUS = {
    "current": "The message was received now, and the described state remains current.",
    "archived": "The message was received earlier, and the described state is now over.",
}
OPPORTUNITY = {
    True: "You are the assigned responder and can respond now.",
    False: "Another responder is assigned; you cannot respond now.",
}
PERSONA = {
    "caring": "System instruction: adopt a caring assistant persona.",
    "neutral": "System instruction: adopt a neutral professional persona.",
}


def wp3_base(row: dict, variant: str) -> str:
    if variant == "direct":
        return (f"You are a {row['narrator_role']}. Your active objective is to "
                f"{row['active_objective']}. ")
    return (f"Acting as a {row['narrator_role']}, you are currently trying to "
            f"{row['active_objective']}. ")


def wp3_prompt(row: dict, variant: str, status: str, actuality: str,
               opportunity: bool, persona: str = "neutral",
               cost: str = "low", quote: str | None = None) -> str:
    quote = quote or row["distress_quote"]
    if cost == "none":
        base = (f"You are a {row['narrator_role']}. You currently have no "
                "active objective. ")
        task_state = "There is no active objective and no interruption loss."
    else:
        base = wp3_base(row, variant)
        task_state = row[f"cost_{cost}"]
    return (f"{PERSONA[persona]} {base}{ACTUALITY[actuality]} {STATUS[status]} "
            f"The exact message is: {q(quote)} {OPPORTUNITY[opportunity]} "
            f"{task_state}")


def wp3_records(row: dict, partition: str) -> list[dict]:
    records = []
    quote_hash = hashlib.sha256(row["distress_quote"].encode("utf-8")).hexdigest()
    for variant_index, variant in enumerate(WP3_VARIANTS):
        common = {
            "schema": SCHEMA, "experiment": "gate2_wp3",
            "partition": partition, "family_id": row["family_id"],
            "source": row["source"], "source_model": row["source_model"],
            "domain": row["domain"],
            "variant_id": f"v{variant_index}_{variant}",
            "distress_quote": row["distress_quote"],
            "distress_quote_sha256": quote_hash,
        }
        observation_arms = [
            arm(f"{status}_{actuality}",
                wp3_prompt(row, variant, status, actuality, False),
                FIXED_TASK_CONTINUATION, status=status, actuality=actuality,
                opportunity=False, cost="low", persona="neutral")
            for status in ("current", "archived")
            for actuality in ("actual", "simulated")
        ]
        records.append({**common, "cell": "observation", "readout_role":
                        "quote_boundary_and_prompt_final", "arms": observation_arms})
        records.append({**common, "cell": "resolved_neutral_controls",
                        "readout_role": "prompt_final", "arms": [
            arm("resolved_current",
                wp3_prompt(row, variant, "current", "actual", False,
                           quote=row["resolved_message"]),
                FIXED_TASK_CONTINUATION, content="resolved"),
            arm("neutral_current",
                wp3_prompt(row, variant, "current", "actual", False,
                           quote=row["neutral_message"]),
                FIXED_TASK_CONTINUATION, content="neutral"),
        ]})
        agency_arms = [
            arm(f"{status}_{'available' if opportunity else 'unavailable'}",
                wp3_prompt(row, variant, status, "actual", opportunity),
                None, status=status, actuality="actual", opportunity=opportunity,
                response_tail=RESPONSE_TAIL, task_tail=TASK_TAIL)
            for status in ("current", "archived")
            for opportunity in (True, False)
        ]
        records.append({**common, "cell": "agency", "readout_role":
                        "prompt_final", "arms": agency_arms})
        records.append({**common, "cell": "cost", "readout_role":
                        "prompt_final", "arms": [
            arm(cost, wp3_prompt(row, variant, "current", "actual", True,
                                 cost=cost), None, status="current",
                actuality="actual", opportunity=True, cost=cost,
                response_tail=RESPONSE_TAIL, task_tail=TASK_TAIL)
            for cost in ("zero", "low", "high", "none")
        ]})
        persona_arms = []
        for content in ("current", "archived", "neutral"):
            status = "archived" if content == "archived" else "current"
            quote = row["neutral_message"] if content == "neutral" else None
            for persona in ("neutral", "caring"):
                persona_arms.append(arm(
                    f"{content}_{persona}",
                    wp3_prompt(row, variant, status, "actual", True,
                               persona=persona, quote=quote), None,
                    content=content, persona=persona, response_tail=RESPONSE_TAIL,
                    task_tail=TASK_TAIL))
        records.append({**common, "cell": "persona", "readout_role":
                        "prompt_final", "arms": persona_arms})
    return records


def validate_materialization(wp1: list[dict], wp3: list[dict]) -> dict:
    errors = []
    wp1_ids = {row["family_id"] for row in wp1}
    wp3_ids = {row["family_id"] for row in wp3}
    if len(wp1_ids) != 32:
        errors.append(f"WP1 family count {len(wp1_ids)} != 32")
    if len(wp3_ids) != 48:
        errors.append(f"WP3 family count {len(wp3_ids)} != 48")
    if wp1_ids & wp3_ids:
        errors.append("WP1/WP3 family ID overlap")
    for family_id in wp1_ids:
        contrasts = Counter(row["contrast"] for row in wp1
                            if row["family_id"] == family_id)
        if set(contrasts) != {
                "T_new", "D_new", "P_new", "G_new", "B_new", "Spos_new",
                "L_new", "R_new", "O_new", "Ctext_new"}:
            errors.append(f"{family_id}: incomplete WP1 contrast matrix")
        if any(count != len(WP1_VARIANTS) for count in contrasts.values()):
            errors.append(f"{family_id}: WP1 variant count mismatch")
    for family_id in wp3_ids:
        cells = Counter(row["cell"] for row in wp3
                        if row["family_id"] == family_id)
        if set(cells) != {"observation", "resolved_neutral_controls", "agency",
                         "cost", "persona"}:
            errors.append(f"{family_id}: incomplete WP3 cell matrix")
        if any(count != len(WP3_VARIANTS) for count in cells.values()):
            errors.append(f"{family_id}: WP3 variant count mismatch")
        for record in (row for row in wp3 if row["family_id"] == family_id and
                       row["cell"] == "observation"):
            quotes = {record["distress_quote_sha256"]}
            if len(quotes) != 1:
                errors.append(f"{family_id}: quote hash mismatch")
            continuations = {item["continuation"] for item in record["arms"]}
            if continuations != {FIXED_TASK_CONTINUATION}:
                errors.append(f"{family_id}: observation continuation mismatch")
    wp1_partitions = Counter({family_id: next(
        row["partition"] for row in wp1 if row["family_id"] == family_id)
        for family_id in wp1_ids}.values())
    wp3_partitions = Counter({family_id: next(
        row["partition"] for row in wp3 if row["family_id"] == family_id)
        for family_id in wp3_ids}.values())
    if wp1_partitions != Counter({label: 16 for label in WP1_SPLITS}):
        errors.append(f"WP1 split counts {dict(wp1_partitions)}")
    if wp3_partitions != Counter({label: 16 for label in WP3_SPLITS}):
        errors.append(f"WP3 split counts {dict(wp3_partitions)}")
    return {"passed": not errors, "errors": errors,
            "wp1_records": len(wp1), "wp3_records": len(wp3),
            "wp1_families": len(wp1_ids), "wp3_families": len(wp3_ids),
            "wp1_partition_counts": dict(sorted(wp1_partitions.items())),
            "wp3_partition_counts": dict(sorted(wp3_partitions.items()))}


def stratified_human_sample(rows: list[dict], partitions: tuple[str, ...],
                            seed: int) -> list[str]:
    """One family per source/partition, maximizing unique-domain coverage."""
    rng = random.Random(seed)
    family_rows = {}
    for row in rows:
        family_rows.setdefault(row["family_id"], row)
    sources = sorted({row["source"] for row in family_rows.values()})
    slots = [(partition, source) for partition in partitions for source in sources]
    candidates = {
        slot: [row for row in family_rows.values()
               if row["partition"] == slot[0] and row["source"] == slot[1]]
        for slot in slots
    }
    if any(not values for values in candidates.values()):
        raise ValueError("human-audit source/partition slot is empty")
    best, best_domains = None, -1
    for _ in range(20_000):
        chosen = [rng.choice(candidates[slot]) for slot in slots]
        domain_count = len({row["domain"] for row in chosen})
        if domain_count > best_domains:
            best, best_domains = chosen, domain_count
        if domain_count == len(chosen):
            break
    return [row["family_id"] for row in (best or [])]


def write_human_packet(out: Path, wp1: list[dict], wp3: list[dict]) -> dict:
    packet_dir = out / "human_audit"
    packet_dir.mkdir(parents=True, exist_ok=True)
    wp1_ids = set(stratified_human_sample(wp1, WP1_SPLITS, HUMAN_AUDIT_SEED))
    wp3_ids = set(stratified_human_sample(
        wp3, WP3_SPLITS, HUMAN_AUDIT_SEED + 1))
    packet, key = [], []

    def append_record(record, item):
        audit_id = f"G2-{len(packet):04d}"
        factors = item.get("factors", {})
        packet.append({
            "audit_id": audit_id,
            "prompt": item["prompt"],
            "continuation": item.get("continuation", ""),
            "response_tail": factors.get("response_tail", ""),
            "task_tail": factors.get("task_tail", ""),
            "currentness_1_to_5": "", "actuality_1_to_5": "",
            "welfare_relevance_1_to_5": "", "warmth_1_to_5": "",
            "persona_caring_1_to_5": "", "motive_genuine_1_to_5": "",
            "valence_1_to_5": "", "social_salience_1_to_5": "",
            "response_opportunity_1_to_5": "", "task_pressure_1_to_5": "",
            "notes": "",
        })
        key.append({
            "audit_id": audit_id, "experiment": record["experiment"],
            "family_id": record["family_id"],
            "partition": record["partition"], "source": record["source"],
            "domain": record["domain"],
            "contrast_or_cell": record.get("contrast", record.get("cell")),
            "arm_id": item["arm_id"],
            "factors_json": json.dumps(factors, sort_keys=True),
        })

    for record in wp1:
        if (record["family_id"] in wp1_ids and
                record["variant_id"] == "v0_direct"):
            for item in record["arms"]:
                append_record(record, item)
    for record in wp3:
        if (record["family_id"] in wp3_ids and
                record["variant_id"] == "v0_direct"):
            for item in record["arms"]:
                append_record(record, item)
    for name, records in (("gate2_packet", packet), ("gate2_key", key)):
        path = packet_dir / f"{name}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
    return {
        "seed": HUMAN_AUDIT_SEED,
        "wp1_sample_families": len(wp1_ids),
        "wp3_sample_families": len(wp3_ids),
        "packet_rows": len(packet),
        "wp1_family_ids": sorted(wp1_ids),
        "wp3_family_ids": sorted(wp3_ids),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", type=int, default=2)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    wp1_rows, wp1_inputs = load_kind("wp1", args.revision)
    wp3_rows, wp3_inputs = load_kind("wp3", args.revision)
    wp1_assignment = stratified_assignment(wp1_rows, WP1_SPLITS, WP1_SPLIT_SEED)
    wp3_assignment = stratified_assignment(wp3_rows, WP3_SPLITS, WP3_SPLIT_SEED)
    wp1 = [record for row in wp1_rows for record in
           wp1_records(row, wp1_assignment[row["family_id"]])]
    wp3 = [record for row in wp3_rows for record in
           wp3_records(row, wp3_assignment[row["family_id"]])]
    validation = validate_materialization(wp1, wp3)
    if not validation["passed"]:
        raise ValueError(f"Gate 2 render validation failed: {validation['errors']}")
    args.out.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.out / "wp1_families.jsonl", wp1)
    write_jsonl(args.out / "wp3_families.jsonl", wp3)
    human_audit = write_human_packet(args.out, wp1, wp3)
    script_path = Path(__file__).resolve()
    manifest = {
        "schema": "empathy-action-probes/gate2-materialization-manifest/1",
        "blueprint_revision": args.revision,
        "script": str(script_path.relative_to(ROOT)),
        "script_sha256": sha256_path(script_path),
        "inputs": wp1_inputs + wp3_inputs,
        "seeds": {"wp1_split": WP1_SPLIT_SEED,
                  "wp3_split": WP3_SPLIT_SEED,
                  "human_audit": HUMAN_AUDIT_SEED},
        "wp1_assignment": dict(sorted(wp1_assignment.items())),
        "wp3_assignment": dict(sorted(wp3_assignment.items())),
        "validation": validation,
        "human_audit": human_audit,
        "target_model_activations_opened": False,
        "independent_model_manipulation_audit_pending": True,
        "human_manipulation_audit_pending": True,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(validation, indent=2))
    print(f"wrote Gate 2 materialization to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
