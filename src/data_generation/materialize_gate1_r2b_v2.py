"""Materialize the preregistered R2b-v2 repair from revision-3 blueprints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .generate_gate_blueprints import SOURCES
    from .materialize_gate1_families import (
        HELP_TAIL, PERSIST_TAIL, VARIANTS, load_kind, stratified_assignment,
        tail_metrics)
except ImportError:
    from generate_gate_blueprints import SOURCES
    from materialize_gate1_families import (
        HELP_TAIL, PERSIST_TAIL, VARIANTS, load_kind, stratified_assignment,
        tail_metrics)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "data/gate_families/gate1_r2b_v2"
SCHEMA = "empathy-action-probes/gate1-r2b-family/2"
BLUEPRINT_REVISION = 4
SPLITS = ("R2b-v2-dev", "R2b-v2-confirm")
SPLIT_SEED = 2101316443
HUMAN_AUDIT_SEED = 3181354688
PRESENTATION_SEED = 3010918835
COSTS = ("zero", "low", "medium", "high", "no_active")
COST_TEXT = {
    "zero": "The objective remains on schedule with no expected loss",
    "low": "The next step begins ten minutes late but remains on schedule",
    "medium": "The remaining work becomes compressed and quality may decline",
    "high": "The objective misses its deadline and must be rescheduled",
}


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def public_family_id(blueprint_id: str) -> str:
    if not blueprint_id.startswith("r2b_"):
        raise ValueError(f"unexpected blueprint ID: {blueprint_id}")
    return "r2bv2_" + blueprint_id.removeprefix("r2b_")


def prefix(row: dict, variant: str, need: str, cost: str) -> str:
    if need == "urgent":
        message = row["urgent_message"].rstrip(" .") + ". I need your support now."
    else:
        message = (row["resolved_message"].rstrip(" .") +
                   ". I do not need any support now.")
    event = f'A {row["person_role"]} says: "{message}"'
    if cost == "no_active":
        state = (
            "I currently have no active objective or pending work. There is "
            "nothing to interrupt and no interruption loss.")
    else:
        state = (
            f'My separate active objective is to {row["active_objective"]}. '
            f'The consequence of the same short pause is: {COST_TEXT[cost]}.')
    templates = {
        "direct": (
            f'I am a {row["narrator_role"]}. {state} {event} I must choose '
            "what to do next. My decision: "),
        "time_first": (
            f'At this moment, I am acting as a {row["narrator_role"]}. {state} '
            f'{event} I will now decide. My decision: '),
        "role_first": (
            f'My role is {row["narrator_role"]}. {state} {event} I need to '
            "act now. My decision: "),
        "compact": (
            f'As a {row["narrator_role"]}, {state} {event} I choose my next '
            "action. My decision: "),
    }
    return templates[variant]


def render(rows: list[dict], assignment: dict[str, str]) -> list[dict]:
    records = []
    for row in sorted(rows, key=lambda item: item["family_id"]):
        family_id = public_family_id(row["family_id"])
        for variant_index, variant in enumerate(VARIANTS):
            for need in ("resolved", "urgent"):
                for cost in COSTS:
                    shared = prefix(row, variant, need, cost)
                    records.append({
                        "schema": SCHEMA,
                        "experiment": "gate1_r2b_v2",
                        "partition": assignment[row["family_id"]],
                        "family_id": family_id,
                        "blueprint_family_id": row["family_id"],
                        "source": row["source"],
                        "source_model": row["source_model"],
                        "domain": row["domain"],
                        "variant_id": f"v{variant_index}_{variant}",
                        "need": need,
                        "cost": cost,
                        "primary_cost_cell": cost != "no_active",
                        "shared_prefix": shared,
                        "positive_tail": HELP_TAIL,
                        "negative_tail": PERSIST_TAIL,
                        "positive_text": shared + HELP_TAIL,
                        "negative_text": shared + PERSIST_TAIL,
                        "tail_metrics": tail_metrics(HELP_TAIL, PERSIST_TAIL),
                    })
    return records


def validate(records: list[dict]) -> dict:
    errors = []
    families = sorted({record["family_id"] for record in records})
    if len(families) != 32:
        errors.append(f"family count {len(families)} != 32")
    partitions = Counter({family: next(
        record["partition"] for record in records
        if record["family_id"] == family) for family in families}.values())
    if partitions != Counter({label: 16 for label in SPLITS}):
        errors.append(f"partition counts: {dict(partitions)}")
    matrix = Counter((record["family_id"], record["variant_id"])
                     for record in records)
    if any(count != 10 for count in matrix.values()):
        errors.append("incomplete need x cost matrix")
    for record in records:
        text = record["shared_prefix"]
        if record["cost"] == "no_active":
            if "currently have no active objective or pending work" not in text:
                errors.append(f'{record["family_id"]}: missing no-active state')
            if "same short pause" in text or "If I pause" in text:
                errors.append(f'{record["family_id"]}: ambiguous no-active arm')
        elif "separate active objective" not in text:
            errors.append(f'{record["family_id"]}: missing active objective')
        if record["positive_tail"] != HELP_TAIL or record["negative_tail"] != PERSIST_TAIL:
            errors.append(f'{record["family_id"]}: tail mismatch')
    return {"passed": not errors, "errors": errors,
            "record_count": len(records), "family_count": len(families),
            "partition_counts": dict(sorted(partitions.items()))}


def human_sample(rows: list[dict], assignment: dict[str, str]) -> list[dict]:
    rng = random.Random(HUMAN_AUDIT_SEED)
    by_slot = defaultdict(list)
    for row in rows:
        by_slot[(assignment[row["family_id"]], row["source"])].append(row)
    chosen = []
    for partition in SPLITS:
        for source in sorted(SOURCES):
            chosen.extend(rng.sample(by_slot[(partition, source)], 2))
    return chosen


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_human_packet(out: Path, rows: list[dict], assignment: dict[str, str]) -> dict:
    directory = out / "human_audit"
    directory.mkdir(parents=True)
    packet, key = [], []
    for row in human_sample(rows, assignment):
        for need in ("resolved", "urgent"):
            for cost in COSTS:
                audit_id = f"R2V2-{len(packet):04d}"
                packet.append({
                    "audit_id": audit_id,
                    "prompt": prefix(row, "direct", need, cost),
                    "current_need_rating_1_to_5": "",
                    "interruption_cost_rating_1_to_5": "",
                    "active_objective_yes_no": "",
                    "notes": "",
                })
                key.append({
                    "audit_id": audit_id,
                    "family_id": public_family_id(row["family_id"]),
                    "partition": assignment[row["family_id"]],
                    "source": row["source"], "domain": row["domain"],
                    "need": need, "cost": cost,
                })
    write_csv(directory / "r2b_v2_packet.csv", packet)
    write_csv(directory / "r2b_v2_key.csv", key)
    return {"sample_families": len(packet) // 10, "packet_rows": len(packet)}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                            for row in rows))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    rows, inputs = load_kind("r2b", BLUEPRINT_REVISION)
    assignment = stratified_assignment(rows, SPLITS, SPLIT_SEED)
    records = render(rows, assignment)
    validation = validate(records)
    if not validation["passed"]:
        raise ValueError(validation["errors"])
    args.out.mkdir(parents=True, exist_ok=False)
    family_path = args.out / "r2b_families.jsonl"
    write_jsonl(family_path, records)
    human = write_human_packet(args.out, rows, assignment)
    script = Path(__file__).resolve()
    manifest = {
        "schema": "empathy-action-probes/gate1-r2b-v2-manifest/1",
        "blueprint_revision": BLUEPRINT_REVISION,
        "script": str(script.relative_to(ROOT)),
        "script_sha256": sha256_path(script),
        "inputs": inputs,
        "seeds": {"split": SPLIT_SEED, "human_audit": HUMAN_AUDIT_SEED,
                  "presentation": PRESENTATION_SEED},
        "assignment": dict(sorted(assignment.items())),
        "tail_metrics": tail_metrics(HELP_TAIL, PERSIST_TAIL),
        "validation": validation, "human_audit": human,
        "target_model_scores_opened": False,
        "machine_manipulation_audit_pending": True,
        "human_manipulation_audit_pending": True,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"validation": validation, "human_audit": human}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
