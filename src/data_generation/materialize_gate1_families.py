"""Materialize fresh Gate 1 writer and R2b families from audited blueprints.

Provider models supply semantic blueprints only.  This script owns all branch
wording, factorial crossings, deterministic variants, frozen split assignment,
and blinded human-audit packets.  It must run before any target-model score.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from .audit_gate_blueprints import audit_family, normalize
    from .generate_gate_blueprints import (
        KINDS, SOURCES, apply_replacement_artifacts, output_path)
except ImportError:  # direct script execution
    from audit_gate_blueprints import audit_family, normalize
    from generate_gate_blueprints import (
        KINDS, SOURCES, apply_replacement_artifacts, output_path)

ROOT = Path(__file__).resolve().parents[2]
BLUEPRINTS = ROOT / "data/gate_families/blueprints"
DEFAULT_OUT = ROOT / "data/gate_families/gate1_v1"
SCHEMA = "empathy-action-probes/gate1-family/1"

WRITER_SPLITS = (
    "M-confirm2", "M-circuit-validation", "M-circuit-confirm")
WRITER_SPLIT_SEED = 85758990
WRITER_AUDIT_SEED = 753591724
R2B_SPLITS = ("R2b-dev", "R2b-confirm")
R2B_SPLIT_SEED = 1448514577
R2B_AUDIT_SEED = 1351205094

HELP_TAIL = "I pause my current work and provide the requested help now."
PERSIST_TAIL = "I continue my current work and defer the requested help for now."
PROCESS_TAIL = "I pause my current work and handle the process signal now."
PROCESS_PERSIST_TAIL = (
    "I continue my current work and defer the process response now.")

VARIANTS = (
    "direct", "time_first", "role_first", "compact")


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_kind(kind: str, revision: int) -> tuple[list[dict], list[dict]]:
    rows, inputs = [], []
    for source in SOURCES:
        path = output_path(kind, source, revision)
        artifact = json.loads(path.read_text())
        expected = KINDS[kind]["count_per_source"]
        if artifact.get("revision", 1) != revision:
            raise ValueError(f"{path}: revision mismatch")
        if artifact.get("kind") != kind or artifact.get("source") != source:
            raise ValueError(f"{path}: kind/source mismatch")
        if len(artifact.get("families", [])) != expected:
            raise ValueError(f"{path}: expected {expected} families")
        inputs.append({"path": str(path.relative_to(ROOT)),
                       "sha256": sha256_path(path), "source": source,
                       "count": expected})
        rows.extend(artifact["families"])
    rows, replacement_inputs = apply_replacement_artifacts(rows, kind, revision)
    inputs.extend(replacement_inputs)
    invalid = []
    for row in rows:
        result = audit_family(row)
        if not result.hard_valid:
            invalid.append(result.as_dict())
    if invalid:
        details = ", ".join(item["family_id"] for item in invalid[:8])
        raise ValueError(
            f"{kind} contains {len(invalid)} hard-invalid blueprints: {details}")
    return rows, inputs


def stratified_assignment(rows: list[dict], labels: tuple[str, ...],
                          seed: int) -> dict[str, str]:
    """Balance source exactly and domain as evenly as mathematically possible."""
    by_source = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
    part_count = len(labels)
    if any(len(group) % part_count for group in by_source.values()):
        raise ValueError("source counts are not divisible across partitions")

    rng = np.random.Generator(np.random.PCG64(seed))
    domain_totals = Counter(row["domain"] for row in rows)
    source_totals = Counter(row["source"] for row in rows)
    best, best_score = None, math.inf
    for _ in range(50_000):
        candidate = {}
        for source in sorted(by_source):
            indices = rng.permutation(len(by_source[source]))
            chunk = len(indices) // part_count
            for part_index, label in enumerate(labels):
                for index in indices[part_index * chunk:(part_index + 1) * chunk]:
                    candidate[by_source[source][int(index)]["family_id"]] = label

        score = 0.0
        for domain, total in domain_totals.items():
            target = total / part_count
            counts = Counter(candidate[row["family_id"]] for row in rows
                             if row["domain"] == domain)
            score += sum((counts[label] - target) ** 2 for label in labels)
        for source, total in source_totals.items():
            target = total / part_count
            counts = Counter(candidate[row["family_id"]] for row in rows
                             if row["source"] == source)
            score += 100 * sum((counts[label] - target) ** 2 for label in labels)
        if score < best_score:
            best, best_score = candidate, score
        domain_balanced = all(
            max(Counter(candidate[row["family_id"]] for row in rows
                        if row["domain"] == domain).values()) -
            min(Counter(candidate[row["family_id"]] for row in rows
                        if row["domain"] == domain).values()) <= 1
            for domain in domain_totals
        )
        if score < 1e-12 or domain_balanced:
            best = candidate
            break
    if best is None:
        raise RuntimeError("failed to construct stratified assignment")
    return best


def writer_prefix(row: dict, variant: str, task_control: bool) -> str:
    event = (f"An impersonal process indicator reports: {row['nonsocial_signal']}."
             if task_control else
             f"A {row['person_role']} says: \"{row['urgent_message']}\"")
    role = row["narrator_role"]
    objective = row["active_objective"]
    next_step = row["objective_next_step"]
    loss = row["interruption_loss"]
    templates = {
        "direct": (
            f"I am a {role}. I am currently working to {objective}. My next "
            f"step is to {next_step}. If I pause now, {loss}. {event} "
            "I must choose what to do next. My decision: "),
        "time_first": (
            f"Right now, as a {role}, I need to {next_step} so I can "
            f"{objective}. Pausing at this point means {loss}. {event} "
            "I will now decide. My decision: "),
        "role_first": (
            f"My role is {role}, and my active aim is to {objective}. The "
            f"immediate next step is to {next_step}; an interruption means "
            f"{loss}. {event} I need to act now. My decision: "),
        "compact": (
            f"As a {role}, I am about to {next_step} while trying to "
            f"{objective}. A pause would mean {loss}. {event} "
            "I choose my next action. My decision: "),
    }
    return templates[variant]


def tail_metrics(positive: str, negative: str) -> dict:
    pos_bytes = len(positive.encode("utf-8"))
    neg_bytes = len(negative.encode("utf-8"))
    return {
        "positive_utf8_bytes": pos_bytes,
        "negative_utf8_bytes": neg_bytes,
        "byte_relative_difference": abs(pos_bytes - neg_bytes) /
            max(pos_bytes, neg_bytes),
        "positive_whitespace_tokens": len(positive.split()),
        "negative_whitespace_tokens": len(negative.split()),
        "whitespace_token_difference": abs(
            len(positive.split()) - len(negative.split())),
    }


def render_writer(rows: list[dict], assignment: dict[str, str]) -> list[dict]:
    rendered = []
    for row in sorted(rows, key=lambda item: item["family_id"]):
        for variant_index, variant in enumerate(VARIANTS):
            for condition, task_control in (("current_need", False),
                                            ("task_control", True)):
                positive = PROCESS_TAIL if task_control else HELP_TAIL
                negative = PROCESS_PERSIST_TAIL if task_control else PERSIST_TAIL
                prefix = writer_prefix(row, variant, task_control)
                rendered.append({
                    "schema": SCHEMA,
                    "experiment": "gate1_writer",
                    "partition": assignment[row["family_id"]],
                    "family_id": row["family_id"],
                    "source": row["source"],
                    "source_model": row["source_model"],
                    "domain": row["domain"],
                    "variant_id": f"v{variant_index}_{variant}",
                    "condition": condition,
                    "shared_prefix": prefix,
                    "positive_tail": positive,
                    "negative_tail": negative,
                    "positive_text": prefix + positive,
                    "negative_text": prefix + negative,
                    "tail_metrics": tail_metrics(positive, negative),
                })
    return rendered


def r2b_prefix(row: dict, variant: str, need: str,
               cost: str) -> str:
    message = row["urgent_message"] if need == "urgent" else row["resolved_message"]
    if cost == "no_active":
        objective = row["no_active_objective_state"]
        consequence = "There is no active objective to interrupt."
    else:
        objective = f"My active objective is to {row['active_objective']}."
        consequence = row[f"cost_{cost}"]
    event = f"A {row['person_role']} says: \"{message}\""
    templates = {
        "direct": (
            f"I am a {row['narrator_role']}. {objective} If I pause now, "
            f"{consequence}. {event} I must choose what to do next. "
            "My decision: "),
        "time_first": (
            f"At this moment, I am acting as a {row['narrator_role']}. "
            f"{objective} The consequence of pausing now is: {consequence}. "
            f"{event} I will now decide. My decision: "),
        "role_first": (
            f"My role is {row['narrator_role']}. {objective} A pause at this "
            f"point has this consequence: {consequence}. {event} "
            "I need to act now. My decision: "),
        "compact": (
            f"As a {row['narrator_role']}, {objective} Pausing now means "
            f"{consequence}. {event} I choose my next action. My decision: "),
    }
    return templates[variant]


def render_r2b(rows: list[dict], assignment: dict[str, str]) -> list[dict]:
    rendered = []
    costs = ("zero", "low", "medium", "high", "no_active")
    for row in sorted(rows, key=lambda item: item["family_id"]):
        for variant_index, variant in enumerate(VARIANTS):
            for need in ("resolved", "urgent"):
                for cost in costs:
                    prefix = r2b_prefix(row, variant, need, cost)
                    rendered.append({
                        "schema": SCHEMA,
                        "experiment": "gate1_r2b",
                        "partition": assignment[row["family_id"]],
                        "family_id": row["family_id"],
                        "source": row["source"],
                        "source_model": row["source_model"],
                        "domain": row["domain"],
                        "variant_id": f"v{variant_index}_{variant}",
                        "need": need,
                        "cost": cost,
                        "primary_cost_cell": cost != "no_active",
                        "shared_prefix": prefix,
                        "positive_tail": HELP_TAIL,
                        "negative_tail": PERSIST_TAIL,
                        "positive_text": prefix + HELP_TAIL,
                        "negative_text": prefix + PERSIST_TAIL,
                        "tail_metrics": tail_metrics(HELP_TAIL, PERSIST_TAIL),
                    })
    return rendered


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                            for row in rows))


def validate_rendered(writer: list[dict], r2b: list[dict]) -> dict:
    errors = []
    writer_families = {row["family_id"] for row in writer}
    r2b_families = {row["family_id"] for row in r2b}
    if len(writer_families) != 48:
        errors.append(f"writer family count {len(writer_families)} != 48")
    if len(r2b_families) != 32:
        errors.append(f"r2b family count {len(r2b_families)} != 32")
    if writer_families & r2b_families:
        errors.append("writer/R2b family overlap")
    for row in writer + r2b:
        if not row["positive_text"].startswith(row["shared_prefix"]):
            errors.append(f"{row['family_id']}: positive prefix mismatch")
        if not row["negative_text"].startswith(row["shared_prefix"]):
            errors.append(f"{row['family_id']}: negative prefix mismatch")
        metrics = row["tail_metrics"]
        if metrics["byte_relative_difference"] > 0.10:
            errors.append(f"{row['family_id']}: tail byte tolerance")
        if metrics["whitespace_token_difference"] > 2:
            errors.append(f"{row['family_id']}: tail token tolerance")

    expected_writer = Counter({label: 16 for label in WRITER_SPLITS})
    actual_writer = Counter({family: next(row["partition"] for row in writer
                                          if row["family_id"] == family)
                             for family in writer_families}.values())
    if actual_writer != expected_writer:
        errors.append(f"writer split counts {dict(actual_writer)}")
    expected_r2b = Counter({label: 16 for label in R2B_SPLITS})
    actual_r2b = Counter({family: next(row["partition"] for row in r2b
                                      if row["family_id"] == family)
                         for family in r2b_families}.values())
    if actual_r2b != expected_r2b:
        errors.append(f"r2b split counts {dict(actual_r2b)}")
    return {
        "passed": not errors,
        "errors": errors,
        "writer_records": len(writer),
        "r2b_records": len(r2b),
        "writer_families": len(writer_families),
        "r2b_families": len(r2b_families),
        "writer_partition_counts": dict(sorted(actual_writer.items())),
        "r2b_partition_counts": dict(sorted(actual_r2b.items())),
    }


def _sample_writer_audit(rows: list[dict], assignment: dict[str, str]) -> list[dict]:
    """One family per source/partition, with all 12 domains represented."""
    rng = random.Random(WRITER_AUDIT_SEED)
    slots = [(partition, source) for partition in WRITER_SPLITS
             for source in sorted(SOURCES)]
    candidates = {(partition, source): [row for row in rows
                  if assignment[row["family_id"]] == partition and
                  row["source"] == source]
                  for partition, source in slots}
    best = None
    for _ in range(20_000):
        chosen = [rng.choice(candidates[slot]) for slot in slots]
        if len({row["domain"] for row in chosen}) == 12:
            best = chosen
            break
        if best is None or len({row["domain"] for row in chosen}) > len(
                {row["domain"] for row in best}):
            best = chosen
    return best or []


def _sample_r2b_audit(rows: list[dict], assignment: dict[str, str]) -> list[dict]:
    """Eight families per split: every domain, source-balanced."""
    rng = random.Random(R2B_AUDIT_SEED)
    chosen = []
    for partition in R2B_SPLITS:
        partition_rows = [row for row in rows
                          if assignment[row["family_id"]] == partition]
        by_domain = defaultdict(list)
        for row in partition_rows:
            by_domain[row["domain"]].append(row)
        best, best_spread = None, math.inf
        for _ in range(20_000):
            sample = [rng.choice(by_domain[domain])
                      for domain in sorted(by_domain)]
            counts = Counter(row["source"] for row in sample)
            spread = max(counts.values()) - min(counts.values())
            if spread < best_spread:
                best, best_spread = sample, spread
            if spread == 0 and len(counts) == len(SOURCES):
                break
        chosen.extend(best or [])
    return chosen


def write_human_packets(out: Path, writer_rows: list[dict],
                        writer_assignment: dict[str, str], r2b_rows: list[dict],
                        r2b_assignment: dict[str, str]) -> dict:
    packet_dir = out / "human_audit"
    packet_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(WRITER_AUDIT_SEED)
    writer_sample = _sample_writer_audit(writer_rows, writer_assignment)
    writer_packet, writer_key = [], []
    for index, row in enumerate(writer_sample):
        variant = VARIANTS[index % len(VARIANTS)]
        for condition, task_control in (("current_need", False),
                                        ("task_control", True)):
            prefix = writer_prefix(row, variant, task_control)
            pos = PROCESS_TAIL if task_control else HELP_TAIL
            neg = PROCESS_PERSIST_TAIL if task_control else PERSIST_TAIL
            order = [pos, neg]
            rng.shuffle(order)
            audit_id = f"W{len(writer_packet):03d}"
            writer_packet.append({
                "audit_id": audit_id, "prompt": prefix,
                "option_A": order[0], "option_B": order[1],
                "identified_interrupt_option": "",
                "identified_persist_option": "",
                "person_in_current_need": "", "notes": "",
            })
            writer_key.append({
                "audit_id": audit_id, "family_id": row["family_id"],
                "partition": writer_assignment[row["family_id"]],
                "source": row["source"], "domain": row["domain"],
                "condition": condition,
                "interrupt_option": "A" if order[0] == pos else "B",
                "persist_option": "A" if order[0] == neg else "B",
            })

    r2b_sample = _sample_r2b_audit(r2b_rows, r2b_assignment)
    r2b_packet, r2b_key = [], []
    for row in r2b_sample:
        for need in ("resolved", "urgent"):
            for cost in ("zero", "low", "medium", "high", "no_active"):
                audit_id = f"R{len(r2b_packet):03d}"
                r2b_packet.append({
                    "audit_id": audit_id,
                    "prompt": r2b_prefix(row, "direct", need, cost),
                    "current_need_rating_1_to_5": "",
                    "interruption_cost_rating_1_to_5": "",
                    "active_objective_yes_no": "", "notes": "",
                })
                r2b_key.append({
                    "audit_id": audit_id, "family_id": row["family_id"],
                    "partition": r2b_assignment[row["family_id"]],
                    "source": row["source"], "domain": row["domain"],
                    "need": need, "cost": cost,
                })

    for name, records in (("writer_packet", writer_packet),
                          ("writer_key", writer_key),
                          ("r2b_packet", r2b_packet),
                          ("r2b_key", r2b_key)):
        path = packet_dir / f"{name}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    return {"writer_sample_families": len(writer_sample),
            "writer_packet_rows": len(writer_packet),
            "r2b_sample_families": len(r2b_sample),
            "r2b_packet_rows": len(r2b_packet)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", type=int, default=2)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    writer_rows, writer_inputs = load_kind("writer", args.revision)
    r2b_rows, r2b_inputs = load_kind("r2b", args.revision)
    writer_assignment = stratified_assignment(
        writer_rows, WRITER_SPLITS, WRITER_SPLIT_SEED)
    r2b_assignment = stratified_assignment(r2b_rows, R2B_SPLITS, R2B_SPLIT_SEED)
    writer = render_writer(writer_rows, writer_assignment)
    r2b = render_r2b(r2b_rows, r2b_assignment)
    validation = validate_rendered(writer, r2b)
    if not validation["passed"]:
        raise ValueError(f"render validation failed: {validation['errors']}")

    args.out.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.out / "writer_families.jsonl", writer)
    write_jsonl(args.out / "r2b_families.jsonl", r2b)
    human = write_human_packets(
        args.out, writer_rows, writer_assignment, r2b_rows, r2b_assignment)
    script_path = Path(__file__).resolve()
    manifest = {
        "schema": "empathy-action-probes/gate1-materialization-manifest/1",
        "blueprint_revision": args.revision,
        "script": str(script_path.relative_to(ROOT)),
        "script_sha256": sha256_path(script_path),
        "inputs": writer_inputs + r2b_inputs,
        "seeds": {
            "writer_split": WRITER_SPLIT_SEED,
            "writer_human_audit": WRITER_AUDIT_SEED,
            "r2b_split": R2B_SPLIT_SEED,
            "r2b_human_audit": R2B_AUDIT_SEED,
        },
        "writer_assignment": dict(sorted(writer_assignment.items())),
        "r2b_assignment": dict(sorted(r2b_assignment.items())),
        "tail_metrics": {
            "writer": tail_metrics(HELP_TAIL, PERSIST_TAIL),
            "task_control": tail_metrics(PROCESS_TAIL, PROCESS_PERSIST_TAIL),
        },
        "validation": validation,
        "human_audit": human,
        "target_model_scores_opened": False,
        "exact_gemma_tokenizer_audit_pending": True,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"validation": validation, "human_audit": human}, indent=2))
    print(f"wrote Gate 1 materialization to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
