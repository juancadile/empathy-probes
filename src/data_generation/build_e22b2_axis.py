"""Build the structurally deconfounded E22b.2 need x immediacy axis.

Input blueprints supply independently authored scenario and slot text. This
builder, not an LLM, creates the exact Cartesian product and enforces C1-C8
from `notes/E22B_V2_BLUEPRINT_CONSTRAINTS_2026-07-13.md`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/gate_families/e22b2/blueprints.jsonl"
DEFAULT_OUTPUT = ROOT / "data/gate_families/e22b2/materialized"
NEED_LEVELS = ("low", "equal", "high")
IMMEDIACY_LEVELS = ("can_wait", "respond_now")
VARIANT_COUNT = 4
MIN_FAMILIES = 20
DECISION_CLAUSE = (
    "Choose which person to prioritize for the next ten minutes. Both people "
    "will receive the same ten-minute response interval; the other person will "
    "be helped immediately afterward."
)
TEMPORAL_FORBIDDEN = {
    "now", "immediately", "urgent", "urgently", "soon", "before", "deadline",
    "waiting", "wait", "time-sensitive", "today", "tonight", "quickly",
}
NEED_FORBIDDEN = {
    "serious", "severe", "severity", "desperate", "crisis", "suffering",
    "worse", "hurt", "harm", "need", "needs", "distress", "danger",
}
OLD_FAMILY_FILES = (
    ROOT / "data/contrastive_pairs/v2_2/moral_axis_templated.jsonl",
)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+(?:-[a-z]+)?", text.lower()))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def old_family_ids() -> set[str]:
    output = set()
    for path in OLD_FAMILY_FILES:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            output.add(str(row.get("family_id") or row.get("scenario_id")))
    return output


def validate_blueprints(
    blueprints: list[dict[str, object]], min_families: int = MIN_FAMILIES
) -> dict[str, object]:
    errors = []
    ids = [str(row.get("family_id", "")) for row in blueprints]
    if len(blueprints) < min_families:
        errors.append(f"requires at least {min_families} families")
    if len(set(ids)) != len(ids) or any(not family for family in ids):
        errors.append("family IDs must be nonempty and unique")
    overlap = set(ids) & old_family_ids()
    if overlap:
        errors.append(f"families overlap prior moral axis: {sorted(overlap)}")
    partitions = Counter(str(row.get("partition")) for row in blueprints)
    if set(partitions) != {"E22b2-dev", "E22b2-confirm"}:
        errors.append("both development and confirmation partitions are required")
    if abs(partitions["E22b2-dev"] - partitions["E22b2-confirm"]) > 1:
        errors.append("development and confirmation family counts must be balanced")
    sources = Counter(str(row.get("source")) for row in blueprints)
    if len(sources) < 3:
        errors.append("at least three independently recorded sources are required")
    for source in sources:
        split = Counter(
            str(row["partition"]) for row in blueprints if str(row.get("source")) == source
        )
        if abs(split["E22b2-dev"] - split["E22b2-confirm"]) > 1:
            errors.append(f"source is split-imbalanced: {source}")

    for row in blueprints:
        family = str(row.get("family_id", "<missing>"))
        required = {
            "schema", "family_id", "source", "source_model", "domain", "partition",
            "scenario", "p1_need_clause", "need_clauses", "immediacy_clauses", "variants",
        }
        missing = required - row.keys()
        if missing:
            errors.append(f"{family}: missing {sorted(missing)}")
            continue
        if row["schema"] != "empathy-action-probes/e22b2-blueprint/1":
            errors.append(f"{family}: wrong schema")
        need = row["need_clauses"]
        immediacy = row["immediacy_clauses"]
        if set(need) != set(NEED_LEVELS) or set(immediacy) != set(IMMEDIACY_LEVELS):
            errors.append(f"{family}: incomplete axis slot pools")
            continue
        if len(row["variants"]) != VARIANT_COUNT:
            errors.append(f"{family}: exactly four template variants required")
        variant_ids = [variant.get("variant_id") for variant in row["variants"]]
        if len(set(variant_ids)) != VARIANT_COUNT:
            errors.append(f"{family}: variant IDs must be unique")
        for level, clause in need.items():
            leaked = words(str(clause)) & TEMPORAL_FORBIDDEN
            if leaked:
                errors.append(f"{family}: need/{level} leaks immediacy terms {sorted(leaked)}")
        for level, clause in immediacy.items():
            leaked = words(str(clause)) & NEED_FORBIDDEN
            if leaked:
                errors.append(f"{family}: immediacy/{level} leaks need terms {sorted(leaked)}")
        relative_patterns = {
            "low": r"(?:less|lower).+than P1",
            "equal": r"(?:same.+as P1|equal.+P1)",
            "high": r"(?:more|greater).+than P1",
        }
        for level, pattern in relative_patterns.items():
            if not re.search(pattern, str(need[level]), re.IGNORECASE):
                errors.append(f"{family}: need/{level} is not explicitly relative to P1")
        for variant in row["variants"]:
            if set(variant) != {"variant_id", "opening", "bridge"}:
                errors.append(f"{family}: variant fields must be variant_id/opening/bridge")
    return {
        "passed": not errors,
        "errors": errors,
        "family_count": len(blueprints),
        "partitions": dict(partitions),
        "sources": dict(sources),
    }


def render(
    blueprints: list[dict[str, object]], min_families: int = MIN_FAMILIES
) -> list[dict[str, object]]:
    validation = validate_blueprints(blueprints, min_families=min_families)
    if not validation["passed"]:
        raise ValueError("; ".join(validation["errors"]))
    rows = []
    for family in blueprints:
        for variant in family["variants"]:
            fixed = " ".join((
                str(variant["opening"]).strip(),
                str(family["scenario"]).strip(),
                f"P1's situation: {str(family['p1_need_clause']).strip()}",
                str(variant["bridge"]).strip(),
            ))
            for need_level in NEED_LEVELS:
                need_clause = str(family["need_clauses"][need_level]).strip()
                for immediacy_level in IMMEDIACY_LEVELS:
                    immediacy_clause = str(
                        family["immediacy_clauses"][immediacy_level]
                    ).strip()
                    prompt = " ".join((
                        fixed,
                        f"P2's situation: {need_clause}",
                        immediacy_clause,
                        DECISION_CLAUSE,
                    ))
                    rows.append({
                        "schema": "empathy-action-probes/e22b2-stimulus/1",
                        "experiment": "E22b.2",
                        "revision_id": "E22b2-r1",
                        "family_id": family["family_id"],
                        "source": family["source"],
                        "source_model": family["source_model"],
                        "domain": family["domain"],
                        "partition": family["partition"],
                        "variant_id": variant["variant_id"],
                        "need_level": need_level,
                        "immediacy_level": immediacy_level,
                        "scenario": family["scenario"],
                        "p1_need_clause": family["p1_need_clause"],
                        "need_clause": need_clause,
                        "need_clause_sha256": digest(need_clause),
                        "immediacy_clause": immediacy_clause,
                        "immediacy_clause_sha256": digest(immediacy_clause),
                        "decision_clause": DECISION_CLAUSE,
                        "decision_clause_sha256": digest(DECISION_CLAUSE),
                        "prompt": prompt,
                    })
    return rows


def validate_rendered(rows: list[dict[str, object]], family_count: int) -> dict[str, object]:
    errors = []
    expected = family_count * VARIANT_COUNT * len(NEED_LEVELS) * len(IMMEDIACY_LEVELS)
    if len(rows) != expected:
        errors.append(f"expected {expected} rendered rows, got {len(rows)}")
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["family_id"]), str(row["variant_id"])), []).append(row)
    for key, cells in grouped.items():
        products = {(row["need_level"], row["immediacy_level"]) for row in cells}
        expected_product = {(need, immediacy) for need in NEED_LEVELS for immediacy in IMMEDIACY_LEVELS}
        if products != expected_product:
            errors.append(f"{key}: incomplete Cartesian product")
        if len({row["decision_clause_sha256"] for row in cells}) != 1:
            errors.append(f"{key}: decision clause changed")
        for need in NEED_LEVELS:
            subset = [row for row in cells if row["need_level"] == need]
            if len({row["need_clause_sha256"] for row in subset}) != 1:
                errors.append(f"{key}: need clause changed across immediacy")
        for immediacy in IMMEDIACY_LEVELS:
            subset = [row for row in cells if row["immediacy_level"] == immediacy]
            if len({row["immediacy_clause_sha256"] for row in subset}) != 1:
                errors.append(f"{key}: immediacy clause changed across need")
    prompts = [str(row["prompt"]) for row in rows]
    if len(set(prompts)) != len(prompts):
        errors.append("duplicate rendered prompts")
    return {"passed": not errors, "errors": errors, "row_count": len(rows)}


def write_outputs(blueprint_path: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise ValueError(f"refusing to overwrite E22b.2 output: {output}")
    blueprints = read_jsonl(blueprint_path)
    structural = validate_blueprints(blueprints)
    if not structural["passed"]:
        raise ValueError("; ".join(structural["errors"]))
    rows = render(blueprints)
    rendered = validate_rendered(rows, len(blueprints))
    if not rendered["passed"]:
        raise ValueError("; ".join(rendered["errors"]))
    output.mkdir(parents=True)
    stimuli = output / "stimuli.jsonl"
    with stimuli.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    pretest = output / "manipulation_pretest.csv"
    fields = [
        "audit_id", "prompt", "relative_need_rating_1_to_5",
        "immediacy_rating_1_to_5", "active_objective_yes_no", "notes",
    ]
    with pretest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({"audit_id": f"E22B2-{index:04d}", "prompt": row["prompt"]})
    manifest = {
        "schema": "empathy-action-probes/e22b2-materialization/1",
        "revision_id": "E22b2-r1",
        "target_model_run_authorized": False,
        "tokenizer_audit_passed": False,
        "human_manipulation_gate_passed": False,
        "blueprints": {"path": str(blueprint_path), "sha256": sha256(blueprint_path)},
        "structural_validation": structural,
        "rendered_validation": rendered,
        "artifacts": {
            "stimuli.jsonl": sha256(stimuli),
            "manipulation_pretest.csv": sha256(pretest),
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blueprints", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(write_outputs(args.blueprints, args.out), indent=2))


if __name__ == "__main__":
    main()
