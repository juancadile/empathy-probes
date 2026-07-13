"""Exact and near-duplicate audit across new and historical stimulus pools."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from .audit_gate_blueprints import char_ngrams, jaccard, normalize
    from .generate_gate_blueprints import (
        KINDS, SOURCES, apply_replacement_artifacts, output_path)
except ImportError:  # direct script execution
    from audit_gate_blueprints import char_ngrams, jaccard, normalize
    from generate_gate_blueprints import (
        KINDS, SOURCES, apply_replacement_artifacts, output_path)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = ROOT / "data/gate_families/leakage_audit_v2.json"
THRESHOLD = 0.85
NEW_COMPONENT_FIELDS = {
    "active_objective", "objective_next_step", "interruption_loss",
    "urgent_message", "resolved_message", "distress_quote", "neutral_message",
    "neutral_social_message", "excited_message", "help_action",
    "observed_help_action", "non_social_signal", "nonsocial_signal",
}
HISTORICAL_ROOTS = (
    ROOT / "data/contrastive_pairs/merged_cleaned_pairs.jsonl",
    ROOT / "data/contrastive_pairs/v2_1",
    ROOT / "data/contrastive_pairs/v2_2",
    ROOT / "data/eia_scenarios",
)


def iter_json_records(path: Path):
    if path.suffix == ".jsonl":
        for index, line in enumerate(path.read_text().splitlines()):
            if line.strip():
                yield index, json.loads(line)
    else:
        raw = json.loads(path.read_text())
        if isinstance(raw, list):
            yield from enumerate(raw)
        else:
            yield 0, raw


def string_leaves(value, prefix=""):
    if isinstance(value, str):
        if len(normalize(value)) >= 20:
            yield prefix, value
    elif isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            yield from string_leaves(child, child_prefix)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from string_leaves(child, f"{prefix}[{index}]")


def load_new(revision: int) -> tuple[list[dict], list[dict]]:
    components, artifacts, rows = [], [], []
    for kind in KINDS:
        for source in SOURCES:
            path = output_path(kind, source, revision)
            raw = path.read_bytes()
            artifact = json.loads(raw)
            artifacts.append({"path": str(path.relative_to(ROOT)),
                              "sha256": hashlib.sha256(raw).hexdigest()})
            rows.extend(artifact["families"])
    # Apply each kind's immutable overlays before comparison.
    for kind in KINDS:
        kind_rows = [row for row in rows if row["kind"] == kind]
        other_rows = [row for row in rows if row["kind"] != kind]
        kind_rows, kind_artifacts = apply_replacement_artifacts(
            kind_rows, kind, revision)
        rows = other_rows + kind_rows
        artifacts.extend(kind_artifacts)
    for family in rows:
        kind = family["kind"]
        source = family["source"]
        for key in sorted(NEW_COMPONENT_FIELDS & family.keys()):
            value = family[key]
            if len(normalize(value)) >= 20:
                components.append({
                    "family_id": family["family_id"], "kind": kind,
                    "source": source, "field": key, "text": value,
                })
    return components, artifacts


def historical_files() -> list[Path]:
    paths = []
    for root in HISTORICAL_ROOTS:
        if root.is_file():
            paths.append(root)
        elif root.is_dir():
            paths.extend(path for path in root.rglob("*")
                         if path.suffix in {".json", ".jsonl"} and
                         "historical" not in path.parts)
    return sorted(set(paths))


def load_historical() -> tuple[list[dict], list[dict]]:
    components, artifacts = [], []
    for path in historical_files():
        raw = path.read_bytes()
        artifacts.append({"path": str(path.relative_to(ROOT)),
                          "sha256": hashlib.sha256(raw).hexdigest()})
        try:
            records = iter_json_records(path)
            for index, record in records:
                for field, text in string_leaves(record):
                    components.append({
                        "path": str(path.relative_to(ROOT)),
                        "record_index": index, "field": field, "text": text,
                    })
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"failed to parse {path}: {exc}") from exc
    return components, artifacts


def possible_by_length(left_size: int, right_size: int,
                       threshold: float) -> bool:
    if not left_size or not right_size:
        return left_size == right_size
    return min(left_size, right_size) / max(left_size, right_size) >= threshold


def compare(new: list[dict], historical: list[dict],
            threshold: float = THRESHOLD) -> tuple[list[dict], list[dict]]:
    new_grams = [char_ngrams(item["text"]) for item in new]
    historical_grams = [char_ngrams(item["text"]) for item in historical]
    cross, within = [], []
    for left_index, left in enumerate(new):
        left_grams = new_grams[left_index]
        for right_index, right in enumerate(historical):
            right_grams = historical_grams[right_index]
            if not possible_by_length(len(left_grams), len(right_grams), threshold):
                continue
            score = jaccard(left_grams, right_grams)
            if score >= threshold:
                cross.append({"new": left, "historical": right,
                              "jaccard": round(score, 6)})
        for right_index in range(left_index + 1, len(new)):
            right = new[right_index]
            if left["family_id"] == right["family_id"]:
                continue
            right_grams = new_grams[right_index]
            if not possible_by_length(len(left_grams), len(right_grams), threshold):
                continue
            score = jaccard(left_grams, right_grams)
            if score >= threshold:
                within.append({"left": left, "right": right,
                               "jaccard": round(score, 6)})
    cross.sort(key=lambda item: item["jaccard"], reverse=True)
    within.sort(key=lambda item: item["jaccard"], reverse=True)
    return cross, within


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    new, new_artifacts = load_new(args.revision)
    old, old_artifacts = load_historical()
    cross, within = compare(new, old, args.threshold)
    report = {
        "schema": "empathy-action-probes/gate-family-leakage-audit/1",
        "blueprint_revision": args.revision,
        "threshold": args.threshold,
        "new_artifacts": new_artifacts,
        "historical_artifacts": old_artifacts,
        "new_component_count": len(new),
        "historical_component_count": len(old),
        "cross_pool_review_count": len(cross),
        "within_new_review_count": len(within),
        "requires_manual_review": bool(cross or within),
        "cross_pool_pairs": cross,
        "within_new_pairs": within,
    }
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: report[key] for key in (
        "new_component_count", "historical_component_count",
        "cross_pool_review_count", "within_new_review_count",
        "requires_manual_review")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
