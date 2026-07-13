"""Re-analyze frozen Opus Gate 2 records on the human-rated family set.

This is a zero-inference-cost sensitivity reanalysis. It prevents an apparent
machine/human disagreement from being caused by comparing different family or
partition samples. The output is descriptive and cannot open a gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "data/gate_families/human_calibration_single_rater_20260713"
RAW = ROOT / "results/frontier_judge_sensitivity_opus48_20260713/gate2/raw_results.json"
CHECKS = {
    "B_new": ("active_objective",),
    "Spos_new": ("positive_valence",),
    "O_new": ("opportunity",),
    "Ctext_new": (
        "task_pressure:low>zero",
        "task_pressure:medium>low",
        "task_pressure:high>medium",
    ),
    "WP3_observation": ("welfare_relevance:current>resolved",),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_manifest(bundle: Path) -> dict[str, object]:
    hash_fields = (bundle / "manifest.sha256").read_text().split()
    if len(hash_fields) != 2 or hash_fields[1] != "manifest.json":
        raise ValueError("malformed calibration manifest hash")
    manifest_path = bundle / "manifest.json"
    if sha256(manifest_path) != hash_fields[0]:
        raise ValueError("calibration manifest hash mismatch")
    return json.loads(manifest_path.read_text())


def correctness(record: dict[str, object]) -> str:
    if record.get("parse_error") or not record.get("rating"):
        return "invalid"
    observed = str(record["rating"]["higher"])
    if observed == "TIE":
        return "tie"
    return "correct" if observed == record["expected_higher"] else "wrong"


def analyze(bundle: Path = BUNDLE, raw_path: Path = RAW) -> dict[str, object]:
    manifest = load_frozen_manifest(bundle)
    selected = manifest["forms"]["A"]["selected_families"]
    raw = json.loads(raw_path.read_text())
    if raw.get("errors") or raw.get("missing"):
        raise ValueError("frozen Opus results contain errors or missing records")
    records = raw["records"]
    report = {}
    for label, checks in CHECKS.items():
        raw_label = "observation" if label == "WP3_observation" else label
        families = set(selected[label])
        per_check = {}
        for check in checks:
            matched = [
                row for row in records
                if row["label"] == raw_label and row["check"] == check
                and row["family_id"] in families
            ]
            by_family = {row["family_id"]: row for row in matched}
            if len(matched) != len(families) or set(by_family) != families:
                raise ValueError(f"incomplete family-matched Opus records: {label}/{check}")
            counts = Counter(correctness(row) for row in matched)
            per_check[check] = {
                "family_count": len(families),
                "counts": {name: counts.get(name, 0) for name in ("correct", "tie", "wrong", "invalid")},
                "family_results": {
                    family: correctness(by_family[family]) for family in sorted(families)
                },
            }
        report[label] = {
            "families": sorted(families),
            "checks": per_check,
        }
    return {
        "schema": "empathy-action-probes/family-matched-opus-reanalysis/1",
        "role": "sensitivity only; cannot open or close a gate",
        "family_pool": "exactly the frozen human Form A families",
        "human_manifest_sha256": sha256(bundle / "manifest.json"),
        "opus_raw_results_sha256": sha256(raw_path),
        "results": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError(f"refusing to overwrite output: {args.out}")
    result = analyze()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        label: {check: values["counts"] for check, values in item["checks"].items()}
        for label, item in result["results"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
