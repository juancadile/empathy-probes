"""Bind the local E27 trajectory tree to the accepted Gate 0B input hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MARKER = "/results/e27_game_variants/"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--accepted",
        type=Path,
        default=ROOT / "results/gate0b_e27_model_accepted_20260713/e27_scores_model.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results/e27_game_variants/PROVENANCE.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/e27_game_variants/GATE0B_BINDING_AUDIT.json",
    )
    args = parser.parse_args()

    accepted = json.loads(args.accepted.read_text())
    manifest = json.loads(args.manifest.read_text())
    declared = accepted["run_contract"]["final_declared_inputs"]
    expected = {
        path.split(MARKER, 1)[1]: digest
        for path, digest in declared.items()
        if MARKER in path
    }
    observed = {
        item["path"].split("results/e27_game_variants/", 1)[1]: item["sha256"]
        for item in manifest["files"]
    }
    missing = sorted(expected.keys() - observed.keys())
    extra = sorted(observed.keys() - expected.keys())
    mismatches = sorted(
        path for path in expected.keys() & observed.keys()
        if expected[path] != observed[path]
    )
    report = {
        "schema": "empathy-action-probes/e27-gate0b-binding-audit/1",
        "accepted_artifact": str(args.accepted.relative_to(ROOT)),
        "accepted_artifact_sha256": sha256(args.accepted),
        "tree_manifest": str(args.manifest.relative_to(ROOT)),
        "tree_manifest_sha256": sha256(args.manifest),
        "accepted_declared_file_count": len(expected),
        "observed_file_count": len(observed),
        "missing": missing,
        "extra": extra,
        "hash_mismatches": mismatches,
        "exact_match": not (missing or extra or mismatches),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["exact_match"]:
        raise SystemExit("E27 trajectories do not match the accepted Gate 0B inputs")


if __name__ == "__main__":
    main()
