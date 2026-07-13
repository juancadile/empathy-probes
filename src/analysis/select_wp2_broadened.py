"""Post-null all-block WP2 development screen.

This search is frozen after the four-block WP2 development null and is therefore
discovery-only. It reuses the original candidate classes and family-grouped
nested-CV rule over every Gemma block. It cannot emit a frozen representation,
open confirmation data, or authorize a scientific claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import select_wp2_representation as wp2


ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "notes/WP2_BROADENED_DEV_LOCK_2026-07-13.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError(f"refusing to overwrite selection output: {args.out}")

    plan = json.loads((args.activations / "extraction_plan.json").read_text())
    if plan.get("authorization_mode") != "broadened_exploratory_dev":
        raise ValueError("broadened selector requires broadened exploratory activations")
    if plan.get("blocks") != list(range(42)) or plan.get("site_policy") != "all_blocks":
        raise ValueError("broadened selector requires all 42 blocks")
    lock = json.loads(LOCK.read_text())
    if lock.get("claim_authorized") is not False or lock.get("confirmation_authorized") is not False:
        raise ValueError("broadened lock must forbid claims and confirmation")

    args.out.mkdir(parents=True)
    data = wp2.Dataset.load(args.activations)
    configs = wp2.candidate_configs([int(block) for block in data.blocks])
    nested = wp2.nested_selection(data, configs)
    target_families = wp2.all_families(data.rows, "wp3")
    control_families = wp2.all_families(data.rows, "wp1")
    full_cv = wp2.cross_validate_configs(
        data, configs, target_families, control_families, 4, wp2.MASTER_SEED
    )
    diagnostic = wp2.choose_candidate(full_cv, [int(block) for block in data.blocks])
    quiet = nested["metrics"]["quiet_auroc"]
    screen_passed = (
        nested["metrics"]["target_auroc"] >= 0.75
        and all(abs(value - 0.5) <= 0.10 for value in quiet.values())
    )
    report = {
        "schema": "empathy-action-probes/wp2-broadened-development-screen/1",
        "experiment_id": lock["experiment_id"],
        "role": "post-null discovery only",
        "claim_authorized": False,
        "confirmation_authorized": False,
        "candidate_count": len(configs),
        "candidate_blocks": [int(block) for block in data.blocks],
        "token_roles": ["quote_boundary", "prompt_final"],
        "candidate_classes": "unchanged from WP2_IMPLEMENTATION_LOCK_2026-07-13.md",
        "nested_outer": nested,
        "full_development_cv": full_cv,
        "screen": {
            "outcome": "candidate-found" if screen_passed else "no-candidate",
            "development_screen_passed": screen_passed,
            "best_full_development_diagnostic": diagnostic["config"],
            "nested_target_auroc": nested["metrics"]["target_auroc"],
            "nested_quiet_auroc": quiet,
        },
        "interpretation": (
            "A candidate-found outcome only motivates a fresh preregistration and fresh "
            "families. A no-candidate outcome strengthens the conditional development "
            "null but is not a nonexistence claim and remains dependent on human target/"
            "control validation."
        ),
        "artifacts": {
            "activation_plan_sha256": wp2.sha256(args.activations / "extraction_plan.json"),
            "activations_sha256": wp2.sha256(args.activations / "activations.npz"),
            "rows_sha256": wp2.sha256(args.activations / "rows.jsonl"),
            "lock_sha256": wp2.sha256(LOCK),
            "selector_sha256": wp2.sha256(Path(__file__).resolve()),
        },
    }
    (args.out / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["screen"], indent=2))


if __name__ == "__main__":
    main()
