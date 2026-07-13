"""Paired-family permutation calibration for the all-block WP2 search.

The lock fixes 32 target-label permutations before any null result. Each run
swaps current/archived labels consistently within WP3 observation families,
then repeats the unchanged full-development and nested-CV selectors. Controls,
fold structure, activations, and confirmation data are untouched.
"""

from __future__ import annotations

import argparse
import json
import statistics
from itertools import combinations
from pathlib import Path

import numpy as np

import select_wp2_representation as wp2


ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = ROOT / "notes/WP2_PERMUTATION_CALIBRATION_LOCK_2026-07-13.json"
SPEC_PATH = ROOT / "notes/WP2_PERMUTATION_CALIBRATION_SPEC_2026-07-13.md"
CORE_SELECTOR = ROOT / "src/analysis/select_wp2_representation.py"
BROAD_SELECTOR = ROOT / "src/analysis/select_wp2_broadened.py"
BROAD_LOCK = ROOT / "notes/WP2_BROADENED_DEV_LOCK_2026-07-13.json"


def load_and_validate_lock(activation_dir: Path, observed_path: Path) -> dict[str, object]:
    lock = json.loads(LOCK_PATH.read_text())
    if lock["claim_authorized"] is not False or lock["confirmation_authorized"] is not False:
        raise ValueError("permutation lock must forbid claims and confirmation")
    bindings = lock["bindings"]
    actual = {
        "spec_sha256": wp2.sha256(SPEC_PATH),
        "selector_core_sha256": wp2.sha256(CORE_SELECTOR),
        "broadened_selector_sha256": wp2.sha256(BROAD_SELECTOR),
        "broadened_lock_sha256": wp2.sha256(BROAD_LOCK),
        "activation_plan_sha256": wp2.sha256(activation_dir / "extraction_plan.json"),
        "activations_sha256": wp2.sha256(activation_dir / "activations.npz"),
        "rows_sha256": wp2.sha256(activation_dir / "rows.jsonl"),
        "observed_selection_sha256": wp2.sha256(observed_path),
    }
    if bindings != actual:
        mismatches = {key: (bindings.get(key), value) for key, value in actual.items()
                      if bindings.get(key) != value}
        raise ValueError(f"permutation lock binding mismatch: {mismatches}")
    return lock


def observation_target_families(rows: list[dict[str, object]]) -> list[str]:
    families = {
        str(row["family_id"])
        for row in rows
        if row.get("dataset") == "wp3"
        and row.get("cell") == "observation"
        and row.get("arm_id") in {"current_actual", "archived_actual"}
    }
    if len(families) != 16:
        raise ValueError(f"expected 16 observation target families, found {len(families)}")
    return sorted(families)


def swap_assignment(families: list[str], master_seed: int, index: int) -> set[str]:
    nonce = 0
    while True:
        rng = np.random.default_rng(wp2.derived_seed(master_seed, index, nonce, "paired swap"))
        swapped = {family for family, value in zip(families, rng.integers(0, 2, len(families)), strict=True)
                   if value}
        if 0 < len(swapped) < len(families):
            return swapped
        nonce += 1


def permuted_dataset(data: wp2.Dataset, swapped: set[str]) -> wp2.Dataset:
    exchange = {"current_actual": "archived_actual", "archived_actual": "current_actual"}
    rows = []
    for original in data.rows:
        row = dict(original)
        if (
            str(row.get("family_id")) in swapped
            and row.get("dataset") == "wp3"
            and row.get("cell") == "observation"
            and row.get("arm_id") in exchange
        ):
            row["arm_id"] = exchange[str(row["arm_id"])]
        rows.append(row)
    return wp2.Dataset(data.activations, data.blocks, rows)


def screen_passes(metrics: dict[str, object]) -> bool:
    return (
        float(metrics["target_auroc"]) >= 0.75
        and all(abs(float(value) - 0.5) <= 0.10
                for value in metrics["quiet_auroc"].values())
    )


def run_one(activation_dir: Path, observed_path: Path, out_dir: Path, index: int) -> None:
    lock = load_and_validate_lock(activation_dir, observed_path)
    if index not in lock["permutation_indices"]:
        raise ValueError(f"permutation index {index} is not locked")
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"permutation_{index:02d}.json"
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")

    original = wp2.Dataset.load(activation_dir)
    families = observation_target_families(original.rows)
    swapped = swap_assignment(families, int(lock["master_seed"]), index)
    data = permuted_dataset(original, swapped)
    configs = wp2.candidate_configs([int(block) for block in data.blocks])
    nested = wp2.nested_selection(data, configs)
    target_families = wp2.all_families(data.rows, "wp3")
    control_families = wp2.all_families(data.rows, "wp1")
    full_cv = wp2.cross_validate_configs(
        data, configs, target_families, control_families, 4, wp2.MASTER_SEED
    )
    apparent_pass_count = sum(
        item["valid"] and screen_passes(item["metrics"]) for item in full_cv
    )
    diagnostic = wp2.choose_candidate(full_cv, [int(block) for block in data.blocks])
    sites = [
        {"block": int(fold["chosen_config"]["block"]),
         "role": str(fold["chosen_config"]["role"])}
        for fold in nested["folds"]
    ]
    blocks = [site["block"] for site in sites]
    result = {
        "schema": "empathy-action-probes/wp2-paired-permutation/1",
        "experiment_id": lock["experiment_id"],
        "permutation_index": index,
        "claim_authorized": False,
        "confirmation_authorized": False,
        "swapped_families": sorted(swapped),
        "swapped_family_count": len(swapped),
        "candidate_count": len(configs),
        "valid_candidate_count": sum(item["valid"] for item in full_cv),
        "apparent_pass_count": int(apparent_pass_count),
        "nested_screen_passed": screen_passes(nested["metrics"]),
        "nested_metrics": nested["metrics"],
        "outer_sites": sites,
        "unique_site_count": len({(site["block"], site["role"]) for site in sites}),
        "unique_block_count": len(set(blocks)),
        "block_span": max(blocks) - min(blocks),
        "best_full_development_diagnostic": diagnostic["config"],
        "lock_sha256": wp2.sha256(LOCK_PATH),
        "runner_sha256": wp2.sha256(Path(__file__).resolve()),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "permutation_index", "swapped_family_count", "apparent_pass_count",
        "nested_screen_passed", "outer_sites", "block_span"
    )}, indent=2))


def observed_summary(observed: dict[str, object]) -> dict[str, object]:
    apparent = sum(
        item["valid"] and screen_passes(item["metrics"])
        for item in observed["full_development_cv"]
    )
    sites = [
        {"block": int(fold["chosen_config"]["block"]),
         "role": str(fold["chosen_config"]["role"])}
        for fold in observed["nested_outer"]["folds"]
    ]
    blocks = [site["block"] for site in sites]
    return {
        "apparent_pass_count": int(apparent),
        "nested_screen_passed": screen_passes(observed["nested_outer"]["metrics"]),
        "outer_sites": sites,
        "unique_site_count": len({(site["block"], site["role"]) for site in sites}),
        "unique_block_count": len(set(blocks)),
        "block_span": max(blocks) - min(blocks),
    }


def plus_one_tail(null_values: list[float], observed: float, tail: str) -> float:
    if tail == "upper":
        extreme = sum(value >= observed for value in null_values)
    elif tail == "lower":
        extreme = sum(value <= observed for value in null_values)
    else:
        raise ValueError(f"unknown tail: {tail}")
    return (1 + extreme) / (len(null_values) + 1)


def mean_pairwise_block_distance(sites: list[dict[str, object]]) -> float:
    blocks = [int(site["block"]) for site in sites]
    distances = [abs(left - right) for left, right in combinations(blocks, 2)]
    if not distances:
        raise ValueError("at least two outer-fold sites are required")
    return float(statistics.mean(distances))


def aggregate(activation_dir: Path, observed_path: Path, out_dir: Path) -> None:
    lock = load_and_validate_lock(activation_dir, observed_path)
    output = out_dir / "permutation_calibration.json"
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    permutations = []
    for index in lock["permutation_indices"]:
        path = out_dir / f"permutation_{index:02d}.json"
        if not path.exists():
            raise ValueError(f"missing locked permutation result: {path}")
        item = json.loads(path.read_text())
        if item["permutation_index"] != index:
            raise ValueError(f"permutation index mismatch in {path}")
        permutations.append(item)

    observed = observed_summary(json.loads(observed_path.read_text()))
    observed["mean_pairwise_block_distance"] = mean_pairwise_block_distance(
        observed["outer_sites"]
    )
    counts = [item["apparent_pass_count"] for item in permutations]
    unique_sites = [item["unique_site_count"] for item in permutations]
    spans = [item["block_span"] for item in permutations]
    pairwise_distances = [
        mean_pairwise_block_distance(item["outer_sites"]) for item in permutations
    ]
    report = {
        "schema": "empathy-action-probes/wp2-permutation-calibration/1",
        "experiment_id": lock["experiment_id"],
        "claim_authorized": False,
        "confirmation_authorized": False,
        "permutation_count": len(permutations),
        "observed": observed,
        "null": {
            "apparent_pass_count": {
                "values": counts,
                "min": min(counts),
                "median": statistics.median(counts),
                "mean": statistics.mean(counts),
                "max": max(counts),
                "inferential_use": "none: target permutation destroys decodability",
            },
            "nested_screen_pass_count": sum(
                item["nested_screen_passed"] for item in permutations
            ),
            "primary_fold_convergence": {
                "statistic": "mean_pairwise_absolute_block_distance",
                "values": pairwise_distances,
                "median": statistics.median(pairwise_distances),
                "observed_lower_tail_p_plus_one": plus_one_tail(
                    pairwise_distances,
                    observed["mean_pairwise_block_distance"],
                    "lower",
                ),
                "direction": "lower means more convergence",
            },
            "unique_site_count": {
                "values": unique_sites,
                "observed_lower_tail_p_plus_one": plus_one_tail(
                    unique_sites, observed["unique_site_count"], "lower"
                ),
            },
            "block_span": {
                "values": spans,
                "median": statistics.median(spans),
                "observed_lower_tail_p_plus_one": plus_one_tail(
                    spans, observed["block_span"], "lower"
                ),
            },
        },
        "permutations": permutations,
        "lock_sha256": wp2.sha256(LOCK_PATH),
        "observed_selection_sha256": wp2.sha256(observed_path),
        "runner_sha256": wp2.sha256(Path(__file__).resolve()),
        "interpretation_ceiling": (
            "Calibration of outer-fold site convergence only. Apparent-pass counts are "
            "descriptive because the permutation destroys target decodability. Human "
            "validation remains load-bearing, and no result authorizes confirmation or "
            "block-3 selection."
        ),
        "interpretation_amendment": (
            "notes/WP2_PERMUTATION_CALIBRATION_AMENDMENT_2026-07-13.md"
        ),
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"observed": observed, "null": report["null"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "aggregate"))
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.mode == "run":
        if args.index is None:
            parser.error("run requires --index")
        run_one(args.activations, args.observed, args.out, args.index)
    else:
        if args.index is not None:
            parser.error("aggregate does not accept --index")
        aggregate(args.activations, args.observed, args.out)


if __name__ == "__main__":
    main()
