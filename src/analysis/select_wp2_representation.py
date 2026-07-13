"""WP2 family-grouped nested-CV representation selection.

Consumes development activations produced by `extract_gate2_activations.py`.
It never reads confirmation data. The output freezes one representation or an
explicit no-representation outcome before confirmation can be extracted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
MASTER_SEED = 1973658643
QUIET_CONTROLS = {
    "T_new": ("interrupt", "persist"),
    "D_new": ("warm", "terse"),
    "P_new": ("caring", "neutral"),
    "G_new": ("genuine", "strategic"),
    "B_new": ("active_zero_cost", "no_active_objective"),
    "Spos_new": ("positive", "neutral"),
    "O_new": ("available", "unavailable"),
    "Ctext_new": ("high", "zero"),
}
DIAGNOSTICS = {
    "L_new": ("current_actual", "archived_actual"),
    "R_new": ("welfare_observation", "neutral_observation"),
}
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0)
SUBSPACE_DIMS = (2, 4, 8)
NUISANCE_DIMS = (1, 2, 4, 8)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derived_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-10:
        raise ValueError("zero-norm representation")
    return vector / norm


@dataclass
class Dataset:
    activations: np.ndarray
    blocks: np.ndarray
    rows: list[dict[str, object]]

    @classmethod
    def load(cls, directory: Path) -> "Dataset":
        plan = json.loads((directory / "extraction_plan.json").read_text())
        if plan["phase"] != "dev":
            raise ValueError("WP2 selection accepts development activations only")
        saved = np.load(directory / "activations.npz")
        rows = [json.loads(line) for line in (directory / "rows.jsonl").read_text().splitlines()]
        activations = saved["activations"].astype(np.float32)
        blocks = saved["blocks"]
        if len(rows) != len(activations):
            raise ValueError("activation/row length mismatch")
        if any("confirm" in str(row["partition"]).lower() for row in rows):
            raise ValueError("confirmation rows found in development artifact")
        return cls(activations, blocks, rows)

    def block_index(self, block: int) -> int:
        found = np.flatnonzero(self.blocks == block)
        if len(found) != 1:
            raise ValueError(f"block {block} absent or duplicated")
        return int(found[0])

    def select(self, block: int, **fields: object) -> tuple[np.ndarray, list[dict[str, object]]]:
        indices = [
            index for index, row in enumerate(self.rows)
            if all(row.get(key) == value for key, value in fields.items())
        ]
        return self.activations[indices, self.block_index(block)], [self.rows[i] for i in indices]


def source_balanced_folds(rows: list[dict[str, object]], n_folds: int, seed: int) -> dict[str, int]:
    families: dict[str, str] = {}
    for row in rows:
        family = str(row["family_id"])
        source = str(row["source"])
        if family in families and families[family] != source:
            raise ValueError("family crosses source strata")
        families[family] = source
    assignment: dict[str, int] = {}
    for source in sorted(set(families.values())):
        group = sorted(family for family, value in families.items() if value == source)
        rng = np.random.default_rng(derived_seed(seed, source))
        rng.shuffle(group)
        for index, family in enumerate(group):
            assignment[family] = index % n_folds
    if set(assignment) != set(families):
        raise ValueError("incomplete fold assignment")
    return assignment


def subset_families(
    X: np.ndarray, rows: list[dict[str, object]], families: set[str]
) -> tuple[np.ndarray, list[dict[str, object]]]:
    indices = [i for i, row in enumerate(rows) if row["family_id"] in families]
    return X[indices], [rows[i] for i in indices]


def family_contrasts(
    X: np.ndarray,
    rows: list[dict[str, object]],
    positive: str,
    negative: str,
) -> np.ndarray:
    by_family: dict[str, dict[str, list[np.ndarray]]] = {}
    for vector, row in zip(X, rows, strict=True):
        arm = str(row["arm_id"])
        if arm not in {positive, negative}:
            continue
        by_family.setdefault(str(row["family_id"]), {}).setdefault(arm, []).append(vector)
    contrasts = []
    for family in sorted(by_family):
        arms = by_family[family]
        if positive not in arms or negative not in arms:
            raise ValueError(f"family {family} lacks {positive}/{negative}")
        contrasts.append(np.mean(arms[positive], axis=0) - np.mean(arms[negative], axis=0))
    if not contrasts:
        raise ValueError("no complete family contrasts")
    return np.stack(contrasts)


def target_rows(
    data: Dataset, block: int, role: str
) -> tuple[np.ndarray, list[dict[str, object]]]:
    X, rows = data.select(
        block,
        dataset="wp3",
        cell="observation",
        readout_role=role,
    )
    keep = [i for i, row in enumerate(rows) if row["arm_id"] in {"current_actual", "archived_actual"}]
    return X[keep], [rows[i] for i in keep]


def control_rows(
    data: Dataset, block: int, contrast: str
) -> tuple[np.ndarray, list[dict[str, object]]]:
    return data.select(
        block,
        dataset="wp1",
        contrast=contrast,
        readout_role="prompt_final",
    )


def family_weights(rows: list[dict[str, object]]) -> np.ndarray:
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["family_id"])] = counts.get(str(row["family_id"]), 0) + 1
    weights = np.asarray([1.0 / counts[str(row["family_id"])] for row in rows])
    return weights / weights.sum()


def fit_ridge_score(
    X: np.ndarray,
    rows: list[dict[str, object]],
    basis: np.ndarray,
    alpha: float,
) -> dict[str, np.ndarray | float]:
    Z = X @ basis
    weights = family_weights(rows)
    center = np.sum(Z * weights[:, None], axis=0)
    scale = np.sqrt(np.sum(((Z - center) ** 2) * weights[:, None], axis=0))
    scale = np.where(scale < 1e-8, 1.0, scale)
    standardized = (Z - center) / scale
    y = np.asarray([row["arm_id"] == "current_actual" for row in rows], dtype=float)
    intercept = float(np.sum(y * weights))
    centered_y = y - intercept
    gram = standardized.T @ (standardized * weights[:, None])
    rhs = standardized.T @ (centered_y * weights)
    coef = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), rhs)
    return {"basis": basis, "center": center, "scale": scale, "coef": coef, "intercept": intercept}


def score(model: dict[str, object], X: np.ndarray) -> np.ndarray:
    Z = X @ np.asarray(model["basis"])
    Z = (Z - np.asarray(model["center"])) / np.asarray(model["scale"])
    return Z @ np.asarray(model["coef"]) + float(model["intercept"])


def family_arm_scores(
    values: np.ndarray, rows: list[dict[str, object]], positive: str, negative: str
) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for value, row in zip(values, rows, strict=True):
        arm = str(row["arm_id"])
        if arm in {positive, negative}:
            grouped.setdefault((str(row["family_id"]), arm), []).append(float(value))
    pos, neg = [], []
    for family in sorted({key[0] for key in grouped}):
        if (family, positive) not in grouped or (family, negative) not in grouped:
            raise ValueError(f"incomplete score pair for {family}")
        pos.append(np.mean(grouped[(family, positive)]))
        neg.append(np.mean(grouped[(family, negative)]))
    return np.asarray(pos), np.asarray(neg)


def paired_auroc(values: np.ndarray, rows: list[dict[str, object]], arms: tuple[str, str]) -> float:
    pos, neg = family_arm_scores(values, rows, *arms)
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(y, np.r_[pos, neg]))


def nuisance_matrix(
    data: Dataset, block: int, families: set[str]
) -> np.ndarray:
    vectors = []
    for contrast, arms in QUIET_CONTROLS.items():
        X, rows = control_rows(data, block, contrast)
        X, rows = subset_families(X, rows, families)
        diffs = family_contrasts(X, rows, *arms)
        vectors.extend(normalize(vector) for vector in diffs)
    return np.stack(vectors)


def candidate_configs(blocks: list[int]) -> list[dict[str, object]]:
    configs = []
    for block in blocks:
        for role in ("quote_boundary", "prompt_final"):
            configs.append({"kind": "mean", "block": block, "role": role, "dim": 1, "alpha": None})
            for dim in NUISANCE_DIMS:
                configs.append({"kind": "residualized", "block": block, "role": role,
                                "dim": 1, "nuisance_dim": dim, "alpha": None})
            for alpha in RIDGE_ALPHAS:
                configs.append({"kind": "plane", "block": block, "role": role,
                                "dim": 2, "alpha": alpha})
            for dim in SUBSPACE_DIMS:
                for alpha in RIDGE_ALPHAS:
                    configs.append({"kind": "supervised", "block": block, "role": role,
                                    "dim": dim, "alpha": alpha})
    return configs


def fit_candidate(
    data: Dataset,
    config: dict[str, object],
    target_families: set[str],
    control_families: set[str],
) -> dict[str, object]:
    block, role = int(config["block"]), str(config["role"])
    X, rows = target_rows(data, block, role)
    X, rows = subset_families(X, rows, target_families)
    diffs = family_contrasts(X, rows, "current_actual", "archived_actual")
    d_target = normalize(np.mean(diffs, axis=0))
    kind = config["kind"]
    if kind == "mean":
        basis = d_target[:, None]
        return {"basis": basis, "center": np.zeros(1), "scale": np.ones(1),
                "coef": np.ones(1), "intercept": 0.0}
    if kind == "residualized":
        matrix = nuisance_matrix(data, block, control_families)
        _, _, vt = np.linalg.svd(matrix, full_matrices=False)
        nuisance_basis = vt[: int(config["nuisance_dim"])].T
        direction = normalize(d_target - nuisance_basis @ (nuisance_basis.T @ d_target))
        return {"basis": direction[:, None], "center": np.zeros(1), "scale": np.ones(1),
                "coef": np.ones(1), "intercept": 0.0}
    if kind == "plane":
        Xt, rt = control_rows(data, block, "T_new")
        Xt, rt = subset_families(Xt, rt, control_families)
        d_task = normalize(np.mean(family_contrasts(Xt, rt, *QUIET_CONTROLS["T_new"]), axis=0))
        q, _ = np.linalg.qr(np.stack([d_target, d_task], axis=1))
        basis = q[:, :2]
    elif kind == "supervised":
        normalized_diffs = np.stack([normalize(vector) for vector in diffs])
        _, _, vt = np.linalg.svd(normalized_diffs, full_matrices=False)
        basis = vt[: int(config["dim"])].T
    else:
        raise ValueError(f"unknown candidate kind: {kind}")
    return fit_ridge_score(X, rows, basis, float(config["alpha"]))


def evaluate_candidate(
    data: Dataset,
    config: dict[str, object],
    model: dict[str, object],
    target_families: set[str],
    control_families: set[str],
) -> dict[str, object]:
    block, role = int(config["block"]), str(config["role"])
    X, rows = target_rows(data, block, role)
    X, rows = subset_families(X, rows, target_families)
    values = score(model, X)
    target_auc = paired_auroc(values, rows, ("current_actual", "archived_actual"))
    y = np.asarray([row["arm_id"] == "current_actual" for row in rows], dtype=float)
    result: dict[str, object] = {
        "target_auroc": target_auc,
        "target_mse": float(mean_squared_error(y, values, sample_weight=family_weights(rows))),
        "quiet_auroc": {},
        "diagnostic_auroc": {},
    }
    for target, store in ((QUIET_CONTROLS, "quiet_auroc"), (DIAGNOSTICS, "diagnostic_auroc")):
        for contrast, arms in target.items():
            Xc, rc = control_rows(data, block, contrast)
            Xc, rc = subset_families(Xc, rc, control_families)
            result[store][contrast] = paired_auroc(score(model, Xc), rc, arms)
    result["max_quiet_two_sided"] = max(
        max(value, 1 - value) for value in result["quiet_auroc"].values()
    )
    return result


def pool_metrics(metrics: list[dict[str, object]]) -> dict[str, object]:
    return {
        "target_auroc": float(np.mean([item["target_auroc"] for item in metrics])),
        "target_mse": float(np.mean([item["target_mse"] for item in metrics])),
        "quiet_auroc": {
            contrast: float(np.mean([item["quiet_auroc"][contrast] for item in metrics]))
            for contrast in QUIET_CONTROLS
        },
        "diagnostic_auroc": {
            contrast: float(np.mean([item["diagnostic_auroc"][contrast] for item in metrics]))
            for contrast in DIAGNOSTICS
        },
    }


def config_key(config: dict[str, object]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def choose_candidate(scored: list[dict[str, object]], blocks: list[int]) -> dict[str, object]:
    valid = [item for item in scored if item.get("valid", True)]
    if not valid:
        raise ValueError("all candidate fits failed")
    best_auc = max(item["metrics"]["target_auroc"] for item in valid)
    eligible = [item for item in valid if item["metrics"]["target_auroc"] >= best_auc - 0.02]
    for item in eligible:
        quiet = item["metrics"]["quiet_auroc"]
        item["selection_max_quiet_two_sided"] = max(max(v, 1 - v) for v in quiet.values())
    middle = min(blocks, key=lambda block: (abs((block + 1) / 42 - 0.48), block))
    return min(
        eligible,
        key=lambda item: (
            item["selection_max_quiet_two_sided"],
            int(item["config"]["dim"]),
            abs(int(item["config"]["block"]) - middle),
            item["config"]["role"] != "prompt_final",
            float(item["config"]["alpha"] or 0.0),
            config_key(item["config"]),
        ),
    )


def all_families(rows: list[dict[str, object]], dataset: str) -> set[str]:
    return {str(row["family_id"]) for row in rows if row["dataset"] == dataset}


def cross_validate_configs(
    data: Dataset,
    configs: list[dict[str, object]],
    target_families: set[str],
    control_families: set[str],
    n_folds: int,
    seed: int,
) -> list[dict[str, object]]:
    target_rows_all = [row for row in data.rows if row["dataset"] == "wp3" and row["family_id"] in target_families]
    control_rows_all = [row for row in data.rows if row["dataset"] == "wp1" and row["family_id"] in control_families]
    tfolds = source_balanced_folds(target_rows_all, n_folds, seed)
    cfolds = source_balanced_folds(control_rows_all, n_folds, seed)
    output = []
    for config in configs:
        fold_metrics, errors = [], []
        for fold in range(n_folds):
            ttrain = {family for family in target_families if tfolds[family] != fold}
            ttest = target_families - ttrain
            ctrain = {family for family in control_families if cfolds[family] != fold}
            ctest = control_families - ctrain
            try:
                model = fit_candidate(data, config, ttrain, ctrain)
                fold_metrics.append(evaluate_candidate(data, config, model, ttest, ctest))
            except (ValueError, np.linalg.LinAlgError) as exc:
                errors.append({"fold": fold, "error": str(exc)})
        output.append({
            "config": config,
            "valid": not errors and len(fold_metrics) == n_folds,
            "metrics": pool_metrics(fold_metrics) if fold_metrics else None,
            "fold_metrics": fold_metrics,
            "errors": errors,
        })
    return output


def nested_selection(data: Dataset, configs: list[dict[str, object]]) -> dict[str, object]:
    target_families = all_families(data.rows, "wp3")
    control_families = all_families(data.rows, "wp1")
    tfolds = source_balanced_folds(
        [row for row in data.rows if row["dataset"] == "wp3"], 4, MASTER_SEED
    )
    cfolds = source_balanced_folds(
        [row for row in data.rows if row["dataset"] == "wp1"], 4, MASTER_SEED
    )
    outer = []
    for fold in range(4):
        ttrain = {family for family in target_families if tfolds[family] != fold}
        ttest = target_families - ttrain
        ctrain = {family for family in control_families if cfolds[family] != fold}
        ctest = control_families - ctrain
        inner = cross_validate_configs(
            data, configs, ttrain, ctrain, 3, derived_seed(MASTER_SEED, fold, "WP2 inner")
        )
        chosen = choose_candidate(inner, [int(block) for block in data.blocks])
        model = fit_candidate(data, chosen["config"], ttrain, ctrain)
        metrics = evaluate_candidate(data, chosen["config"], model, ttest, ctest)
        outer.append({
            "fold": fold,
            "chosen_config": chosen["config"],
            "inner_candidates": inner,
            "metrics": metrics,
        })
    return {"folds": outer, "metrics": pool_metrics([item["metrics"] for item in outer])}


def serialize_model(model: dict[str, object], path: Path) -> None:
    np.savez_compressed(path, **{key: np.asarray(value) for key, value in model.items()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError(f"refusing to overwrite selection output: {args.out}")
    args.out.mkdir(parents=True)

    data = Dataset.load(args.activations)
    configs = candidate_configs([int(block) for block in data.blocks])
    nested = nested_selection(data, configs)
    target_families = all_families(data.rows, "wp3")
    control_families = all_families(data.rows, "wp1")
    full_cv = cross_validate_configs(data, configs, target_families, control_families, 4, MASTER_SEED)
    chosen = choose_candidate(full_cv, [int(block) for block in data.blocks])
    nested_quiet = nested["metrics"]["quiet_auroc"]
    passes = (
        nested["metrics"]["target_auroc"] >= 0.75
        and all(abs(value - 0.5) <= 0.10 for value in nested_quiet.values())
    )
    outcome = "representation" if passes else "no-representation"
    fitted_path = None
    if passes:
        model = fit_candidate(data, chosen["config"], target_families, control_families)
        fitted_path = args.out / "frozen_representation.npz"
        serialize_model(model, fitted_path)

    report = {
        "schema": "empathy-action-probes/wp2-selection/1",
        "model": "google/gemma-2-9b-it",
        "revision": "11c9b309abf73637e4b6f9a3fa1e92e615547819",
        "master_seed": MASTER_SEED,
        "implementation_lock": "notes/WP2_IMPLEMENTATION_LOCK_2026-07-13.md",
        "implementation_lock_sha256": sha256(
            ROOT / "notes/WP2_IMPLEMENTATION_LOCK_2026-07-13.md"
        ),
        "code": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "development_activations": {
            "path": str(args.activations),
            "plan_sha256": sha256(args.activations / "extraction_plan.json"),
            "activations_sha256": sha256(args.activations / "activations.npz"),
            "rows_sha256": sha256(args.activations / "rows.jsonl"),
        },
        "candidate_count": len(configs),
        "nested_outer": nested,
        "full_development_cv": full_cv,
        "selection": {
            "outcome": outcome,
            "chosen_config": chosen["config"] if passes else None,
            "best_development_config_diagnostic": chosen["config"],
            "nested_target_auroc": nested["metrics"]["target_auroc"],
            "nested_quiet_auroc": nested_quiet,
            "development_gate_passed": passes,
            "frozen_representation": fitted_path.name if fitted_path else None,
            "frozen_representation_sha256": sha256(fitted_path) if fitted_path else None,
        },
    }
    (args.out / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["selection"], indent=2))


if __name__ == "__main__":
    main()
