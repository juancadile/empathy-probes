"""E18: cost-axis dose-response + social/non-social interaction (issue #34).

Evaluates weight-edit conditions on the two clause-matched V2.2 axes:
  cost_axis      — competing alternative is a DISTRESSED person (welfare-laden)
  nonsocial_axis — competing alternative is an EXCITED person (no welfare stake)
Both share the same 10 scenario families, cost stems, and universal decision
clauses; pos = uptake of the alternative, neg = stay on own work.

Outputs per condition:
  - delta vs baseline per (cell, cost_level), family-clustered bootstrap CIs
  - cost dose-response slope (delta vs cost_rank) per cell
  - THE KEY STATISTIC (codex-specified, E14c sparring): the edit x
    alternative-type interaction = d_uptake(cost_axis) - d_uptake(nonsocial),
    bootstrapped by resampling the SHARED families once per replicate so the
    interaction CI respects the pairing across cells.
Predictions: suppressors = salient-alternative capture -> interaction ~ 0;
suppressors = welfare reweighting -> interaction > 0. Writers (helping-
specific) -> negative shift concentrated on cost_axis -> interaction < 0.

Usage (Spark, `empathy` env):
  python -u src/e18_interaction.py \
    --model google/gemma-2-9b-it \
    --direction results/controlled_directions_gemma2_9b_it/direction_M_block20.npy \
    --block 20 --out results/e18_interaction_gemma2_9b_it
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

try:
    from src.weight_orthogonalization import (
        choice_scores, load_pairs, orthogonalize_component, parse_component,
        restore_weights, snapshot_weights,
    )
except ModuleNotFoundError:
    from weight_orthogonalization import (
        choice_scores, load_pairs, orthogonalize_component, parse_component,
        restore_weights, snapshot_weights,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("e18")

CELLS = {
    "cost_axis": "data/contrastive_pairs/v2_2/cost_axis_templated.jsonl",
    "nonsocial_axis": "data/contrastive_pairs/v2_2/nonsocial_axis_templated.jsonl",
}
GEMMA_CONDITIONS = {
    "positive_writers_k2": "L19MLP,L20H15",
    "suppressors_k4": "L19H12,L15H15,L17H13,L18H13",
    "targeted_k6": "L19MLP,L20H15,L19H12,L15H15,L17H13,L18H13",
    "random_k6": "L1MLP,L19H7,L15H9,L17H5,L18H12,L20H12",
}
COST_LEVELS = ["free", "low", "medium", "high"]


def interaction_bootstrap(d_cost, d_non, fams_cost, fams_non, seed=0, n_boot=5000):
    """CI on mean(d_cost) - mean(d_non), resampling shared families jointly."""
    fams_cost, fams_non = np.asarray(fams_cost), np.asarray(fams_non)
    unique = sorted(set(fams_cost))
    assert set(unique) == set(fams_non)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(len(unique), len(unique), replace=True)
        c = np.concatenate([d_cost[fams_cost == unique[i]] for i in pick])
        n = np.concatenate([d_non[fams_non == unique[i]] for i in pick])
        stats.append(c.mean() - n.mean())
    return {"mean": float(d_cost.mean() - d_non.mean()),
            "ci95": [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]}


def clustered_ci(d, fams, seed=0, n_boot=5000):
    fams = np.asarray(fams)
    unique = sorted(set(fams))
    rng = np.random.default_rng(seed)
    means = [np.concatenate([d[fams == unique[i]]
                             for i in rng.choice(len(unique), len(unique), replace=True)]).mean()
             for _ in range(n_boot)]
    return {"mean": float(d.mean()),
            "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--direction", required=True)
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cells", nargs="+", default=None,
                    help="name=path overrides for CELLS")
    ap.add_argument("--out", default="results/e18_interaction_gemma2_9b_it")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager").to(device)
    model.eval()
    direction = torch.tensor(np.load(args.direction), dtype=torch.float32, device=device)
    direction /= torch.linalg.vector_norm(direction)

    cell_paths = (dict(s.split("=", 1) for s in args.cells) if args.cells else CELLS)
    cells = {n: load_pairs(p) for n, p in cell_paths.items()}
    meta = {n: {"fams": [p["scenario_id"] for p in pairs],
                "cost": [p["cost_level"] for p in pairs]} for n, pairs in cells.items()}

    def eval_cells():
        return {n: np.asarray(choice_scores(model, tokenizer, pairs, args.seed,
                                            args.batch_size, args.max_tokens, device))
                for n, pairs in cells.items()}

    log.info("baseline eval")
    base = eval_cells()
    results = {"model": args.model, "block": args.block,
               "baseline": {n: {"mean": float(v.mean()),
                                "by_cost": {c: float(v[np.asarray(meta[n]["cost"]) == c].mean())
                                            for c in COST_LEVELS}}
                            for n, v in base.items()},
               "conditions": {}}
    log.info("baseline uptake: %s", {n: round(float(v.mean()), 3) for n, v in base.items()})

    for cond, spec in GEMMA_CONDITIONS.items():
        comps = [parse_component(v) for v in spec.split(",")]
        snap = snapshot_weights(model, comps)
        try:
            for c in comps:
                orthogonalize_component(model, c, direction)
            edited = eval_cells()
        finally:
            restore_weights(snap)
        entry = {}
        d = {n: edited[n] - base[n] for n in cells}
        for n in cells:
            fams, cost = meta[n]["fams"], np.asarray(meta[n]["cost"])
            entry[n] = {"per_pair_delta": d[n].tolist(),
                        "families": list(fams),
                        "delta_overall": clustered_ci(d[n], fams, args.seed),
                        "delta_by_cost": {c: clustered_ci(d[n][cost == c],
                                                          np.asarray(fams)[cost == c], args.seed)
                                          for c in COST_LEVELS}}
            ranks = np.array([COST_LEVELS.index(c) for c in cost])
            entry[n]["dose_slope"] = float(np.polyfit(ranks, d[n], 1)[0])
        entry["interaction_cost_minus_nonsocial"] = interaction_bootstrap(
            d["cost_axis"], d["nonsocial_axis"],
            meta["cost_axis"]["fams"], meta["nonsocial_axis"]["fams"], args.seed)
        results["conditions"][cond] = entry
        log.info("%s: d_cost %+0.4f | d_nonsoc %+0.4f | interaction %+0.4f CI %s",
                 cond, d["cost_axis"].mean(), d["nonsocial_axis"].mean(),
                 entry["interaction_cost_minus_nonsocial"]["mean"],
                 entry["interaction_cost_minus_nonsocial"]["ci95"])

    (out / "e18.json").write_text(json.dumps(results, indent=2))
    log.info("wrote %s", out / "e18.json")


if __name__ == "__main__":
    main()
