"""E28b: E18 slope-interaction null distribution over matched four-head sets.

Codex (E28 + rescue3c design review): the composition-matched random set
showed a NONZERO cost-slope interaction (-0.025), so zero is not an adequate
null for direction-removal edits on this battery. This builds the empirical
null the suppressor slope claim needs. Design per codex:
  - LAYER-MULTISET-MATCHED sets: exactly one head from each targeted layer
    (L17/L18/L19/L20), targeted heads excluded (primary disjoint null)
  - >= 20 distinct sets (default 24; 0/24 gives min p = 1/25 = .04)
  - statistic = mean per-family cost-slope difference (welfare - nonsocial)
  - plus a SAME-HEAD random-DIRECTION control (--n-random-directions):
    the targeted four heads edited along random unit directions — tests
    direction-specificity of the slope signature, which component nulls can't
Realized rank-1 delta norms recorded per component throughout.

Usage (Spark, `empathy` env):
  python -u src/e28b_slope_nulls.py \
    --direction results/controlled_directions_gemma2_9b_it/direction_M_resid_block20.npy \
    --suppressors "L18H13,L20H10,L19H12,L17H7" \
    --n-sets 24 --n-random-directions 8 --out results/e28b_slope_nulls_gemma
"""

import argparse
import json
import random as pyrandom
from pathlib import Path

import numpy as np
import torch

try:
    from src.weight_orthogonalization import (
        choice_scores, load_pairs, orthogonalize_component, parse_component,
        restore_weights, snapshot_weights,
    )
    from src.norm_matched_controls import targeted_delta_norm
except ModuleNotFoundError:
    from weight_orthogonalization import (
        choice_scores, load_pairs, orthogonalize_component, parse_component,
        restore_weights, snapshot_weights,
    )
    from norm_matched_controls import targeted_delta_norm

N_HEADS = 16  # gemma-2-9b


def layer_multiset_matched(rng, target_names, n_sets):
    """distinct sets with exactly one head per targeted layer, targets excluded."""
    layers = sorted(parse_component(n)["layer"] for n in target_names)
    pools = []
    for l in layers:
        pools.append([f"L{l}H{h}" for h in range(N_HEADS)
                      if f"L{l}H{h}" not in target_names])
    seen, sets = set(), []
    while len(sets) < n_sets:
        s = tuple(sorted(rng.choice(pool) for pool in pools))
        if s not in seen:
            seen.add(s)
            sets.append(list(s))
    return sets

CELLS = {
    "cost_axis": "data/contrastive_pairs/v2_2/cost_axis_templated.jsonl",
    "nonsocial_axis": "data/contrastive_pairs/v2_2/nonsocial_axis_templated.jsonl",
}
COST_RANK = {"free": 0, "low": 1, "medium": 2, "high": 3}


def slope_diff(deltas, meta):
    """mean over families of (welfare cost slope - nonsocial cost slope)."""
    per_axis = {}
    for axis in CELLS:
        d, fams, ranks = deltas[axis], meta[axis]["fams"], meta[axis]["ranks"]
        per_axis[axis] = {f: float(np.polyfit(ranks[fams == f], d[fams == f], 1)[0])
                          for f in sorted(set(fams))}
    fams = sorted(set(per_axis["cost_axis"]) & set(per_axis["nonsocial_axis"]))
    diffs = [per_axis["cost_axis"][f] - per_axis["nonsocial_axis"][f] for f in fams]
    return float(np.mean(diffs)), {f: round(d, 4) for f, d in zip(fams, diffs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--direction", required=True)
    ap.add_argument("--suppressors", required=True)
    ap.add_argument("--n-sets", type=int, default=24)
    ap.add_argument("--n-random-directions", type=int, default=8,
                    help="same-head random-direction slope controls (0 = skip)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/e28b_slope_nulls_gemma")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager").to(device)
    model.eval()
    direction = torch.tensor(np.load(args.direction), dtype=torch.float32, device=device)
    direction /= torch.linalg.vector_norm(direction)

    cells = {n: load_pairs(p) for n, p in CELLS.items()}
    meta = {n: {"fams": np.array([p["scenario_id"] for p in prs]),
                "ranks": np.array([COST_RANK[p["cost_level"]] for p in prs])}
            for n, prs in cells.items()}

    def eval_cells():
        return {n: np.asarray(choice_scores(model, tok, prs, args.seed,
                                            args.batch_size, args.max_tokens, device))
                for n, prs in cells.items()}

    def run_set(names, edit_direction=None):
        ed = direction if edit_direction is None else edit_direction
        comps = [parse_component(n) for n in names]
        norms = {c["name"]: round(targeted_delta_norm(model, c, ed), 3)
                 for c in comps}
        snap = snapshot_weights(model, comps)
        try:
            for c in comps:
                orthogonalize_component(model, c, ed)
            edited = eval_cells()
        finally:
            restore_weights(snap)
        deltas = {n: edited[n] - base[n] for n in cells}
        sd, per_fam = slope_diff(deltas, meta)
        return sd, per_fam, norms

    base = eval_cells()
    print("baseline uptake:", {n: round(float(v.mean()), 3) for n, v in base.items()})

    target_names = args.suppressors.split(",")
    rng = pyrandom.Random(args.seed)
    null_names = layer_multiset_matched(rng, target_names, args.n_sets)
    print(f"{len(null_names)} layer-multiset-matched null sets "
          f"(one head per targeted layer, targets excluded)")

    tgt_sd, tgt_fam, tgt_norms = run_set(target_names)
    print(f"targeted {target_names}: slope-diff {tgt_sd:+.4f}")
    rows = []
    for i, names in enumerate(null_names):
        sd, per_fam, norms = run_set(names)
        rows.append({"set": names, "slope_diff": sd, "per_family": per_fam,
                     "delta_norms": norms})
        print(f"  null {i+1}/{len(null_names)} {names}: {sd:+.4f}")

    nd = np.array([r["slope_diff"] for r in rows])
    res = {
        "direction": args.direction,
        "targeted_set": target_names, "targeted_slope_diff": tgt_sd,
        "targeted_per_family": tgt_fam, "targeted_delta_norms": tgt_norms,
        "null_sets": rows, "null_mean": float(nd.mean()),
        "null_std": float(nd.std(ddof=1)),
        "null_range": [float(nd.min()), float(nd.max())],
        "z_secondary": float((tgt_sd - nd.mean()) / nd.std(ddof=1)),
        "n_null_as_extreme": int((np.abs(nd) >= abs(tgt_sd)).sum()),
        "exact_p": (1 + int((np.abs(nd) >= abs(tgt_sd)).sum())) / (len(rows) + 1),
    }
    print(f"targeted {tgt_sd:+.4f} | null {res['null_mean']:+.4f}±{res['null_std']:.4f} "
          f"| {res['n_null_as_extreme']}/{len(rows)} as extreme | p={res['exact_p']:.3f}")

    if args.n_random_directions:
        torch_gen = np.random.default_rng(args.seed)
        drows = []
        for i in range(args.n_random_directions):
            rd = torch.tensor(torch_gen.standard_normal(direction.shape[0]),
                              dtype=torch.float32, device=device)
            rd /= torch.linalg.vector_norm(rd)
            sd, per_fam, norms = run_set(target_names, edit_direction=rd)
            drows.append({"slope_diff": sd, "delta_norms": norms})
            print(f"  same-head random-dir {i+1}/{args.n_random_directions}: {sd:+.4f}")
        dd = np.array([r["slope_diff"] for r in drows])
        res["same_head_random_directions"] = {
            "rows": drows, "mean": float(dd.mean()), "std": float(dd.std(ddof=1)),
            "n_as_extreme": int((np.abs(dd) >= abs(tgt_sd)).sum()),
        }
        print(f"same-head random-dir null: {dd.mean():+.4f}±{dd.std(ddof=1):.4f} | "
              f"{res['same_head_random_directions']['n_as_extreme']}/{len(dd)} as extreme")

    (out / "slope_nulls.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {out}/slope_nulls.json")


if __name__ == "__main__":
    main()
