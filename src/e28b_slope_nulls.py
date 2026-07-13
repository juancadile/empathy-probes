"""E28b: E18 slope-interaction null distribution over matched four-head sets.

Codex (E28 + rescue3c design review): the composition-matched random set
showed a NONZERO cost-slope interaction (-0.025), so zero is not an adequate
null for direction-removal edits on this battery. This builds the empirical
null the suppressor slope claim needs. Design (QA-revised):
  - NORM-AND-LAYER-MULTISET-MATCHED sets: exactly one head from each targeted
    layer (L17/L18/L19/L20), targeted heads excluded, drawn per layer from the
    --pool-size heads whose theoretical (pre-cast) rank-1 delta norm under d_resid is
    nearest the targeted head's norm in that layer. The layer multiset is
    preserved exactly; the per-component edit magnitude is matched by
    selection (matching a raw uniform draw would leave edit norm confounded —
    the E26b lesson: the targeted set had the largest delta norms). Matching
    by selection is APPROXIMATE, not exact: each null set's ACTUAL realized
    post-bf16 delta norms are recorded, and total-norm mismatch vs the
    targeted set is computed on both realized and theoretical norms. If every
    null set's |total realized mismatch| <= NULL_TOTAL_NORM_TOL (predeclared
    5%), this establishes approximate TOTAL-norm balance, not exact or
    per-component matching. Otherwise the sets are
    labeled norm-SELECTED only and edit norm must be retained as a
    covariate / stratification variable in any downstream reading.
  - primary inference: two-sided empirical Monte Carlo p =
    (1 + #{|null| >= |targeted|}) / (n_sets + 1). With n_sets=24 the minimum
    attainable p is 1/25 = .04; sets are SAMPLED from the matched pool
    product, not exhaustive. z-score is secondary/descriptive only.
  - statistic = mean per-family cost-slope difference (welfare - nonsocial)
  - SAME-HEAD random-DIRECTION controls (--n-random-directions): the targeted
    set runs FIRST with its ACTUAL realized post-bf16 delta norm measured per
    component (orthogonalize_component_measured); the controls then edit the
    same four heads with NORM-MATCHED rank-1 deltas (apply_norm_matched_
    random): one shared random unit vector per replicate, per-component delta
    scaled to that component's targeted REALIZED norm. Same weights, same
    realized per-component damage size, random content. (Raw orthogonalization
    along a random vector would leave the removed norm uncontrolled.) Each
    control edit's realized norm is gated at 3% relative error from the
    targeted realized norm.

Usage (Spark, `empathy` env):
  python -u src/e28b_slope_nulls.py \
    --direction results/controlled_directions_gemma2_9b_it/direction_M_resid_block20.npy \
    --suppressors "L18H13,L20H10,L19H12,L17H7" \
    --n-sets 24 --n-random-directions 20 --out results/e28b_slope_nulls_gemma
"""

import argparse
import hashlib
import json
import random as pyrandom
from pathlib import Path

import numpy as np
import torch

try:
    from src.weight_orthogonalization import (
        choice_scores, load_pairs, parse_component,
        restore_weights, snapshot_weights,
    )
    from src.norm_matched_controls import (
        apply_norm_matched_random, orthogonalize_component_measured,
        targeted_delta_norm,
    )
except ModuleNotFoundError:
    from weight_orthogonalization import (
        choice_scores, load_pairs, parse_component,
        restore_weights, snapshot_weights,
    )
    from norm_matched_controls import (
        apply_norm_matched_random, orthogonalize_component_measured,
        targeted_delta_norm,
    )

N_HEADS = 16  # gemma-2-9b
# Predeclared tolerance for approximate total realized-norm balance. This does
# not imply exact or per-component norm matching for random-component sets.
NULL_TOTAL_NORM_TOL = 0.05


def norm_matched_multiset_sets(norm_fn, target_names, n_sets, rng, pool_size):
    """Distinct sets, one head per targeted layer, targets excluded, each layer
    pool = pool_size heads with theoretical delta norm nearest the target's.

    norm_fn(name) -> THEORETICAL (float32, pre-bf16-cast) rank-1 delta norm
    for that component. Returns (sets, pools, candidate_norms,
    target_norms_by_layer).
    """
    by_layer = {}
    for n in target_names:
        by_layer[parse_component(n)["layer"]] = n
    pools, cand_norms, tgt_by_layer = [], {}, {}
    for l in sorted(by_layer):
        tnorm = norm_fn(by_layer[l])
        tgt_by_layer[f"L{l}"] = {"component": by_layer[l], "theoretical_norm": tnorm}
        cands = []
        for h in range(N_HEADS):
            name = f"L{l}H{h}"
            if name in target_names:
                continue
            nrm = norm_fn(name)
            cand_norms[name] = nrm
            cands.append((abs(nrm - tnorm), name))
        cands.sort()
        pools.append([name for _, name in cands[:pool_size]])
    n_combos = int(np.prod([len(p) for p in pools]))
    if n_sets > n_combos:
        raise ValueError(f"n_sets={n_sets} > {n_combos} distinct matched sets")
    seen, sets = set(), []
    while len(sets) < n_sets:
        s = tuple(sorted(rng.choice(pool) for pool in pools))
        if s not in seen:
            seen.add(s)
            sets.append(list(s))
    return sets, pools, cand_norms, tgt_by_layer

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


def mc_p(null_values, targeted):
    """Two-sided empirical Monte Carlo p with the +1 correction."""
    null_values = np.asarray(null_values)
    k = int((np.abs(null_values) >= abs(targeted)).sum())
    return k, (1 + k) / (len(null_values) + 1)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--direction", required=True)
    ap.add_argument("--suppressors", required=True)
    ap.add_argument("--n-sets", type=int, default=24)
    ap.add_argument("--pool-size", type=int, default=6,
                    help="per-layer candidate heads nearest the targeted delta norm")
    ap.add_argument("--n-random-directions", type=int, default=8,
                    help="same-head norm-matched random-direction controls (0 = skip)")
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

    def run_set(names):
        """Orthogonalize the named components with per-edit realized post-bf16
        delta norm measurement; returns (slope_diff, per_family, edits)."""
        comps = [parse_component(n) for n in names]
        snap = snapshot_weights(model, comps)
        try:
            edits = [orthogonalize_component_measured(model, c, direction)
                     for c in comps]
            edited = eval_cells()
        finally:
            restore_weights(snap)
        deltas = {n: edited[n] - base[n] for n in cells}
        sd, per_fam = slope_diff(deltas, meta)
        return sd, per_fam, edits

    base = eval_cells()
    print("baseline uptake:", {n: round(float(v.mean()), 3) for n, v in base.items()})

    target_names = args.suppressors.split(",")
    rng = pyrandom.Random(args.seed)

    # targeted set runs FIRST: its ACTUAL realized norms anchor both the
    # same-head random-direction controls and the null-set mismatch accounting
    tgt_sd, tgt_fam, tgt_edits = run_set(target_names)
    tgt_realized = {e["component"]: e["realized_delta_norm"] for e in tgt_edits}
    tgt_norms = {e["component"]: round(e["theoretical_delta_norm"], 3)
                 for e in tgt_edits}
    tgt_total = sum(e["theoretical_delta_norm"] for e in tgt_edits)
    tgt_total_realized = sum(tgt_realized.values())
    print(f"targeted {target_names}: slope-diff {tgt_sd:+.4f} "
          f"(total delta norm realized {tgt_total_realized:.3f} / "
          f"theoretical {tgt_total:.3f})")

    def norm_fn(name):
        return float(targeted_delta_norm(model, parse_component(name), direction))

    null_names, pools, cand_norms, tgt_by_layer = norm_matched_multiset_sets(
        norm_fn, target_names, args.n_sets, rng, args.pool_size)
    n_combos = int(np.prod([len(p) for p in pools]))
    print(f"{len(null_names)} norm-and-layer-multiset-matched null sets "
          f"(one head per targeted layer, targets excluded; per-layer pool = "
          f"{args.pool_size} nearest-norm heads; {n_combos} distinct combos)")

    rows = []
    for i, names in enumerate(null_names):
        sd, per_fam, edits = run_set(names)
        theo = {e["component"]: e["theoretical_delta_norm"] for e in edits}
        real = {e["component"]: e["realized_delta_norm"] for e in edits}
        total_theo, total_real = sum(theo.values()), sum(real.values())
        rows.append({"set": names, "slope_diff": sd, "per_family": per_fam,
                     "theoretical_delta_norms": {k: round(v, 3) for k, v in theo.items()},
                     "realized_delta_norms": {k: round(v, 4) for k, v in real.items()},
                     "total_theoretical_delta_norm": round(total_theo, 3),
                     "total_realized_delta_norm": round(total_real, 4),
                     "total_norm_rel_mismatch": round(total_theo / tgt_total - 1, 4),
                     "total_norm_rel_mismatch_realized":
                         round(total_real / tgt_total_realized - 1, 4)})
        print(f"  null {i+1}/{len(null_names)} {names}: {sd:+.4f} "
              f"(norm mismatch theo {rows[-1]['total_norm_rel_mismatch']:+.1%} / "
              f"realized {rows[-1]['total_norm_rel_mismatch_realized']:+.1%})")

    nd = np.array([r["slope_diff"] for r in rows])
    mis = np.array([abs(r["total_norm_rel_mismatch"]) for r in rows])
    mis_real = np.array([abs(r["total_norm_rel_mismatch_realized"]) for r in rows])
    total_norm_balance_ok = bool(mis_real.max() <= NULL_TOTAL_NORM_TOL)
    k, p = mc_p(nd, tgt_sd)
    res = {
        "direction": {"path": args.direction, "sha256": sha256(args.direction)},
        "cell_files": {n: {"path": p_, "sha256": sha256(p_)} for n, p_ in CELLS.items()},
        "design": {
            "null": "layer multiset preserved (one head per targeted layer, targets "
                    "excluded); per-layer pool = the pool_size heads whose "
                    "THEORETICAL (pre-cast) rank-1 delta norm under d_resid is "
                    "nearest the targeted head's; sets sampled distinct from the "
                    "pool product; ACTUAL realized norms recorded per set with "
                    "realized+theoretical total-norm mismatch vs the targeted set",
            "primary_inference": f"two-sided empirical Monte Carlo p = (1+k)/(n+1); "
                                 f"n_sets={len(rows)} gives min attainable p = "
                                 f"{1/(len(rows)+1):.3f}; sampled from {n_combos} "
                                 f"matched combos, NOT exhaustive",
            "secondary_inference": "z-score vs null mean/std (descriptive)",
            "direction_control": "same four heads; one shared random unit vector per "
                                 "replicate; per-component rank-1 delta scaled to the "
                                 "targeted ACTUAL REALIZED post-bf16 norm "
                                 "(apply_norm_matched_random); each edit's realized "
                                 "norm gated at 3% rel error vs the targeted realized "
                                 "norm; "
                                 f"min attainable p = {1/(args.n_random_directions+1):.3f}"
                                 if args.n_random_directions else "skipped",
        },
        "pool_size": args.pool_size, "pools": pools,
        "candidate_theoretical_delta_norms": {k_: round(v, 3)
                                              for k_, v in cand_norms.items()},
        "targeted_theoretical_norms_by_layer": tgt_by_layer,
        "targeted_set": target_names, "targeted_slope_diff": tgt_sd,
        "targeted_per_family": tgt_fam,
        "targeted_edits": tgt_edits,
        "targeted_theoretical_delta_norms": tgt_norms,
        "targeted_realized_delta_norms": {k_: round(v, 4)
                                          for k_, v in tgt_realized.items()},
        "targeted_total_theoretical_delta_norm": round(tgt_total, 3),
        "targeted_total_realized_delta_norm": round(tgt_total_realized, 4),
        "null_sets": rows, "null_mean": float(nd.mean()),
        "null_std": float(nd.std(ddof=1)),
        "null_range": [float(nd.min()), float(nd.max())],
        "norm_match_quality": {
            "theoretical": {"mean_abs_rel_mismatch": float(mis.mean()),
                            "max_abs_rel_mismatch": float(mis.max())},
            "realized": {"mean_abs_rel_mismatch": float(mis_real.mean()),
                         "max_abs_rel_mismatch": float(mis_real.max())},
            "predeclared_total_norm_tol": NULL_TOTAL_NORM_TOL,
            "total_norm_balance_within_tolerance": total_norm_balance_ok,
            "exact_norm_matching": False,
            "claim": ("all null sets are approximately TOTAL-norm balanced within "
                      "the predeclared realized-norm tolerance "
                      f"({NULL_TOTAL_NORM_TOL:.0%}); this is not exact or per-component matching"
                      if total_norm_balance_ok else
                      "null sets are norm-SELECTED (nearest-norm pools) "
                      "but exceed the predeclared total realized-norm tolerance "
                      f"({NULL_TOTAL_NORM_TOL:.0%}); retain edit norm as a "
                      "covariate / stratify by norm in any downstream inference"),
        },
        "z_secondary": float((tgt_sd - nd.mean()) / nd.std(ddof=1)),
        "n_null_as_extreme": k,
        "mc_p": p,
    }
    print(f"targeted {tgt_sd:+.4f} | null {res['null_mean']:+.4f}±{res['null_std']:.4f} "
          f"| {k}/{len(rows)} as extreme | MC p={p:.3f} (min {1/(len(rows)+1):.3f}) "
          f"| norm mismatch (realized) mean {mis_real.mean():.1%} max {mis_real.max():.1%} "
          f"| approximate total-norm balance: {total_norm_balance_ok} "
          f"(tol {NULL_TOTAL_NORM_TOL:.0%})")
    if not total_norm_balance_ok:
        print("WARNING: null-set total realized norms exceed the predeclared "
              f"{NULL_TOTAL_NORM_TOL:.0%} tolerance — do NOT describe the null as "
              "exactly norm-matched; norm-covariate/stratified caveat applies")

    if args.n_random_directions:
        np_rng = np.random.default_rng(args.seed)
        comps = [parse_component(n) for n in target_names]
        drows = []
        for i in range(args.n_random_directions):
            rd = torch.tensor(np_rng.standard_normal(direction.shape[0]),
                              dtype=torch.float32)
            rd /= torch.linalg.vector_norm(rd)
            snap = snapshot_weights(model, comps)
            try:
                # requested norm = the targeted edit's ACTUAL realized norm,
                # so the 3% gate inside the helper is realized-vs-realized
                edits = [apply_norm_matched_random(
                    model, c, direction, tgt_realized[c["name"]], None, rand_vec=rd)
                    for c in comps]
                edited = eval_cells()
            finally:
                restore_weights(snap)
            deltas = {n: edited[n] - base[n] for n in cells}
            sd, per_fam = slope_diff(deltas, meta)
            drows.append({"slope_diff": sd, "per_family": per_fam, "edits": edits})
            print(f"  same-head norm-matched random-dir "
                  f"{i+1}/{args.n_random_directions}: {sd:+.4f}")
        dd = np.array([r["slope_diff"] for r in drows])
        dk, dp = mc_p(dd, tgt_sd)
        # realized post-bf16 delta norms come back from apply_norm_matched_random
        # (per-edit 3% assert inside the helper); persist + re-gate the max here
        norm_errs = [abs(e["relative_norm_error"])
                     for r in drows for e in r["edits"]]
        assert max(norm_errs) <= 0.03, (
            f"same-head random-direction realized norm error "
            f"{max(norm_errs):.4f} exceeds 3% gate")
        res["same_head_random_directions"] = {
            "rows": drows, "mean": float(dd.mean()), "std": float(dd.std(ddof=1)),
            "n_as_extreme": dk, "mc_p": dp,
            "min_attainable_p": 1 / (len(dd) + 1),
            "requested_norms": "targeted ACTUAL realized post-bf16 delta norms",
            "max_abs_relative_norm_error": float(max(norm_errs)),
            "realized_norm_gate": 0.03,
        }
        print(f"same-head norm-matched random-dir null: "
              f"{dd.mean():+.4f}±{dd.std(ddof=1):.4f} | {dk}/{len(dd)} as extreme "
              f"| MC p={dp:.3f} (min {1/(len(dd)+1):.3f})")

    (out / "slope_nulls.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {out}/slope_nulls.json")


if __name__ == "__main__":
    main()
