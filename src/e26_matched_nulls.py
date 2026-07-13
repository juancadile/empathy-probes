"""E26b: MATCHED random-component nulls (rescue step 5).

The earlier 12-set null (E14d-b) mixed ks, types, and layers. Here each null
family matches its targeted set's composition EXACTLY (k and MLP/head type
counts are derived from the targeted set, not hardcoded), with layers drawn
from the targeted layers' range widened by +/-3 (the exact range can leave
too few same-type candidates once targeted components are excluded).
Null sets are distinct; when the whole combination space is <= --n-sets the
space is enumerated instead, making n_more_extreme an exact permutation count.
Realized rank-1 delta Frobenius norms are recorded per component so norm
mismatch is measurable rather than assumed. With ~12 draws the primary
statistic is n_more_extreme; z is reported as a coarse secondary.

Usage (Spark, `empathy` env):
  python -u src/e26_matched_nulls.py \
    --direction results/controlled_directions_gemma2_9b_it/direction_M_resid_block20.npy \
    --writers "L20MLP,L19MLP" --suppressors "L18H13,L20H10,L19H12,L17H7" \
    --n-sets 12 --out results/e26_matched_nulls_gemma
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

M_CONFIRM = "data/contrastive_pairs/v2_1/M_confirm_templated.jsonl"
N_HEADS = 16  # gemma-2-9b
N_LAYERS = 42
BAND = 3  # widen the targeted layer range by this much on each side


def matched_spec(names):
    """k_mlp/k_head derived from the targeted set itself, never hardcoded."""
    ns = names.split(",")
    k_mlp = sum(1 for n in ns if n.endswith("MLP"))
    ls = [parse_component(n)["layer"] for n in ns]
    lo = max(0, min(ls) - BAND)
    hi = min(N_LAYERS - 1, max(ls) + BAND)
    return dict(names=names, k_mlp=k_mlp, k_head=len(ns) - k_mlp, lo=lo, hi=hi)


def candidate_sets(rng, k_mlp, k_head, lo, hi, exclude, n_sets):
    """Distinct type-matched sets; enumerate the space if it is small."""
    from itertools import combinations
    from math import comb
    mlps = [f"L{l}MLP" for l in range(lo, hi + 1) if f"L{l}MLP" not in exclude]
    heads = [f"L{l}H{h}" for l in range(lo, hi + 1) for h in range(N_HEADS)
             if f"L{l}H{h}" not in exclude]
    total = comb(len(mlps), k_mlp) * comb(len(heads), k_head)
    if total <= max(n_sets, 20):
        return [sorted(m + h) for m in combinations(mlps, k_mlp)
                for h in combinations(heads, k_head)], True
    seen, sets = set(), []
    while len(sets) < n_sets:
        s = tuple(sorted(rng.sample(mlps, k_mlp) + rng.sample(heads, k_head)))
        if s not in seen:
            seen.add(s)
            sets.append(list(s))
    return sets, False


#: Integrity Repair A QA Q4 (2026-07-13): this CLI performs direct weight
#: edits but has NOT been migrated to the shared accepted/exploratory
#: evidence-run contract. Every artifact it emits carries the permanent
#: classification below; there is deliberately NO accepted mode here.
EVIDENCE_ELIGIBILITY = "historical_or_exploratory_only"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--direction", required=True)
    ap.add_argument("--writers", required=True)
    ap.add_argument("--suppressors", required=True)
    ap.add_argument("--n-sets", type=int, default=12)
    ap.add_argument("--include-overlap", action="store_true",
                    help="when a family's disjoint space is exhaustive, also "
                         "evaluate sets sharing components with the targeted set "
                         "for a FULL-universe rank test (the disjoint 15 alone "
                         "support only a disjoint-control comparison)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/e26_matched_nulls_gemma")
    args = ap.parse_args()
    print(f"evidence eligibility: {EVIDENCE_ELIGIBILITY} — no accepted mode; artifacts cannot support confirmatory claims", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    if (out / "matched_nulls.json").exists():
        raise SystemExit(f"refusing to overwrite historical artifact in {out}; choose a fresh --out")
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager").to(device)
    model.eval()
    direction = torch.tensor(np.load(args.direction), dtype=torch.float32, device=device)
    direction /= torch.linalg.vector_norm(direction)
    pairs = load_pairs(M_CONFIRM)

    def eval_now():
        return float(np.mean(choice_scores(model, tok, pairs, args.seed,
                                           args.batch_size, args.max_tokens, device)))

    def run_set(names):
        comps = [parse_component(n) for n in names]
        norms = {c["name"]: targeted_delta_norm(model, c, direction) for c in comps}
        snap = snapshot_weights(model, comps)
        try:
            for c in comps:
                orthogonalize_component(model, c, direction)
            m = eval_now()
        finally:
            restore_weights(snap)
        return m, norms

    base = eval_now()
    print(f"baseline {base:+.4f}")
    results = {"baseline": base, "direction": args.direction,
               "evidence_eligibility": EVIDENCE_ELIGIBILITY, "families": {}}

    specs = {
        "writer_matched": matched_spec(args.writers),
        "suppressor_matched": matched_spec(args.suppressors),
    }
    rng = pyrandom.Random(args.seed)
    for fam, spec in specs.items():
        target_names = spec["names"].split(",")
        null_names, exhaustive = candidate_sets(
            rng, spec["k_mlp"], spec["k_head"], spec["lo"], spec["hi"],
            set(target_names), args.n_sets)
        print(f"{fam}: k_mlp={spec['k_mlp']} k_head={spec['k_head']} "
              f"layers {spec['lo']}-{spec['hi']} | {len(null_names)} null sets"
              f"{' (exhaustive)' if exhaustive else ''}")
        tgt_m, tgt_norms = run_set(target_names)
        rows = []
        for i, names in enumerate(null_names):
            m, norms = run_set(names)
            rows.append({"set": names, "delta": m - base,
                         "delta_norms": {k: round(v, 3) for k, v in norms.items()}})
            print(f"  {fam} {i+1}/{len(null_names)} {names}: {m - base:+.4f}")
        deltas = np.array([r["delta"] for r in rows])
        results["families"][fam] = {
            "targeted_set": target_names, "targeted_delta": tgt_m - base,
            "targeted_delta_norms": {k: round(v, 3) for k, v in tgt_norms.items()},
            "spec": {k: spec[k] for k in ("k_mlp", "k_head", "lo", "hi")},
            "exhaustive": exhaustive,
            "null_sets": rows, "null_mean": float(deltas.mean()),
            "null_std": float(deltas.std(ddof=1)),
            "null_range": [float(deltas.min()), float(deltas.max())],
            "z": float((tgt_m - base - deltas.mean()) / deltas.std(ddof=1)),
            "n_null_more_extreme": int((np.abs(deltas) >= abs(tgt_m - base)).sum()),
        }
        f = results["families"][fam]
        print(f"{fam}: targeted {f['targeted_delta']:+.4f} | null "
              f"{f['null_mean']:+.4f}±{f['null_std']:.4f} | z={f['z']:+.1f} | "
              f"{f['n_null_more_extreme']}/{len(null_names)} as extreme")

        if args.include_overlap and exhaustive:
            from itertools import combinations
            pool_m = [f"L{l}MLP" for l in range(spec["lo"], spec["hi"] + 1)]
            pool_h = [f"L{l}H{h}" for l in range(spec["lo"], spec["hi"] + 1)
                      for h in range(N_HEADS)]
            allsets = [sorted(m + h) for m in combinations(pool_m, spec["k_mlp"])
                       for h in combinations(pool_h, spec["k_head"])]
            tgt_sorted = sorted(target_names)
            overlap = [s for s in allsets
                       if s != tgt_sorted and any(c in target_names for c in s)]
            orows = []
            for i, names in enumerate(overlap):
                m, norms = run_set(names)
                orows.append({"set": names, "delta": m - base,
                              "delta_norms": {k: round(v, 3) for k, v in norms.items()}})
                print(f"  {fam} overlap {i+1}/{len(overlap)} {names}: {m - base:+.4f}")
            full = np.array([r["delta"] for r in rows + orows])
            n_ext = int((np.abs(full) >= abs(tgt_m - base)).sum())
            f["overlap_sets"] = orows
            f["full_universe_rank"] = {
                "n_universe": len(full) + 1,  # incl. targeted
                "n_as_extreme": n_ext,
                "exact_p": (1 + n_ext) / (len(full) + 1),
            }
            print(f"{fam} FULL-universe rank: {n_ext}/{len(full)} as extreme -> "
                  f"exact p = {(1 + n_ext) / (len(full) + 1):.4f}")

    (out / "matched_nulls.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {out}/matched_nulls.json")


if __name__ == "__main__":
    main()
