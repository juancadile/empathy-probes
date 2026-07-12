"""E26b: MATCHED random-component nulls (rescue step 5).

The earlier 12-set null (E14d-b) mixed ks, types, and layers. Here nulls are
sampled to match each targeted set's structure exactly:
  writer-matched     : k=2 = 1 MLP + 1 head, layers drawn from the targeted
                       writer layers' range, excluding targeted components
  suppressor-matched : k=4 = 4 heads, layers from the targeted suppressor range
Realized rank-1 delta Frobenius norms are recorded per component so norm
mismatch is measurable rather than assumed.

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


def layer_range(names):
    ls = [parse_component(n)["layer"] for n in names.split(",")]
    return min(ls), max(ls)


def sample_matched(rng, k_mlp, k_head, lo, hi, exclude):
    comps = set()
    while sum(1 for c in comps if c.endswith("MLP")) < k_mlp:
        c = f"L{rng.randint(lo, hi)}MLP"
        if c not in exclude:
            comps.add(c)
    while len(comps) < k_mlp + k_head:
        c = f"L{rng.randint(lo, hi)}H{rng.randint(0, N_HEADS - 1)}"
        if c not in exclude and not c.endswith("MLP"):
            comps.add(c)
    return sorted(comps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--direction", required=True)
    ap.add_argument("--writers", required=True)
    ap.add_argument("--suppressors", required=True)
    ap.add_argument("--n-sets", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/e26_matched_nulls_gemma")
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
    results = {"baseline": base, "direction": args.direction, "families": {}}

    specs = {
        "writer_matched": dict(names=args.writers, k_mlp=1, k_head=1),
        "suppressor_matched": dict(names=args.suppressors, k_mlp=0, k_head=4),
    }
    rng = pyrandom.Random(args.seed)
    for fam, spec in specs.items():
        target_names = spec["names"].split(",")
        lo, hi = layer_range(spec["names"])
        tgt_m, tgt_norms = run_set(target_names)
        rows = []
        for i in range(args.n_sets):
            names = sample_matched(rng, spec["k_mlp"], spec["k_head"], lo, hi,
                                   set(target_names))
            m, norms = run_set(names)
            rows.append({"set": names, "delta": m - base,
                         "delta_norms": {k: round(v, 3) for k, v in norms.items()}})
            print(f"  {fam} {i+1}/{args.n_sets} {names}: {m - base:+.4f}")
        deltas = np.array([r["delta"] for r in rows])
        results["families"][fam] = {
            "targeted_set": target_names, "targeted_delta": tgt_m - base,
            "targeted_delta_norms": {k: round(v, 3) for k, v in tgt_norms.items()},
            "null_sets": rows, "null_mean": float(deltas.mean()),
            "null_std": float(deltas.std(ddof=1)),
            "null_range": [float(deltas.min()), float(deltas.max())],
            "z": float((tgt_m - base - deltas.mean()) / deltas.std(ddof=1)),
            "n_null_more_extreme": int((np.abs(deltas) >= abs(tgt_m - base)).sum()),
        }
        f = results["families"][fam]
        print(f"{fam}: targeted {f['targeted_delta']:+.4f} | null "
              f"{f['null_mean']:+.4f}±{f['null_std']:.4f} | z={f['z']:+.1f} | "
              f"{f['n_null_more_extreme']}/{args.n_sets} as extreme")

    (out / "matched_nulls.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {out}/matched_nulls.json")


if __name__ == "__main__":
    main()
