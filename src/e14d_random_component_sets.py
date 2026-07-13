"""E14d-b: random-COMPONENT-set null distribution (stress-test fix 4).

One deterministic random component set is a comparator, not a null. This
samples N random k=6 component sets (layers 9-21, heads 0-15 or MLP, excluding
the targeted six), removes the target direction from each, and evaluates the
confirmatory-M choice delta — giving a distribution against which the writer
(-0.184) and suppressor (+0.150) effects and the historical random-set leak
(-0.057) can be judged.

Usage (Spark, `empathy` env):
  python -u src/e14d_random_component_sets.py \
    --direction results/controlled_directions_gemma2_9b_it/direction_M_block20.npy \
    --n-sets 12 --out results/e14d_gemma2_9b_it
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
except ModuleNotFoundError:
    from weight_orthogonalization import (
        choice_scores, load_pairs, orthogonalize_component, parse_component,
        restore_weights, snapshot_weights,
    )

TARGETED = {"L19MLP", "L20H15", "L19H12", "L15H15", "L17H13", "L18H13"}


def sample_set(rng):
    comps = set()
    while len(comps) < 6:
        layer = rng.randint(9, 21)
        head = rng.randint(0, 16)  # 16 => MLP
        name = f"L{layer}MLP" if head == 16 else f"L{layer}H{head}"
        if name not in TARGETED:
            comps.add(name)
    return sorted(comps)


#: Integrity Repair A QA Q4 (2026-07-13): this CLI performs direct weight
#: edits but has NOT been migrated to the shared accepted/exploratory
#: evidence-run contract. Every artifact it emits carries the permanent
#: classification below; there is deliberately NO accepted mode here.
EVIDENCE_ELIGIBILITY = "historical_or_exploratory_only"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--direction", required=True)
    ap.add_argument("--n-sets", type=int, default=12)
    ap.add_argument("--m-pairs", default="data/contrastive_pairs/v2_1/M_confirm_templated.jsonl")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/e14d_gemma2_9b_it")
    args = ap.parse_args()
    print(f"evidence eligibility: {EVIDENCE_ELIGIBILITY} — no accepted mode; artifacts cannot support confirmatory claims", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    if (out / "random_component_sets.json").exists():
        raise SystemExit(f"refusing to overwrite historical artifact in {out}; choose a fresh --out")
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager").to(device)
    model.eval()
    direction = torch.tensor(np.load(args.direction), dtype=torch.float32, device=device)
    direction /= torch.linalg.vector_norm(direction)
    pairs = load_pairs(args.m_pairs)

    base = np.mean(choice_scores(model, tok, pairs, args.seed, args.batch_size,
                                 args.max_tokens, device))
    print(f"baseline M_confirm {base:+.4f}")
    rng = pyrandom.Random(args.seed)
    rows = []
    for i in range(args.n_sets):
        names = sample_set(rng)
        comps = [parse_component(n) for n in names]
        snap = snapshot_weights(model, comps)
        try:
            for c in comps:
                orthogonalize_component(model, c, direction)
            m = np.mean(choice_scores(model, tok, pairs, args.seed, args.batch_size,
                                      args.max_tokens, device))
        finally:
            restore_weights(snap)
        rows.append({"set": names, "delta": float(m - base)})
        print(f"set {i+1}/{args.n_sets} {names}: delta {m - base:+.4f}")

    deltas = np.array([r["delta"] for r in rows])
    summary = {"baseline": float(base), "sets": rows,
               "evidence_eligibility": EVIDENCE_ELIGIBILITY,
               "delta_mean": float(deltas.mean()), "delta_std": float(deltas.std(ddof=1)),
               "delta_min": float(deltas.min()), "delta_max": float(deltas.max()),
               "writer_ref": -0.1836, "suppressor_ref": 0.1497,
               "writer_z": float((-0.1836 - deltas.mean()) / deltas.std(ddof=1)),
               "suppressor_z": float((0.1497 - deltas.mean()) / deltas.std(ddof=1))}
    (out / "random_component_sets.json").write_text(json.dumps(summary, indent=2))
    print(f"null: {deltas.mean():+.4f} ± {deltas.std(ddof=1):.4f} "
          f"[{deltas.min():+.4f}, {deltas.max():+.4f}] | "
          f"writer z {summary['writer_z']:+.1f} suppressor z {summary['suppressor_z']:+.1f}")


if __name__ == "__main__":
    main()
