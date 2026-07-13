"""LB2: per-layer logit-lens trajectory of the A/B decision.

Decode logit(A) - logit(B) from final_norm(h_l) @ W_U at the choice position
for every layer l (classic logit lens). Questions: where does the decision
crystallize, and does it match the L16-20 band localized by the matched nulls
(E26b)? Run for baseline and (optionally) edited conditions.

Usage (Spark, `empathy` env):
  python -u src/analysis/logit_lens_trajectory.py \
    --direction results/controlled_directions_gemma2_9b_it/direction_M_resid_block20.npy \
    --writers "L20MLP,L19MLP" --suppressors "L18H13,L20H10,L19H12,L17H7" \
    --out results/lb2_logit_lens_gemma
"""

import argparse
import json
import random as pyrandom
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from weight_orthogonalization import (  # noqa: E402
    load_pairs, orthogonalize_component, parse_component,
    restore_weights, snapshot_weights,
)
from activation_patching import build_choice_prompt  # noqa: E402

SETS = {
    "M_confirm": "data/contrastive_pairs/v2_1/M_confirm_templated.jsonl",
    "T_confirm": "data/contrastive_pairs/v2_1/T_confirm_templated.jsonl",
}


@torch.no_grad()
def trajectory(model, tok, pairs, device, batch_size, max_tokens, seed):
    """returns (n_pairs, n_layers+1) logit-diff-by-layer, flip-corrected."""
    ta = tok.encode("A", add_special_tokens=False)[0]
    tb = tok.encode("B", add_special_tokens=False)[0]
    w_u = model.get_output_embeddings().weight.float()
    du = (w_u[ta] - w_u[tb]).to(device)
    norm = model.model.norm

    rng = pyrandom.Random(seed)
    flips = [rng.random() < 0.5 for _ in pairs]
    prompts = [build_choice_prompt(p, f) for p, f in zip(pairs, flips)]
    signs = np.array([-1.0 if f else 1.0 for f in flips])

    rows = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_tokens).to(device)
        lens = enc["attention_mask"].sum(1)
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        for r_i, L in enumerate(lens.tolist()):
            # hs[-1] is ALREADY post-final-RMSNorm in HF Gemma-2/Llama; apply
            # the lens norm only to hs[:-1] and decode hs[-1] directly (the
            # final point is then the model's true PRE-softcap logit diff).
            states = torch.stack([h[r_i, L - 1].float() for h in hs[:-1]])
            ld = (norm(states) @ du).cpu().numpy()
            ld_final = float(hs[-1][r_i, L - 1].float() @ du)
            rows.append(np.concatenate([ld, [ld_final]]))
    return np.array(rows) * signs[:, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--direction", default=None,
                    help="if set with component args, also run edited conditions")
    ap.add_argument("--writers", default=None)
    ap.add_argument("--suppressors", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/lb2_logit_lens_gemma")
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

    conds = {"baseline": None}
    if args.direction and args.writers:
        conds["writers_edited"] = args.writers
    if args.direction and args.suppressors:
        conds["suppressors_edited"] = args.suppressors
    direction = None
    if args.direction:
        direction = torch.tensor(np.load(args.direction), dtype=torch.float32, device=device)
        direction /= torch.linalg.vector_norm(direction)

    report = {"model": args.model,
              "index_semantics": "index 0 = embeddings; index l (1..n_layers-1) "
                                 "= residual after block l-1, decoded through the "
                                 "final RMSNorm (logit lens); last index = "
                                 "post-final-norm state decoded directly = true "
                                 "pre-softcap logit diff (Gemma-2 softcap is "
                                 "monotone and not applied here)",
              "conditions": {}}
    for cond, spec in conds.items():
        snap = None
        if spec:
            comps = [parse_component(v) for v in spec.split(",")]
            snap = snapshot_weights(model, comps)
            for c in comps:
                orthogonalize_component(model, c, direction)
        try:
            entry = {}
            for name, path in SETS.items():
                pairs = load_pairs(path)
                traj = trajectory(model, tok, pairs, device, args.batch_size,
                                  args.max_tokens, args.seed)
                mean_traj = traj.mean(0)
                # crystallization: first hidden index where the mean trajectory
                # reaches half its final value and stays above it
                final = mean_traj[-1]
                half = None
                for l in range(len(mean_traj)):
                    if np.sign(mean_traj[l]) == np.sign(final) and \
                       abs(mean_traj[l]) >= abs(final) / 2 and \
                       all(abs(mean_traj[m]) >= abs(final) / 2 for m in range(l, len(mean_traj))):
                        half = l
                        break
                entry[name] = {"mean_by_layer": [round(float(v), 4) for v in mean_traj],
                               "final": round(float(final), 4),
                               "half_crystallization_hidden_index": half}
                print(f"{cond}/{name}: final {final:+.3f}, half-crystallization at "
                      f"hidden {half} (block {None if half is None else half - 1})")
            report["conditions"][cond] = entry
        finally:
            if snap:
                restore_weights(snap)

    (out / "logit_lens.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out}/logit_lens.json")


if __name__ == "__main__":
    main()
