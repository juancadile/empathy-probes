"""LB3 (= roadmap C1): SVD alignment of the direction with component weights.

For each targeted component (and every same-type component as a distribution):
SVD its residual-writing weight matrix (MLP down_proj, or the head's o_proj
column block), transform by the layer's sandwich-norm gain diag(1 + w), and
measure how much of d_resid's energy lies in the top-k left-singular
subspace: E_k = || U_k^T d ||^2. A direction the component "natively" writes
concentrates energy in few singular vectors; an off-axis direction doesn't.
Also relates to the E26b edit-norm confound (targeted pair had the largest
rank-1 delta norms).

CPU-only on weights; no forward passes.

Usage:
  python -u src/analysis/svd_alignment.py \
    --direction results/controlled_directions_gemma2_9b_it/direction_M_resid_block20.npy \
    --out results/lb3_svd_alignment_gemma
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

WRITERS = ["L19MLP", "L20MLP"]
SUPPRESSORS = ["L18H13", "L20H10", "L19H12", "L17H7"]
TOPK = [1, 4, 16, 64]


def energy_curve(W, gain, d):
    """E_k = ||U_k^T d_tilde||^2 with d_tilde = d / ||d|| in the gained basis.

    The residual write is diag(gain) @ W @ x, so its column space is
    diag(gain) @ colspace(W): SVD the gained matrix directly.
    """
    Wg = gain[:, None] * W
    U, S, _ = np.linalg.svd(Wg, full_matrices=False)
    proj = U.T @ d  # (r,)
    cum = np.cumsum(proj ** 2)
    return {f"top{k}": float(cum[min(k, len(cum)) - 1]) for k in TOPK} | {
        "best_single_cos": float(np.max(np.abs(proj))),
        "total_in_colspace": float(cum[-1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--direction", required=True)
    ap.add_argument("--out", default="results/lb3_svd_alignment_gemma")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()
    cfg = model.config
    n_layers, n_heads = cfg.num_hidden_layers, cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads

    d = np.load(args.direction).astype(np.float64)
    d /= np.linalg.norm(d)
    rng = np.random.default_rng(0)
    d_rand = rng.standard_normal(len(d))
    d_rand /= np.linalg.norm(d_rand)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"direction": args.direction, "mlp": {}, "heads": {}, "random_direction_reference": {}}

    for l in range(n_layers):
        layer = model.model.layers[l]
        gain_m = 1.0 + layer.post_feedforward_layernorm.weight.detach().numpy().astype(np.float64)
        W = layer.mlp.down_proj.weight.detach().numpy().astype(np.float64)  # (d_model, d_ff)
        report["mlp"][f"L{l}MLP"] = energy_curve(W, gain_m, d)
        if f"L{l}MLP" in WRITERS:
            report["random_direction_reference"][f"L{l}MLP"] = energy_curve(W, gain_m, d_rand)
        gain_a = 1.0 + layer.post_attention_layernorm.weight.detach().numpy().astype(np.float64)
        Wo = layer.self_attn.o_proj.weight.detach().numpy().astype(np.float64)  # (d_model, H*hd)
        for name in [c for c in SUPPRESSORS if c.startswith(f"L{l}H")]:
            h = int(name.split("H")[1])
            Wh = Wo[:, h * head_dim:(h + 1) * head_dim]
            report["heads"][name] = energy_curve(Wh, gain_a, d)
            report["random_direction_reference"][name] = energy_curve(Wh, gain_a, d_rand)
        print(f"L{l} done", flush=True)

    # summary: targeted vs all-MLP distribution
    for c in WRITERS:
        e = report["mlp"][c]
        all16 = sorted((v["top16"] for v in report["mlp"].values()), reverse=True)
        rank = all16.index(e["top16"]) + 1
        print(f"{c}: top16 energy {e['top16']:.4f} (rank {rank}/{n_layers} among MLPs), "
              f"best single cos {e['best_single_cos']:.4f}, "
              f"rand-dir ref top16 {report['random_direction_reference'][c]['top16']:.4f}")
    for c in SUPPRESSORS:
        e = report["heads"][c]
        print(f"{c}: top4 {e['top4']:.4f} colspace {e['total_in_colspace']:.4f} "
              f"(rand ref {report['random_direction_reference'][c]['total_in_colspace']:.4f})")

    (out / "svd_alignment.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out}/svd_alignment.json")


if __name__ == "__main__":
    main()
