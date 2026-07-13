"""LB3 (= roadmap C1): SVD alignment of the direction with component weights.

For each targeted component (and every same-type component as a distribution):
truncated SVD of its residual-writing weight matrix (MLP down_proj, or the
head's o_proj column block), transformed by the layer's sandwich-norm gain
diag(1 + w), and measure how much of d_resid's energy lies in the top-k
left-singular subspace: E_k = || U_k^T d ||^2. A direction the component
"natively" writes concentrates energy in few singular vectors; an off-axis
direction doesn't. Also relates to the E26b edit-norm confound (targeted pair
had the largest rank-1 delta norms).

Scalability: full float64 SVD of 42 matrices of shape 3584x14336 is not
acceptable; we use a seeded randomized truncated SVD (Halko range finder with
power iterations + oversampling, max 2 power iterations by default) in
float32, top-64 singular vectors only, processing one layer at a time.
`total_in_colspace` was REMOVED: a truncated SVD cannot establish total
column-space energy. A numerical self-test against exact SVD on a toy matrix
runs before the sweep.

No forward passes. --device auto uses CUDA when available: the model stays on
CPU and ONE weight matrix (plus the direction/gain vectors) is moved to the
GPU at a time, then freed; CPU is the fallback and remains exact. The device
used is recorded in the output JSON. This step is EXPLORATORY — chain scripts
must not let it block later steps.

Usage:
  python -u src/analysis/svd_alignment.py \
    --direction results/controlled_directions_gemma2_9b_it/direction_M_resid_block20.npy \
    --device auto --out results/lb3_svd_alignment_gemma
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

WRITERS = ["L19MLP", "L20MLP"]
SUPPRESSORS = ["L18H13", "L20H10", "L19H12", "L17H7"]
TOPK = [1, 4, 16, 64]
RANK = 64
OVERSAMPLE = 16
POWER_ITERS = 2  # max 2 by default; enough with QR stabilization + oversampling
SVD_SEED = 0


@torch.no_grad()
def randomized_topk_left_singular(W, k=RANK, oversample=OVERSAMPLE,
                                  seed=SVD_SEED, niter=POWER_ITERS):
    """Seeded randomized truncated SVD (Halko et al. 2011): top-k left singular
    vectors of W in float32, on W's device. QR-stabilized power iterations
    sharpen the subspace when the spectrum decays slowly; the seeded range
    finder is drawn on CPU so results are device-independent."""
    W = W.float()
    m, n = W.shape
    q = min(k + oversample, m, n)
    g = torch.Generator().manual_seed(seed)
    omega = torch.randn(n, q, generator=g, dtype=torch.float32).to(W.device)
    Q, _ = torch.linalg.qr(W @ omega)
    for _ in range(niter):
        Z, _ = torch.linalg.qr(W.T @ Q)
        Q, _ = torch.linalg.qr(W @ Z)
    B = Q.T @ W  # (q, n)
    Ub, S, _ = torch.linalg.svd(B, full_matrices=False)
    U = Q @ Ub
    k = min(k, q)
    return U[:, :k], S[:k]


def energy_curve(W, gain, d, device="cpu"):
    """E_k = ||U_k^T d||^2 for k in TOPK, from the truncated top-RANK left
    singular basis of the gained matrix diag(gain) @ W.

    On CUDA, exactly one matrix (plus gain/direction) lives on the GPU per
    call and is freed before returning; CPU is the fallback.
    best_single_cos is the max |cos| over the computed top-RANK vectors only.
    Total column-space energy is NOT reported — truncated SVD can't see it.
    """
    Wg = (gain.to(device)[:, None] * W.to(device))
    U, _ = randomized_topk_left_singular(Wg)
    proj = (U.T @ d.to(device)).cpu().numpy()  # (<=RANK,)
    del Wg, U
    if device != "cpu":
        torch.cuda.empty_cache()
    cum = np.cumsum(proj ** 2)
    return {f"top{k}": float(cum[min(k, len(cum)) - 1]) for k in TOPK} | {
        "best_single_cos": float(np.max(np.abs(proj))),
    }


def self_test():
    """Truncated randomized SVD must match exact SVD energy on a toy matrix."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((200, 120)).astype(np.float32)
    U0, S0, Vt0 = np.linalg.svd(A, full_matrices=False)
    S0 = S0 * np.exp(-np.arange(len(S0)) / 15.0)  # decaying spectrum
    A = (U0 * S0) @ Vt0
    d = rng.standard_normal(200).astype(np.float32)
    d /= np.linalg.norm(d)
    k = 16
    exact_U = np.linalg.svd(A, full_matrices=False)[0][:, :k]
    exact = np.cumsum((exact_U.T @ d) ** 2)
    approx_U, _ = randomized_topk_left_singular(torch.from_numpy(A), k=k,
                                                oversample=8, seed=0)
    approx = np.cumsum((approx_U.numpy().T @ d) ** 2)
    err = float(np.max(np.abs(exact - approx)))
    assert err <= 1e-3, f"randomized SVD self-test failed: max energy err {err:.2e}"
    print(f"self-test OK: max |E_k exact - E_k randomized| = {err:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--direction", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="auto = CUDA if available (one matrix on GPU at a "
                         "time, freed after), else CPU fallback")
    ap.add_argument("--out", default="results/lb3_svd_alignment_gemma")
    args = ap.parse_args()

    self_test()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            print("--device cuda requested but CUDA unavailable — CPU fallback")
            device = "cpu"
    print(f"SVD device: {device}")

    from transformers import AutoModelForCausalLM
    # bf16 load (checkpoint-native, exact); each matrix is cast to float32
    # one layer at a time for the truncated SVD
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.eval()
    cfg = model.config
    n_layers, n_heads = cfg.num_hidden_layers, cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads

    d = torch.tensor(np.load(args.direction), dtype=torch.float32)
    d /= torch.linalg.vector_norm(d)
    rng = np.random.default_rng(0)
    d_rand = torch.tensor(rng.standard_normal(len(d)), dtype=torch.float32)
    d_rand /= torch.linalg.vector_norm(d_rand)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"direction": args.direction,
              "svd": {"method": "randomized truncated (Halko), float32",
                      "rank": RANK, "oversample": OVERSAMPLE,
                      "power_iters": POWER_ITERS, "seed": SVD_SEED,
                      "device": device,
                      "note": "total_in_colspace removed — not estimable "
                              "from a truncated SVD"},
              "mlp": {}, "heads": {}, "random_direction_reference": {}}

    for l in range(n_layers):
        layer = model.model.layers[l]
        gain_m = 1.0 + layer.post_feedforward_layernorm.weight.detach().float()
        W = layer.mlp.down_proj.weight.detach().float()  # (d_model, d_ff), CPU
        report["mlp"][f"L{l}MLP"] = energy_curve(W, gain_m, d, device)
        if f"L{l}MLP" in WRITERS:
            report["random_direction_reference"][f"L{l}MLP"] = energy_curve(
                W, gain_m, d_rand, device)
        del W
        gain_a = 1.0 + layer.post_attention_layernorm.weight.detach().float()
        Wo = layer.self_attn.o_proj.weight.detach()  # (d_model, H*hd), bf16, CPU
        for name in [c for c in SUPPRESSORS if c.startswith(f"L{l}H")]:
            h = int(name.split("H")[1])
            Wh = Wo[:, h * head_dim:(h + 1) * head_dim].float()
            report["heads"][name] = energy_curve(Wh, gain_a, d, device)
            report["random_direction_reference"][name] = energy_curve(
                Wh, gain_a, d_rand, device)
        del Wo
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
        print(f"{c}: top4 {e['top4']:.4f} top64 {e['top64']:.4f} "
              f"(rand ref top64 {report['random_direction_reference'][c]['top64']:.4f})")

    (out / "svd_alignment.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out}/svd_alignment.json")


if __name__ == "__main__":
    main()
