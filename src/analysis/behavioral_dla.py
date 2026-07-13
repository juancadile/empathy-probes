"""LB1: TRUE direct logit attribution onto the behavioral readout.

DFA (E01/E09/E25) attributed the DIRECTION; the direction turned out not to be
welfare-pure held-out (E25b). This attributes the BEHAVIOR: decompose
logit(A) - logit(B) at the choice position into per-component direct-path
contributions, using the residual stream's additivity:

  resid_final = embed + sum_l (attn_add_l + mlp_add_l)
  logit_diff  = du @ final_norm(resid_final)
             ~= r * sum_c (du_scaled @ contrib_c)        [r frozen per example]

where du = W_U[:,A] - W_U[:,B], du_scaled = (1 + w_final) * du, and
r = rsqrt(mean(resid_final^2) + eps) at the readout position. Exactness of the
decomposition is checked against the model's actual logit difference.

Question: do the re-derived writers/suppressors top the BEHAVIORAL ranking,
not just the direction ranking? Reported for M_confirm and T_confirm.

Usage (Spark, `empathy` env):
  python -u src/analysis/behavioral_dla.py --out results/lb1_behavioral_dla_gemma
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
from direct_feature_attribution import DFAHooks  # noqa: E402
from weight_orthogonalization import load_pairs  # noqa: E402
from activation_patching import build_choice_prompt  # noqa: E402

SETS = {
    "M_confirm": "data/contrastive_pairs/v2_1/M_confirm_templated.jsonl",
    "T_confirm": "data/contrastive_pairs/v2_1/T_confirm_templated.jsonl",
}
WRITERS = ["L19MLP", "L20MLP"]
SUPPRESSORS = ["L18H13", "L20H10", "L19H12", "L17H7"]


@torch.no_grad()
def run(model, tok, pairs, device, batch_size, max_tokens, seed):
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    ta_ids = tok.encode("A", add_special_tokens=False)
    tb_ids = tok.encode("B", add_special_tokens=False)
    assert len(ta_ids) == 1 and len(tb_ids) == 1, "A/B must be single tokens"
    ta, tb = ta_ids[0], tb_ids[0]
    w_u = model.get_output_embeddings().weight.float()  # (vocab, d_model)
    du = (w_u[ta] - w_u[tb]).to(device)  # (d_model,)
    w_final = model.model.norm.weight.float().to(device)
    du_scaled = du * (1.0 + w_final)
    eps = cfg.rms_norm_eps

    rng = pyrandom.Random(seed)
    flips = [rng.random() < 0.5 for _ in pairs]
    prompts = [build_choice_prompt(p, f) for p, f in zip(pairs, flips)]
    signs = np.array([-1.0 if f else 1.0 for f in flips])  # flip: shown B is "pos"

    hooks = DFAHooks(model, du_scaled, n_layers - 1)
    hooks.attach()
    # HF hidden_states[-1] is POST-final-RMSNorm; the frozen-r linearization
    # needs the PRE-norm final residual, captured at the norm's input.
    pre = {}
    pre_handle = model.model.norm.register_forward_pre_hook(
        lambda module, inputs: pre.__setitem__("resid", inputs[0].detach()))
    H, M, E, LD_true, LD_recon, LD_capped = [], [], [], [], [], []
    try:
        for i in range(0, len(prompts), batch_size):
            enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_tokens).to(device)
            lens = enc["attention_mask"].sum(1)
            # one-hot readout mask: ONLY the final real token position
            pool = torch.zeros_like(enc["attention_mask"], dtype=torch.bool)
            for r_i, L in enumerate(lens.tolist()):
                pool[r_i, L - 1] = True
            hooks.set_mask(pool)
            out = model(**enc, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states
            for r_i, L in enumerate(lens.tolist()):
                resid = pre["resid"][r_i, L - 1].float()  # pre-norm final residual
                r = torch.rsqrt(resid.pow(2).mean() + eps)
                embed_c = float(hs[0][r_i, L - 1].float() @ du_scaled) * float(r)
                heads_c = hooks.head_dots  # layer -> (b, n_heads)
                h_row = np.stack([heads_c[l][r_i].numpy() for l in range(n_layers)]) * float(r)
                m_row = np.array([float(hooks.mlp_dots[l][r_i]) for l in range(n_layers)]) * float(r)
                a_row = np.array([float(hooks.attn_ln_dots[l][r_i]) for l in range(n_layers)]) * float(r)
                # decomposition target: PRE-softcap logit diff = du @ norm(resid)
                # (hs[-1] is the post-norm state; lm_head is linear on it).
                # out.logits are post-tanh-softcap in Gemma-2, kept for reference.
                LD_true.append(float(hs[-1][r_i, L - 1].float() @ du))
                LD_capped.append(float(out.logits[r_i, L - 1, ta])
                                 - float(out.logits[r_i, L - 1, tb]))
                LD_recon.append(embed_c + a_row.sum() + m_row.sum())
                H.append(h_row)
                M.append(m_row)
                E.append(embed_c)
            print(f"  dla {min(i + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    finally:
        hooks.detach()
        pre_handle.remove()
    H, M, E = np.array(H), np.array(M), np.array(E)
    signs_col = signs[:, None]
    return (H * signs[:, None, None], M * signs_col, E * signs,
            np.array(LD_true) * signs, np.array(LD_recon) * signs,
            np.array(LD_capped) * signs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/lb1_behavioral_dla_gemma")
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
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads

    report = {"model": args.model,
              "final_logit_softcapping": getattr(model.config,
                                                 "final_logit_softcapping", None),
              "note": "decomposition targets the PRE-softcap logit difference "
                      "(linear in the residual stream); the behavioral readout "
                      "(choice_scores) uses post-softcap logits — tanh softcap "
                      "is monotone, so signs/ordering agree, magnitudes compress",
              "sets": {}}
    for name, path in SETS.items():
        pairs = load_pairs(path)
        H, M, E, ld_true, ld_recon, ld_capped = run(
            model, tok, pairs, device, args.batch_size, args.max_tokens, args.seed)
        # exactness: frozen-r linearization should reconstruct the true logit diff
        recon_r = float(np.corrcoef(ld_true, ld_recon)[0, 1])
        recon_ratio = float(np.mean(ld_recon) / np.mean(ld_true))
        # rank all components by mean contribution
        comps = {}
        for l in range(n_layers):
            comps[f"L{l}MLP"] = float(M[:, l].mean())
            for h in range(n_heads):
                comps[f"L{l}H{h}"] = float(H[:, l, h].mean())
        ranked = sorted(comps.items(), key=lambda kv: -abs(kv[1]))
        rank_of = {c: i + 1 for i, (c, _) in enumerate(ranked)}
        report["sets"][name] = {
            "n_pairs": len(pairs),
            "logit_diff_mean_true": float(ld_true.mean()),
            "logit_diff_mean_post_softcap": float(ld_capped.mean()),
            "logit_diff_mean_reconstructed": float(ld_recon.mean()),
            "reconstruction_corr": recon_r, "reconstruction_ratio": recon_ratio,
            "embed_mean": float(E.mean()),
            "top20": ranked[:20],
            "targeted_ranks": {c: {"rank": rank_of[c], "mean_contribution": comps[c]}
                               for c in WRITERS + SUPPRESSORS},
            "mlp_by_layer": [float(M[:, l].mean()) for l in range(n_layers)],
        }
        print(f"{name}: true LD {ld_true.mean():+.3f} recon {ld_recon.mean():+.3f} "
              f"(corr {recon_r:.4f}); top5 {ranked[:5]}")
        print(f"  targeted ranks: " + " ".join(
            f"{c}:{rank_of[c]}" for c in WRITERS + SUPPRESSORS))

    (out / "behavioral_dla.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out}/behavioral_dla.json")


if __name__ == "__main__":
    main()
