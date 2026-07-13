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
import hashlib
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


def save_raw_npz(path, **arrays):
    """Persist raw arrays as compressed NPZ; return a JSON-able pointer with
    the file's sha256 so the summary JSON can reference the exact bytes."""
    np.savez_compressed(path, **arrays)
    return {"path": str(path),
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


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
            np.array(LD_capped) * signs, signs)


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
        H, M, E, ld_true, ld_recon, ld_capped, signs = run(
            model, tok, pairs, device, args.batch_size, args.max_tokens, args.seed)
        # auditability (LB1): persist the flip-corrected raw arrays BEFORE the
        # reconstruction gates, so a gate failure still leaves the evidence.
        # Row i corresponds to pairs[i] (input JSONL order); every array is
        # already multiplied by flip_signs[i] — identical to what the rankings
        # below consume. flip_signs recovers the as-run (uncorrected) values.
        raw_ptr = save_raw_npz(
            out / f"behavioral_dla_{name}_raw.npz",
            H=H, M=M, E=E,
            ld_true_presoftcap=ld_true,
            ld_recon_presoftcap=ld_recon,
            ld_post_softcap=ld_capped,
            flip_signs=signs,
            scenario_ids=np.array([p.get("scenario_id", str(i))
                                   for i, p in enumerate(pairs)]),
            pair_indices=np.array([int(p.get("pair_index", i))
                                   for i, p in enumerate(pairs)]),
        )
        # exactness: frozen-r linearization should reconstruct the true logit diff
        recon_r = float(np.corrcoef(ld_true, ld_recon)[0, 1])
        recon_ratio = float(np.mean(ld_recon) / np.mean(ld_true))
        # per-example error gates (pre-softcap target): scale-stable relative
        # error guards small-|LD| examples via the 1e-3 floor
        abs_err = np.abs(ld_recon - ld_true)
        rel_err = abs_err / np.maximum(np.abs(ld_true), 1e-3)
        err_stats = {
            "max_abs": float(abs_err.max()),
            "median_abs": float(np.median(abs_err)),
            "p95_abs": float(np.percentile(abs_err, 95)),
            "max_rel": float(rel_err.max()),
            "median_rel": float(np.median(rel_err)),
            "p95_rel": float(np.percentile(rel_err, 95)),
            "gates": {"max_abs": 0.05, "p95_rel": 0.02},
        }
        if err_stats["max_abs"] > 0.05 or err_stats["p95_rel"] > 0.02:
            raise RuntimeError(
                f"{name}: DLA reconstruction gate failed "
                f"(max_abs={err_stats['max_abs']:.4f} > 0.05 or "
                f"p95_rel={err_stats['p95_rel']:.4f} > 0.02) — "
                f"decomposition not exact, aborting before ranking")
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
            "raw_npz": {
                **raw_ptr,
                "arrays": "H (n,layers,heads) per-head, M (n,layers) per-MLP, "
                          "E (n,) embed contributions; ld_true_presoftcap / "
                          "ld_recon_presoftcap / ld_post_softcap (n,) logit "
                          "diffs; flip_signs (n,) in {-1,+1}; scenario_ids, "
                          "pair_indices (n,)",
                "note": "rows follow input JSONL order; contribution and "
                        "logit-diff arrays are flip-corrected (multiplied by "
                        "flip_signs) exactly as the rankings consume them",
            },
            "logit_diff_mean_true": float(ld_true.mean()),
            "logit_diff_mean_post_softcap": float(ld_capped.mean()),
            "logit_diff_mean_reconstructed": float(ld_recon.mean()),
            "reconstruction_corr": recon_r, "reconstruction_ratio": recon_ratio,
            "reconstruction_error": err_stats,
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
