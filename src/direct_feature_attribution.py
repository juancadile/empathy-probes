"""Direct Feature Attribution (DFA) — Plan Phase A1, GitHub issue #25.

Which components (attention heads, MLPs) WRITE the empathy direction into the
residual stream?

The residual stream at block L is an exact sum:

    resid_post(L) = embed + sum_{l<=L} (attn_add(l) + mlp_add(l))

where for Gemma-2 the per-block adds are the outputs of the post-attention /
post-feedforward RMSNorms (sandwich norm). Dotting each term with the empathy
direction decomposes the probe projection exactly into per-component
contributions.

Per-head attribution: attn_out = sum_h z_h @ W_O[h] (o_proj has no bias), and
RMSNorm scales every head's contribution by the same per-position scalar
r(s) = rsqrt(mean(attn_out(s)^2) + eps), so

    attn_add(l) = sum_h (z_h @ W_O[h]) * r(s) * (1 + w_ln)

is an exact per-head decomposition conditional on the realized norm
("frozen RMS" attribution).

Two stages, one GPU job:
  1. Layer sweep: mean-diff direction per layer from train pairs, AUROC on
     held-out pairs, pick the best readout layer L*.
  2. DFA at L*: per-head and per-MLP projections onto the direction for every
     component at l <= L*, aggregated as paired (empathic - non_empathic)
     differences.

Outputs (under --out):
  dfa_summary.json       ranking, concentration metrics, layer sweep, checks
  empathy_direction_layer{L}.npy
  component_projections.npz   per-example matrices for further analysis
"""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dfa")


def load_pairs(path: Path, n_pairs: int, seed: int) -> list[dict]:
    pairs = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("empathic_text") and d.get("non_empathic_text"):
                pairs.append({
                    **d,
                    "positive_text": d["empathic_text"],
                    "negative_text": d["non_empathic_text"],
                })
            elif d.get("pos_text") and d.get("neg_text"):
                pairs.append({
                    **d,
                    "positive_text": d["pos_text"],
                    "negative_text": d["neg_text"],
                })
    random.Random(seed).shuffle(pairs)
    return pairs[:n_pairs]


@torch.no_grad()
def pooled_hidden_states(model, tok, texts, batch_size, max_tokens, device):
    """Masked mean-pooled resid_post for every layer. Returns (n_layers+1, n_texts, d)."""
    outs = []
    for i in range(0, len(texts), batch_size):
        batch = tok(
            texts[i : i + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_tokens,
        ).to(device)
        hs = model(**batch, output_hidden_states=True).hidden_states
        mask = batch["attention_mask"].unsqueeze(-1).to(hs[0].dtype)
        denom = mask.sum(dim=1)
        pooled = torch.stack([(h * mask).sum(dim=1) / denom for h in hs])  # (L+1, b, d)
        outs.append(pooled.float().cpu())
        if i % (batch_size * 8) == 0:
            log.info("  pooled %d/%d", i + len(batch["input_ids"]), len(texts))
    return torch.cat(outs, dim=1).numpy()


def layer_sweep(acts, n_pairs, train_frac, seed):
    """acts: (L+1, 2*n_pairs, d) with empathic at even rows, non at odd rows."""
    idx = list(range(n_pairs))
    random.Random(seed).shuffle(idx)
    n_train = int(train_frac * n_pairs)
    train, test = idx[:n_train], idx[n_train:]
    results = []
    for layer in range(acts.shape[0]):
        emp = acts[layer, [2 * i for i in train]]
        non = acts[layer, [2 * i + 1 for i in train]]
        d = emp.mean(0) - non.mean(0)
        d /= np.linalg.norm(d)
        proj_emp = acts[layer, [2 * i for i in test]] @ d
        proj_non = acts[layer, [2 * i + 1 for i in test]] @ d
        auroc = roc_auc_score(
            [1] * len(test) + [0] * len(test), np.concatenate([proj_emp, proj_non])
        )
        results.append({"layer": layer, "auroc": float(auroc)})
    return results, (train, test)


class DFAHooks:
    """Collects pooled per-head and per-MLP projections onto a direction."""

    def __init__(self, model, direction: torch.Tensor, max_layer: int):
        self.d = direction  # (d_model,) float32 on device
        self.max_layer = max_layer
        self.model = model
        cfg = model.config
        self.n_heads = cfg.num_attention_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.eps = cfg.rms_norm_eps
        # Gemma-2 sandwich norms rescale each block's output before the residual
        # add; Llama-style pre-norm blocks write o_proj / down_proj outputs directly.
        self.sandwich = hasattr(model.model.layers[0], "post_feedforward_layernorm")
        self.handles = []
        self.mask = None  # set per batch: (b, s, 1) float
        self.head_dots = {}  # layer -> (b, n_heads)
        self.attn_ln_dots = {}  # layer -> (b,)  exact block-level add, for checks
        self.mlp_dots = {}  # layer -> (b,)

    def set_mask(self, attention_mask):
        self.mask = attention_mask.unsqueeze(-1).float()

    def _o_proj_hook(self, layer_idx):
        layer = self.model.model.layers[layer_idx]
        if self.sandwich:
            w_ln = layer.post_attention_layernorm.weight.float()  # Gemma RMSNorm: (1 + w)
            d_eff = self.d * (1.0 + w_ln)  # (d_model,)
        else:
            d_eff = self.d  # Llama: attn_out enters the residual unscaled
        w_o = layer.self_attn.o_proj.weight.float()  # (d_model, n_heads*head_dim)
        per_dim = d_eff @ w_o  # (n_heads*head_dim,)

        def hook(module, inputs, output):
            z = inputs[0].float()  # (b, s, n_heads*head_dim)
            if self.sandwich:
                attn_out = output.float()  # (b, s, d_model)
                r = torch.rsqrt(attn_out.pow(2).mean(-1, keepdim=True) + self.eps)  # (b,s,1)
                z = z * r
            denom = self.mask.sum(dim=1)  # (b,1)
            z_pooled = (z * self.mask).sum(dim=1) / denom  # (b, n_heads*head_dim)
            dots = (z_pooled * per_dim).view(-1, self.n_heads, self.head_dim).sum(-1)
            self.head_dots[layer_idx] = dots.cpu()

        return layer.self_attn.o_proj.register_forward_hook(hook)

    def _ln_hook(self, layer_idx, which):
        def hook(module, inputs, output):
            add = output.float()  # (b, s, d_model) — exact residual add
            denom = self.mask.sum(dim=1)
            pooled = (add * self.mask).sum(dim=1) / denom
            dots = pooled @ self.d
            (self.attn_ln_dots if which == "attn" else self.mlp_dots)[layer_idx] = dots.cpu()

        layer = self.model.model.layers[layer_idx]
        if self.sandwich:
            mod = layer.post_attention_layernorm if which == "attn" else layer.post_feedforward_layernorm
        else:
            mod = layer.self_attn.o_proj if which == "attn" else layer.mlp.down_proj
        return mod.register_forward_hook(hook)

    def attach(self):
        for l in range(self.max_layer + 1):
            self.handles.append(self._o_proj_hook(l))
            self.handles.append(self._ln_hook(l, "attn"))
            self.handles.append(self._ln_hook(l, "mlp"))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.no_grad()
def run_dfa(model, tok, texts, direction, best_layer, batch_size, max_tokens,
            device, prefixes=None):
    d_vec = torch.tensor(direction, dtype=torch.float32, device=device)
    hooks = DFAHooks(model, d_vec, best_layer)
    hooks.attach()
    n_layers = best_layer + 1
    all_heads, all_attn, all_mlp, all_embed, all_resid = [], [], [], [], []
    try:
        for i in range(0, len(texts), batch_size):
            batch_prefixes = prefixes[i:i + batch_size] if prefixes is not None else None
            encoded = tok(
                texts[i : i + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_tokens,
                return_offsets_mapping=batch_prefixes is not None,
            )
            offsets = encoded.pop("offset_mapping", None)
            batch = encoded.to(device)
            pool_mask = batch["attention_mask"].bool()
            if batch_prefixes is not None:
                for row, prefix in enumerate(batch_prefixes):
                    if prefix is None:
                        continue
                    candidate = (
                        (offsets[row, :, 1] > len(prefix))
                        & batch["attention_mask"][row].cpu().bool()
                    )
                    if candidate.any():
                        pool_mask[row] = candidate.to(device)
            hooks.set_mask(pool_mask)
            hs = model(**batch, output_hidden_states=True).hidden_states
            mask = pool_mask.unsqueeze(-1).float()
            denom = mask.sum(dim=1)
            embed_pool = (hs[0].float() * mask).sum(dim=1) / denom
            resid_pool = (hs[best_layer + 1].float() * mask).sum(dim=1) / denom
            all_embed.append((embed_pool @ d_vec).cpu())
            all_resid.append((resid_pool @ d_vec).cpu())
            all_heads.append(torch.stack([hooks.head_dots[l] for l in range(n_layers)], 1))
            all_attn.append(torch.stack([hooks.attn_ln_dots[l] for l in range(n_layers)], 1))
            all_mlp.append(torch.stack([hooks.mlp_dots[l] for l in range(n_layers)], 1))
            if i % (batch_size * 8) == 0:
                log.info("  dfa %d/%d", i + len(batch["input_ids"]), len(texts))
    finally:
        hooks.detach()
    return (
        torch.cat(all_heads).numpy(),  # (n, L, H)
        torch.cat(all_attn).numpy(),  # (n, L)
        torch.cat(all_mlp).numpy(),  # (n, L)
        torch.cat(all_embed).numpy(),  # (n,)
        torch.cat(all_resid).numpy(),  # (n,)
    )


def concentration(scores: np.ndarray) -> dict:
    a = np.sort(np.abs(scores))[::-1]
    total = a.sum()
    share = np.cumsum(a) / total
    return {
        "n_components": int(a.size),
        "top5_share": float(share[min(4, a.size - 1)]),
        "top10_share": float(share[min(9, a.size - 1)]),
        "top20_share": float(share[min(19, a.size - 1)]),
        "n_for_50pct": int(np.searchsorted(share, 0.5) + 1),
        "n_for_80pct": int(np.searchsorted(share, 0.8) + 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-2-9b-it")
    p.add_argument("--pairs", default="data/contrastive_pairs/merged_cleaned_pairs.jsonl")
    p.add_argument("--n-pairs", type=int, default=300, help="pairs for layer sweep / direction")
    p.add_argument("--dfa-pairs", type=int, default=200, help="pairs for the DFA pass")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--layer", type=int, default=None, help="force readout layer (skip sweep choice)")
    p.add_argument("--direction", help="external direction .npy; requires --layer")
    p.add_argument(
        "--decision-tail",
        action="store_true",
        help="pool only text after each pair's shared_prefix during DFA",
    )
    p.add_argument("--out", default="results/dfa")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    log.info("loading %s (bf16) on %s", args.model, device)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:  # Llama-3.1 ships without one
        tok.pad_token = tok.eos_token
    # eager attention: Gemma-2 uses attention-logit soft-capping, which some fused
    # kernels skip — activations must be exact for attribution
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device)
    model.eval()

    pairs = load_pairs(Path(args.pairs), args.n_pairs, args.seed)
    log.info("loaded %d pairs", len(pairs))
    texts, prefixes = [], []
    for pr in pairs:  # even index = positive, odd = negative
        texts.extend([pr["positive_text"], pr["negative_text"]])
        prefixes.extend([pr.get("shared_prefix"), pr.get("shared_prefix")])

    if args.direction:
        if args.layer is None:
            p.error("--direction requires --layer")
        direction = np.load(args.direction).astype(np.float32)
        direction /= np.linalg.norm(direction)
        block_sweep = []
        best_layer = args.layer
        best_auroc = None
        log.info("using external direction at block %d", best_layer)
    else:
        log.info("stage 1: layer sweep over %d texts", len(texts))
        acts = pooled_hidden_states(model, tok, texts, args.batch_size, args.max_tokens, device)
        sweep, (train_idx, _) = layer_sweep(acts, len(pairs), args.train_frac, args.seed)
        # hidden_states[0] is the (scaled) embedding stream; block L is index L+1
        block_sweep = sweep[1:]
        best = max(block_sweep, key=lambda r: r["auroc"])
        best_layer = args.layer if args.layer is not None else best["layer"] - 1
        best_auroc = next(r["auroc"] for r in block_sweep if r["layer"] - 1 == best_layer)
        log.info("best readout block L*=%d (test AUROC %.4f)", best_layer, best_auroc)

        emp = acts[best_layer + 1, [2 * i for i in train_idx]]
        non = acts[best_layer + 1, [2 * i + 1 for i in train_idx]]
        direction = emp.mean(0) - non.mean(0)
        direction /= np.linalg.norm(direction)
    np.save(out / f"empathy_direction_layer{best_layer}.npy", direction)

    dfa_pairs = pairs[: args.dfa_pairs]
    dfa_texts = texts[: 2 * len(dfa_pairs)]
    log.info("stage 2: DFA over %d texts, components at blocks 0..%d", len(dfa_texts), best_layer)
    heads, attn_ln, mlp, embed, resid = run_dfa(
        model, tok, dfa_texts, direction, best_layer, args.batch_size, args.max_tokens,
        device, prefixes[:len(dfa_texts)] if args.decision_tail else None,
    )

    # sanity 1: per-head decomposition sums to the exact block-level attn add
    head_sum_err = float(np.abs(heads.sum(-1) - attn_ln).max())
    # sanity 2: exact residual reconstruction: embed + sum(attn) + sum(mlp) = resid_post(L*)
    recon = embed + attn_ln.sum(-1) + mlp.sum(-1)
    recon_err = float(np.abs(recon - resid).max() / (np.abs(resid).max() + 1e-9))
    log.info("checks: head-sum err %.4g, resid recon rel err %.4g", head_sum_err, recon_err)

    # paired differences (empathic - non_empathic), per component
    def paired(x):  # x: (2n, ...) -> (n, ...)
        return x[0::2] - x[1::2]

    head_diff, mlp_diff = paired(heads), paired(mlp)
    comps = []
    for l in range(best_layer + 1):
        for h in range(heads.shape[-1]):
            dmean, dstd = head_diff[:, l, h].mean(), head_diff[:, l, h].std()
            comps.append({
                "component": f"L{l}H{h}", "type": "head", "layer": l, "head": h,
                "mean_paired_diff": float(dmean),
                "cohens_d": float(dmean / (dstd + 1e-9)),
            })
        dmean, dstd = mlp_diff[:, l].mean(), mlp_diff[:, l].std()
        comps.append({
            "component": f"L{l}MLP", "type": "mlp", "layer": l, "head": None,
            "mean_paired_diff": float(dmean),
            "cohens_d": float(dmean / (dstd + 1e-9)),
        })
    comps.sort(key=lambda c: abs(c["mean_paired_diff"]), reverse=True)
    scores = np.array([c["mean_paired_diff"] for c in comps])
    conc = concentration(scores)
    total_diff = float(paired(resid).mean())
    log.info("total resid paired diff %.4f; top10 share %.3f; %d comps for 80%%",
             total_diff, conc["top10_share"], conc["n_for_80pct"])

    summary = {
        "model": args.model,
        "n_pairs_sweep": len(pairs),
        "n_pairs_dfa": len(dfa_pairs),
        "seed": args.seed,
        "direction_source": args.direction or "estimated_from_pairs",
        "pooling": "decision_tail" if args.decision_tail else "masked_mean_all_tokens",
        "layer_sweep": [{"block": r["layer"] - 1, "auroc": r["auroc"]} for r in block_sweep],
        "best_layer": best_layer,
        "best_layer_auroc": best_auroc,
        "checks": {"head_sum_abs_err": head_sum_err, "resid_recon_rel_err": recon_err},
        "total_resid_paired_diff": total_diff,
        "embed_paired_diff": float(paired(embed).mean()),
        "concentration": conc,
        "top_components": comps[:50],
    }
    with open(out / "dfa_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(
        out / "component_projections.npz",
        heads=heads, attn_ln=attn_ln, mlp=mlp, embed=embed, resid=resid,
    )
    log.info("wrote %s", out / "dfa_summary.json")


if __name__ == "__main__":
    main()
