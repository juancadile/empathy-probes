"""Component ablation / causal necessity — Plan Phase A2, issues #11 #12.

DFA (issue #25) found which components WRITE the empathy/decision direction.
This script tests which are CAUSALLY NECESSARY, via mean-ablation in
TransformerLens (stack equivalence vs raw hooks verified: cos(TL,HF)=1.000 at
blocks 8 and 20).

Stimuli: cell M matched-lexicon pairs (issue #33) — the lexically controlled
set where the L20 direction shows a real decision signal. Two metrics per
ablated component:

  1. separation: mean paired difference of decision-token projections onto
     d_L20 (pos - neg). A causally necessary component's ablation shrinks it.
  2. behavior: forced-choice logit diff. Each M context becomes
     "<context+ack> If I had to choose one action right now: (A) <clause1>
     (B) <clause2>. I choose (" and we read logit(A) - logit(B), sign-corrected
     for counterbalanced option order.

Components tested: union of top-K writers from the L8 and L20 DFA rankings
(positive and negative), plus N random non-top components as controls.

Usage (Spark, `empathy` env):
  python src/activation_patching.py \
    --dfa results/dfa_gemma2_9b_it/dfa_summary.json results/dfa_gemma2_9b_it_L20/dfa_summary.json \
    --direction results/dfa_gemma2_9b_it_L20/empathy_direction_layer20.npy \
    --out results/patching_gemma2_9b_it
"""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("patch")

READOUT_BLOCK = 20  # d_L20 is read at resid_post of this block


def load_components(dfa_paths, top_k, n_random, seed, n_layers, n_heads):
    comps, seen = [], set()
    for p in dfa_paths:
        d = json.load(open(p))
        for c in d["top_components"][:top_k]:
            key = (c["layer"], c["head"])  # head=None => MLP
            if key not in seen:
                seen.add(key)
                comps.append({"layer": c["layer"], "head": c["head"], "source": "dfa"})
    rng = random.Random(seed)
    while sum(c["source"] == "random" for c in comps) < n_random:
        layer = rng.randrange(0, READOUT_BLOCK + 1)
        head = rng.choice([None] + list(range(n_heads)))
        if (layer, head) not in seen:
            seen.add((layer, head))
            comps.append({"layer": layer, "head": head, "source": "random"})
    return comps


def build_choice_prompt(item, flip):
    pos_clause = item["pos_text"][len(item["shared_prefix"]):].strip()
    neg_clause = item["neg_text"][len(item["shared_prefix"]):].strip()
    a, b = (neg_clause, pos_clause) if flip else (pos_clause, neg_clause)
    return (
        item["shared_prefix"]
        + f"If I had to choose one action right now: (A) {a} (B) {b} I choose ("
    )


class Ablator:
    """Mean-ablation hooks for one component at a time."""

    def __init__(self, model, means_z, means_mlp):
        self.model = model
        self.means_z = means_z      # {layer: tensor(heads, d_head)}
        self.means_mlp = means_mlp  # {layer: tensor(d_model)}
        self.active = None          # (layer, head or None)

    def hooks(self):
        if self.active is None:
            return []
        layer, head = self.active
        if head is None:
            mean = self.means_mlp[layer]
            def hook(value, hook):
                value[:] = mean.to(value.dtype)
                return value
            return [(f"blocks.{layer}.hook_mlp_out", hook)]
        mean = self.means_z[layer][head]
        def hook(value, hook):  # value: (batch, pos, head, d_head)
            value[:, :, head, :] = mean.to(value.dtype)
            return value
        return [(f"blocks.{layer}.attn.hook_z", hook)]


@torch.no_grad()
def run_metrics(model, ablator, m_items, choice_prompts, choice_flips, d_vec, tok_a, tok_b, prefix_lens):
    """Returns (separation, behavior) under the current ablation setting."""
    diffs = []
    for i, item in enumerate(m_items):
        projs = {}
        for side in ("pos_text", "neg_text"):
            _, cache = model.run_with_cache(
                item[side],
                names_filter=f"blocks.{READOUT_BLOCK}.hook_resid_post",
                fwd_hooks=ablator.hooks(),
                return_type=None,
            )
            h = cache[f"blocks.{READOUT_BLOCK}.hook_resid_post"][0].float()
            cut = min(prefix_lens[i] - 2, h.shape[0] - 1)
            projs[side] = float(h[cut:].mean(0) @ d_vec)
        diffs.append(projs["pos_text"] - projs["neg_text"])
    separation = float(np.mean(diffs))

    logit_diffs = []
    for prompt, flip in zip(choice_prompts, choice_flips):
        logits = model.run_with_hooks(prompt, fwd_hooks=ablator.hooks(), return_type="logits")
        la, lb = float(logits[0, -1, tok_a]), float(logits[0, -1, tok_b])
        help_minus_task = (lb - la) if flip else (la - lb)
        logit_diffs.append(help_minus_task)
    behavior = float(np.mean(logit_diffs))
    return separation, behavior


@torch.no_grad()
def compute_means(model, texts, n_layers):
    """Dataset-mean z per (layer, head) and mlp_out per layer, pooled over positions."""
    sums_z, sums_mlp, count = {}, {}, 0
    names = lambda n: n.endswith("attn.hook_z") or n.endswith("hook_mlp_out")
    for t in texts:
        _, cache = model.run_with_cache(t, names_filter=names, return_type=None)
        for L in range(n_layers):
            z = cache[f"blocks.{L}.attn.hook_z"][0].float()      # (pos, head, d_head)
            m = cache[f"blocks.{L}.hook_mlp_out"][0].float()     # (pos, d_model)
            sums_z[L] = sums_z.get(L, 0) + z.sum(0)
            sums_mlp[L] = sums_mlp.get(L, 0) + m.sum(0)
        count += z.shape[0]
    return ({L: (s / count) for L, s in sums_z.items()},
            {L: (s / count) for L, s in sums_mlp.items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-2-9b-it")
    p.add_argument("--pairs", default="data/contrastive_pairs/v2_1/M_templated.jsonl")
    p.add_argument("--dfa", nargs="+", required=True)
    p.add_argument("--direction", required=True)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--n-random", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/patching_gemma2_9b_it")
    args = p.parse_args()

    from transformer_lens import HookedTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = HookedTransformer.from_pretrained_no_processing(
        args.model, dtype=torch.bfloat16, device=device
    )
    model.eval()
    n_heads = model.cfg.n_heads

    m_items = [json.loads(l) for l in open(args.pairs)]
    d_vec = torch.tensor(np.load(args.direction), dtype=torch.float32, device=device)

    comps = load_components(args.dfa, args.top_k, args.n_random, args.seed,
                            READOUT_BLOCK + 1, n_heads)
    log.info("%d components (%d dfa + %d random controls)",
             len(comps), sum(c['source'] == 'dfa' for c in comps),
             sum(c['source'] == 'random' for c in comps))

    prefix_lens = [len(model.to_tokens(it["shared_prefix"])[0]) for it in m_items]
    rng = random.Random(args.seed)
    choice_flips = [rng.random() < 0.5 for _ in m_items]
    choice_prompts = [build_choice_prompt(it, f) for it, f in zip(m_items, choice_flips)]
    tok_a = model.to_single_token("A")
    tok_b = model.to_single_token("B")

    all_texts = [t for it in m_items for t in (it["pos_text"], it["neg_text"])]
    log.info("computing dataset means over %d texts x %d blocks", len(all_texts), READOUT_BLOCK + 1)
    means_z, means_mlp = compute_means(model, all_texts, READOUT_BLOCK + 1)
    ablator = Ablator(model, means_z, means_mlp)

    log.info("baseline (no ablation)")
    base_sep, base_beh = run_metrics(model, ablator, m_items, choice_prompts,
                                     choice_flips, d_vec, tok_a, tok_b, prefix_lens)
    log.info("baseline: separation %.4f | behavior(help-task logit diff) %.4f", base_sep, base_beh)

    results = []
    for i, c in enumerate(comps):
        ablator.active = (c["layer"], c["head"])
        sep, beh = run_metrics(model, ablator, m_items, choice_prompts,
                               choice_flips, d_vec, tok_a, tok_b, prefix_lens)
        name = f"L{c['layer']}" + ("MLP" if c["head"] is None else f"H{c['head']}")
        rec = {"component": name, **c,
               "separation": sep, "delta_separation": sep - base_sep,
               "behavior": beh, "delta_behavior": beh - base_beh}
        results.append(rec)
        log.info("[%d/%d] %s (%s): dSep %+.4f dBeh %+.4f",
                 i + 1, len(comps), name, c["source"], rec["delta_separation"], rec["delta_behavior"])
    ablator.active = None

    results.sort(key=lambda r: r["delta_separation"])
    with open(out / "ablation_results.json", "w") as f:
        json.dump({"model": args.model, "readout_block": READOUT_BLOCK,
                   "baseline_separation": base_sep, "baseline_behavior": base_beh,
                   "n_pairs": len(m_items), "results": results}, f, indent=2)
    log.info("wrote %s", out / "ablation_results.json")


if __name__ == "__main__":
    main()
