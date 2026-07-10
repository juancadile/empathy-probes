"""Lexical vs behavioral separability stress tests — Plan Phase A5, issue #33.

Tests 1 & 2 of the battery (shuffled-token control, lexical-direction
regression, logit-lens readout of the direction). Tests 3 (matched-lexicon
minimal pairs) and 4 (prospective probing via the EIA harness) live elsewhere.

Design:
  1. Shuffled-token control. Shuffling word order within each text destroys
     decisions/syntax but preserves the lexicon exactly. Three AUROCs per layer:
       orig->orig   baseline (as in DFA stage 1)
       shuf->shuf   how much a purely lexical, order-free signal supports
       orig->shuf   does the original direction transfer to bag-of-words?
  2. Lexical-direction regression. Orthogonalize d_L against the embedding-
     stream direction (hs[0]) and jointly against blocks 0-2; residual AUROC.
  3. Logit lens. Top vocabulary tokens of W_U @ d_L. Suggestive only.

Usage (on the Spark):
  python src/lexical_battery.py --layers 8 20 --out results/lexical_battery
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

from direct_feature_attribution import load_pairs, pooled_hidden_states

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("lexbat")


def shuffle_words(text: str, seed: int) -> str:
    words = text.split()
    random.Random(seed).shuffle(words)
    return " ".join(words)


def direction_and_auroc(acts_train, acts_test, train_idx, test_idx, d=None):
    """Mean-diff direction from acts_train (unless given); AUROC on acts_test."""
    if d is None:
        emp = acts_train[[2 * i for i in train_idx]]
        non = acts_train[[2 * i + 1 for i in train_idx]]
        d = emp.mean(0) - non.mean(0)
        d = d / np.linalg.norm(d)
    proj_e = acts_test[[2 * i for i in test_idx]] @ d
    proj_n = acts_test[[2 * i + 1 for i in test_idx]] @ d
    auroc = roc_auc_score([1] * len(test_idx) + [0] * len(test_idx), np.concatenate([proj_e, proj_n]))
    return d, float(auroc)


def orthogonalize(d, basis_vecs):
    """Project d onto the orthogonal complement of span(basis_vecs)."""
    B = np.stack([b / np.linalg.norm(b) for b in basis_vecs])
    Q, _ = np.linalg.qr(B.T)
    d_res = d - Q @ (Q.T @ d)
    return d_res / np.linalg.norm(d_res)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-2-9b-it")
    p.add_argument("--pairs", default="data/contrastive_pairs/merged_cleaned_pairs.jsonl")
    p.add_argument("--n-pairs", type=int, default=300)
    p.add_argument("--layers", type=int, nargs="+", default=[8, 20])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/lexical_battery")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device)
    model.eval()

    # identical sampling to the DFA runs (same seed) for comparability
    pairs = load_pairs(Path(args.pairs), args.n_pairs, args.seed)
    texts = []
    for pr in pairs:
        texts.append(pr["empathic_text"])
        texts.append(pr["non_empathic_text"])
    shuf_texts = [shuffle_words(t, args.seed + j) for j, t in enumerate(texts)]

    idx = list(range(len(pairs)))
    random.Random(args.seed).shuffle(idx)
    n_train = int(args.train_frac * len(pairs))
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    log.info("extracting activations: original texts")
    acts_o = pooled_hidden_states(model, tok, texts, args.batch_size, args.max_tokens, device)
    log.info("extracting activations: shuffled texts")
    acts_s = pooled_hidden_states(model, tok, shuf_texts, args.batch_size, args.max_tokens, device)

    results = {"model": args.model, "n_pairs": len(pairs), "seed": args.seed, "layers": {}}
    for L in args.layers:
        h = L + 1  # hidden_states index for block L
        d_orig, au_oo = direction_and_auroc(acts_o[h], acts_o[h], train_idx, test_idx)
        _, au_ss = direction_and_auroc(acts_s[h], acts_s[h], train_idx, test_idx)
        _, au_os = direction_and_auroc(None, acts_s[h], train_idx, test_idx, d=d_orig)

        # lexical directions from original texts: embedding stream + blocks 0-2
        lex_dirs = []
        for hh in (0, 1, 2, 3):  # hs[0]=embed, hs[1..3]=blocks 0..2
            emp = acts_o[hh][[2 * i for i in train_idx]]
            non = acts_o[hh][[2 * i + 1 for i in train_idx]]
            lex_dirs.append(emp.mean(0) - non.mean(0))
        d_min = orthogonalize(d_orig, lex_dirs[:1])       # embed only
        d_full = orthogonalize(d_orig, lex_dirs)          # embed + blocks 0-2
        _, au_min = direction_and_auroc(None, acts_o[h], train_idx, test_idx, d=d_min)
        _, au_full = direction_and_auroc(None, acts_o[h], train_idx, test_idx, d=d_full)

        cos_lex = float(d_orig @ (lex_dirs[0] / np.linalg.norm(lex_dirs[0])))

        # logit lens: top tokens of W_U @ d (Gemma ties embeddings; use embed_tokens)
        W = model.get_input_embeddings().weight.detach().float()  # (V, d)
        logits = W.cpu().numpy() @ d_orig
        top = np.argsort(-logits)[:20]
        bot = np.argsort(logits)[:20]
        top_toks = [tok.decode([int(t)]).strip() for t in top]
        bot_toks = [tok.decode([int(t)]).strip() for t in bot]

        results["layers"][str(L)] = {
            "auroc_orig_orig": au_oo,
            "auroc_shuf_shuf": au_ss,
            "auroc_origdir_on_shuf": au_os,
            "auroc_after_embed_regression": au_min,
            "auroc_after_embed_plus_blocks012_regression": au_full,
            "cos_with_embed_direction": cos_lex,
            "logit_lens_top_tokens": top_toks,
            "logit_lens_bottom_tokens": bot_toks,
        }
        log.info("L%d: orig %.4f | shuf-trained %.4f | origdir-on-shuf %.4f | "
                 "resid(embed) %.4f | resid(embed+0-2) %.4f | cos_lex %.3f",
                 L, au_oo, au_ss, au_os, au_min, au_full, cos_lex)

    with open(out / "lexical_battery.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info("wrote %s", out / "lexical_battery.json")


if __name__ == "__main__":
    main()
