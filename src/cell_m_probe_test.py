"""Cell M probe test (Plan A5 test 3, issue #33).

Do the DFA empathy directions separate matched-lexicon minimal pairs?
Both sides of each pair are identical except the decision clause, so a
lexical/bag-of-words direction sees (almost) identical texts -> AUROC ~0.5,
while a decision-reading direction separates them.

Reports, per direction (L8, L20):
  - AUROC on full-text mean-pooled activations
  - AUROC on decision-clause tokens only (positions after the shared prefix)
  - mean paired projection difference (pos - neg), full vs decision-only

Usage (on the Spark, after the DFA runs):
  python src/cell_m_probe_test.py \
    --directions results/dfa_gemma2_9b_it/empathy_direction_layer8.npy:8 \
                 results/dfa_gemma2_9b_it_L20/empathy_direction_layer20.npy:20
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cellm")


@torch.no_grad()
def pooled_full_and_decision(model, tok, text, prefix, layers, device, max_tokens=768):
    """Mean-pooled resid_post at given blocks: full text and decision-only spans."""
    n_prefix = len(tok(prefix, add_special_tokens=True)["input_ids"])
    batch = tok(text, return_tensors="pt", truncation=True, max_length=max_tokens).to(device)
    hs = model(**batch, output_hidden_states=True).hidden_states
    out = {}
    for L in layers:
        h = hs[L + 1][0].float()  # (seq, d)
        cut = min(n_prefix - 2, h.shape[0] - 1)  # start slightly before clause boundary
        out[L] = (h.mean(0).cpu().numpy(), h[cut:].mean(0).cpu().numpy())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-2-9b-it")
    p.add_argument("--pairs", default="data/contrastive_pairs/v2_1/M_templated.jsonl")
    p.add_argument("--directions", nargs="+", required=True, help="path.npy:LAYER entries")
    p.add_argument("--out", default="results/lexical_battery/cell_m.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dirs = {}
    for spec in args.directions:
        path, layer = spec.rsplit(":", 1)
        dirs[int(layer)] = np.load(path)
    layers = sorted(dirs)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device)
    model.eval()

    pairs = [json.loads(l) for l in open(args.pairs)]
    log.info("scoring %d matched-lexicon pairs at blocks %s", len(pairs), layers)

    proj = {L: {"pos_full": [], "neg_full": [], "pos_dec": [], "neg_dec": []} for L in layers}
    for i, pr in enumerate(pairs):
        for side in ("pos", "neg"):
            reps = pooled_full_and_decision(model, tok, pr[f"{side}_text"], pr["shared_prefix"], layers, device)
            for L in layers:
                full, dec = reps[L]
                proj[L][f"{side}_full"].append(float(full @ dirs[L]))
                proj[L][f"{side}_dec"].append(float(dec @ dirs[L]))
        if i % 10 == 0:
            log.info("  %d/%d", i + 1, len(pairs))

    results = {"n_pairs": len(pairs), "model": args.model, "layers": {}}
    labels = [1] * len(pairs) + [0] * len(pairs)
    for L in layers:
        pf, nf = np.array(proj[L]["pos_full"]), np.array(proj[L]["neg_full"])
        pd_, nd = np.array(proj[L]["pos_dec"]), np.array(proj[L]["neg_dec"])
        results["layers"][str(L)] = {
            "auroc_full_text": float(roc_auc_score(labels, np.concatenate([pf, nf]))),
            "auroc_decision_tokens": float(roc_auc_score(labels, np.concatenate([pd_, nd]))),
            "mean_paired_diff_full": float((pf - nf).mean()),
            "mean_paired_diff_decision": float((pd_ - nd).mean()),
        }
        log.info("L%d: full-text AUROC %.4f | decision-tokens AUROC %.4f",
                 L, results["layers"][str(L)]["auroc_full_text"],
                 results["layers"][str(L)]["auroc_decision_tokens"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
