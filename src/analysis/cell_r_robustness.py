"""Cell R robustness analysis (Lazar robustness criterion; owed since Stage 0).

Cell R holds perturbed variants of cell-A pairs (paraphrase / name_swap /
register_shift). Test: does the FROZEN direction of record still separate
pos/neg under each perturbation type, compared to the unperturbed cell-A
reference AUROC? Pooling matches cell A (masked mean over all tokens).

Usage (Spark, `empathy` env):
  python -u src/analysis/cell_r_robustness.py \
    --model google/gemma-2-9b-it \
    --direction results/controlled_directions_gemma2_9b_it/direction_M_block20.npy \
    --block 20 --out results/cell_r_robustness_gemma2_9b_it
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def load_rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


@torch.no_grad()
def pooled_at_block(model, tok, texts, block, batch_size, max_tokens, device):
    outs = []
    for i in range(0, len(texts), batch_size):
        batch = tok(texts[i:i + batch_size], return_tensors="pt", padding=True,
                    truncation=True, max_length=max_tokens).to(device)
        hs = model(**batch, output_hidden_states=True).hidden_states[block + 1]
        mask = batch["attention_mask"].unsqueeze(-1).to(hs.dtype)
        outs.append(((hs * mask).sum(1) / mask.sum(1)).float().cpu())
    return torch.cat(outs).numpy()


def auroc(pos, neg, d):
    scores = np.concatenate([pos @ d, neg @ d])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return float(roc_auc_score(labels, scores))


def boot_ci(pos, neg, d, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(pos), len(pos))
        vals.append(auroc(pos[i], neg[i], d))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--direction", required=True)
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--r-pairs", default="data/contrastive_pairs/v2_1/R_claude-haiku.jsonl")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", default="results/cell_r_robustness_gemma2_9b_it")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager").to(device)
    model.eval()
    d = np.load(args.direction).astype(np.float32)
    d /= np.linalg.norm(d)

    rows = load_rows(args.r_pairs)
    texts = [t for r in rows for t in (r["pos_text"], r["neg_text"])]
    acts = pooled_at_block(model, tok, texts, args.block, args.batch_size,
                           args.max_tokens, device)
    pos, neg = acts[0::2], acts[1::2]

    # unperturbed reference: cached cell-A activations if present
    ref = None
    ref_npz = Path(args.direction).parent / "activations" / "cell_A.npz"
    if ref_npz.exists():
        f = np.load(ref_npz)
        ref = auroc(f["pos"][args.block + 1].astype(np.float32),
                    f["neg"][args.block + 1].astype(np.float32), d)

    types = [r["perturbation"] for r in rows]
    report = {"model": args.model, "block": args.block, "n_pairs": len(rows),
              "reference_cell_A_auroc": ref, "by_perturbation": {}}
    print(f"reference (unperturbed A) AUROC: {ref}")
    overall = auroc(pos, neg, d)
    report["overall_auroc"] = overall
    report["overall_ci95"] = boot_ci(pos, neg, d)
    print(f"R overall: {overall:.4f} CI {report['overall_ci95']}")
    for t in sorted(set(types)):
        idx = np.array([i for i, x in enumerate(types) if x == t])
        a = auroc(pos[idx], neg[idx], d)
        ci = boot_ci(pos[idx], neg[idx], d)
        report["by_perturbation"][t] = {"auroc": a, "ci95": ci, "n": len(idx)}
        print(f"  {t:15s} AUROC {a:.4f} CI {ci} (n={len(idx)})")

    (out / "cell_r.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out}/cell_r.json")


if __name__ == "__main__":
    main()
