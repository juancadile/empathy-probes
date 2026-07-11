"""Capability benchmark under costly-helping weight edits.

Publication gate 2 (after 2ed963d): the pilot's neutral-drift check (12 prompts,
KL + top-1 agreement) is a smoke test, not a capability benchmark. This runs a
real one under each edit condition:

  - MMLU: n stratified test questions, 0-shot, 4-way letter-choice by logits
  - Perplexity: wikitext-2-raw test slice, sliding non-overlapping windows

Conditions: baseline, positive_writers (k=2), suppressors (k=4), targeted (k=6).
The selectivity claim requires MMLU accuracy and perplexity to be ~unchanged
while helping behavior moves (shown elsewhere).

Usage (Spark, `empathy` env; needs `datasets`):
  python -u src/capability_eval.py \
    --direction results/controlled_directions_gemma2_9b_it/direction_M_block20.npy \
    --out results/capability_eval_gemma2_9b_it
"""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch

try:
    from src.weight_orthogonalization import (
        POSITIVE_WRITERS, SUPPRESSORS, TARGETED,
        orthogonalize_component, parse_component, restore_weights, snapshot_weights,
    )
except ModuleNotFoundError:
    from weight_orthogonalization import (
        POSITIVE_WRITERS, SUPPRESSORS, TARGETED,
        orthogonalize_component, parse_component, restore_weights, snapshot_weights,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("capability")

LETTERS = ["A", "B", "C", "D"]


def sample_mmlu(n, seed):
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), n)
    items = []
    for i in indices:
        row = ds[i]
        items.append({"question": row["question"], "choices": row["choices"],
                      "answer": int(row["answer"]), "subject": row["subject"]})
    return items


def mmlu_prompt(item):
    lines = [f"Question: {item['question']}"]
    for letter, choice in zip(LETTERS, item["choices"]):
        lines.append(f"{letter}) {choice}")
    lines.append("Answer: (")
    return "\n".join(lines)


@torch.no_grad()
def mmlu_accuracy(model, tokenizer, items, batch_size, max_tokens, device):
    letter_ids = [tokenizer.encode(l, add_special_tokens=False)[0] for l in LETTERS]
    correct = []
    prompts = [mmlu_prompt(it) for it in items]
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_tokens).to(device)
        logits = model(**enc, use_cache=False).logits
        lengths = enc["attention_mask"].sum(1)
        for row, length in enumerate(lengths.tolist()):
            option_logits = [float(logits[row, length - 1, t]) for t in letter_ids]
            pred = int(np.argmax(option_logits))
            correct.append(pred == items[start + row]["answer"])
    return float(np.mean(correct)), correct


@torch.no_grad()
def wikitext_nll(model, tokenizer, device, n_chars=200_000, window=1024):
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(t for t in ds["text"] if t.strip())[:n_chars]
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    total_nll, total_tokens = 0.0, 0
    for start in range(0, ids.shape[0] - 1, window):
        chunk = ids[start:start + window + 1]
        if chunk.shape[0] < 2:
            break
        inp = chunk[:-1][None, :].to(device)
        tgt = chunk[1:][None, :].to(device)
        logits = model(inp, use_cache=False).logits.float()
        nll = torch.nn.functional.cross_entropy(
            logits[0], tgt[0], reduction="sum"
        )
        total_nll += float(nll)
        total_tokens += tgt.shape[1]
    return {"nll_per_token": total_nll / total_tokens,
            "perplexity": float(np.exp(total_nll / total_tokens)),
            "n_tokens": total_tokens}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2-9b-it")
    parser.add_argument("--direction", required=True)
    parser.add_argument("--n-mmlu", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/capability_eval_gemma2_9b_it")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device)
    model.eval()
    direction = torch.tensor(np.load(args.direction), dtype=torch.float32, device=device)

    items = sample_mmlu(args.n_mmlu, args.seed)
    log.info("MMLU sample: %d questions", len(items))

    conditions = {
        "baseline": [],
        "positive_writers_k2": [parse_component(v) for v in POSITIVE_WRITERS.split(",")],
        "suppressors_k4": [parse_component(v) for v in SUPPRESSORS.split(",")],
        "targeted_k6": [parse_component(v) for v in TARGETED.split(",")],
    }

    results = {"model": args.model, "n_mmlu": len(items), "seed": args.seed, "conditions": {}}
    for name, components in conditions.items():
        snapshots = snapshot_weights(model, components) if components else {}
        try:
            edits = [orthogonalize_component(model, c, direction) for c in components]
            acc, per_item = mmlu_accuracy(model, tokenizer, items, args.batch_size,
                                          args.max_tokens, device)
            wiki = wikitext_nll(model, tokenizer, device)
            results["conditions"][name] = {
                "components": [c["name"] for c in components],
                "edits": edits,
                "mmlu_accuracy": acc,
                "mmlu_per_item": per_item,
                "wikitext": wiki,
            }
            log.info("%s: MMLU %.4f | ppl %.4f", name, acc, wiki["perplexity"])
        finally:
            if snapshots:
                restore_weights(snapshots)

    base = results["conditions"]["baseline"]
    for name, cond in results["conditions"].items():
        if name == "baseline":
            continue
        # paired McNemar-style bootstrap on per-item correctness deltas
        b = np.asarray(base["mmlu_per_item"], dtype=float)
        e = np.asarray(cond["mmlu_per_item"], dtype=float)
        rng = np.random.default_rng(args.seed)
        deltas = []
        for _ in range(20000):
            idx = rng.integers(0, len(b), len(b))
            deltas.append(float((e[idx] - b[idx]).mean()))
        cond["mmlu_delta_vs_baseline"] = {
            "mean": float((e - b).mean()),
            "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        }
        cond["ppl_ratio_vs_baseline"] = cond["wikitext"]["perplexity"] / base["wikitext"]["perplexity"]

    with open(out / "capability_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info("wrote %s", out / "capability_eval.json")


if __name__ == "__main__":
    main()
