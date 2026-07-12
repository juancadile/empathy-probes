"""E17b: audit the Llama stage-3 null (codex-ranked artifact mechanisms 1-2).

Test 1 — chat-template sensitivity (pre-registered, amendment 6): rerun the
choice evaluation for {baseline, writers_k2, suppressors_k4, targeted_k6,
random_k6} on confirmatory M/T with prompts wrapped in the model's chat
template (add_generation_prompt=True). The A/B readout stays the final-token
logit difference.

Test 2 — activation-level direction ablation: with UNEDITED weights, project
the frozen direction out of the residual stream at a given block (every
position) during choice scoring, at block B* and late blocks. If activation
removal moves choice while weight edits did not, rank-1 component edits were
insufficient/bypassed; if it also does nothing, the direction is decodable but
not causally read (NULL-LIKELY-REAL).

Usage (Spark, `empathy` env):
  python -u src/e17b_null_audit.py \
    --direction results/controlled_directions_llama31_8b_it/direction_M_grouped_block15.npy \
    --block 15 --ablate-blocks 15 20 24 28 --out results/e17b_null_audit_llama31_8b_it
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
        load_pairs, orthogonalize_component, parse_component,
        restore_weights, snapshot_weights, tokenize,
    )
    from src.activation_patching import build_choice_prompt
except ModuleNotFoundError:
    from weight_orthogonalization import (
        load_pairs, orthogonalize_component, parse_component,
        restore_weights, snapshot_weights, tokenize,
    )
    from activation_patching import build_choice_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("e17b")

POSITIVE_WRITERS = "L15MLP,L12H4"
SUPPRESSORS = "L15H6,L14H27,L12H20,L11MLP"
RANDOM = "L1MLP,L12H15,L15H17,L14H6,L12H31,L2MLP"

EVAL_SETS = {
    "M_confirm": "data/contrastive_pairs/v2_1/M_confirm_templated.jsonl",
    "T_confirm": "data/contrastive_pairs/v2_1/T_confirm_templated.jsonl",
}


@torch.no_grad()
def choice_scores_fmt(model, tokenizer, pairs, seed, batch_size, max_tokens,
                      device, chat=False):
    """choice_scores with optional chat-template wrapping."""
    rng = random.Random(seed)
    flips = [rng.random() < 0.5 for _ in pairs]
    prompts = [build_choice_prompt(pair, flip) for pair, flip in zip(pairs, flips)]
    if chat:
        prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True, tokenize=False) for p in prompts]
    tok_a = tokenizer.encode("A", add_special_tokens=False)
    tok_b = tokenizer.encode("B", add_special_tokens=False)
    assert len(tok_a) == 1 and len(tok_b) == 1
    scores = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        encoded, _ = tokenize(tokenizer, chunk, max_tokens)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        logits = model(**encoded, use_cache=False).logits
        lengths = encoded["attention_mask"].sum(1)
        for row, (length, flip) in enumerate(zip(lengths.tolist(),
                                                 flips[start:start + batch_size])):
            la = float(logits[row, length - 1, tok_a[0]])
            lb = float(logits[row, length - 1, tok_b[0]])
            scores.append((lb - la) if flip else (la - lb))
    return scores


class DirectionAblator:
    """Removes the direction component from a block's output at every position."""

    def __init__(self, model, block, direction):
        self.handle = None
        self.model, self.block = model, block
        self.d = direction  # unit, float32, on device

    def __enter__(self):
        def hook(_m, _i, output):
            h = output[0] if isinstance(output, tuple) else output
            proj = (h.float() @ self.d)[..., None] * self.d  # (b,s,d)
            h_new = (h.float() - proj).to(h.dtype)
            if isinstance(output, tuple):
                return (h_new,) + tuple(output[1:])
            return h_new
        self.handle = self.model.model.layers[self.block].register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        self.handle.remove()


def clustered_delta(cond_scores, base_scores, families, seed=0, n_boot=5000):
    d = np.asarray(cond_scores) - np.asarray(base_scores)
    fams = np.asarray(families)
    unique = sorted(set(fams))
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        pick = rng.choice(len(unique), len(unique), replace=True)
        means.append(np.concatenate([d[fams == unique[i]] for i in pick]).mean())
    return {"mean": float(d.mean()),
            "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--direction", required=True)
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--ablate-blocks", type=int, nargs="+", default=[15, 20, 24, 28])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-test1", action="store_true",
                    help="only run activation ablation (e.g. cross-model normalization)")
    ap.add_argument("--out", default="results/e17b_null_audit_llama31_8b_it")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager").to(device)
    model.eval()
    direction = torch.tensor(np.load(args.direction), dtype=torch.float32, device=device)
    direction /= torch.linalg.vector_norm(direction)

    pair_sets = {n: load_pairs(p) for n, p in EVAL_SETS.items()}
    fams = {n: [p["scenario_id"] for p in pairs] for n, pairs in pair_sets.items()}

    conditions = {
        "positive_writers_k2": POSITIVE_WRITERS,
        "suppressors_k4": SUPPRESSORS,
        "targeted_k6": POSITIVE_WRITERS + "," + SUPPRESSORS,
        "random_k6": RANDOM,
    }
    results = {"model": args.model, "block": args.block}

    # ---- test 1: chat-template sensitivity ----
    for fmt, chat in ([] if args.skip_test1 else [("raw", False), ("chat", True)]):
        base = {n: choice_scores_fmt(model, tokenizer, pairs, args.seed,
                                     args.batch_size, args.max_tokens, device, chat)
                for n, pairs in pair_sets.items()}
        block_res = {"baseline": {n: float(np.mean(v)) for n, v in base.items()}}
        for cond, spec in conditions.items():
            comps = [parse_component(v) for v in spec.split(",")]
            snap = snapshot_weights(model, comps)
            try:
                for c in comps:
                    orthogonalize_component(model, c, direction)
                deltas = {}
                for n, pairs in pair_sets.items():
                    s = choice_scores_fmt(model, tokenizer, pairs, args.seed,
                                          args.batch_size, args.max_tokens, device, chat)
                    deltas[n] = clustered_delta(s, base[n], fams[n], seed=args.seed)
            finally:
                restore_weights(snap)
            block_res[cond] = deltas
            log.info("[%s] %s: dM %+0.4f CI %s | dT %+0.4f", fmt, cond,
                     deltas["M_confirm"]["mean"], deltas["M_confirm"]["ci95"],
                     deltas["T_confirm"]["mean"])
        results[f"test1_{fmt}"] = block_res

    # ---- test 2: activation-level direction ablation (unedited weights) ----
    base_raw = {n: choice_scores_fmt(model, tokenizer, pairs, args.seed,
                                     args.batch_size, args.max_tokens, device, False)
                for n, pairs in pair_sets.items()}
    test2 = {}
    for blk in args.ablate_blocks:
        with DirectionAblator(model, blk, direction):
            deltas = {}
            for n, pairs in pair_sets.items():
                s = choice_scores_fmt(model, tokenizer, pairs, args.seed,
                                      args.batch_size, args.max_tokens, device, False)
                deltas[n] = clustered_delta(s, base_raw[n], fams[n], seed=args.seed)
        test2[f"block_{blk}"] = deltas
        log.info("[ablate B%d] dM %+0.4f CI %s | dT %+0.4f", blk,
                 deltas["M_confirm"]["mean"], deltas["M_confirm"]["ci95"],
                 deltas["T_confirm"]["mean"])
    results["test2_activation_ablation"] = test2

    (out / "e17b.json").write_text(json.dumps(results, indent=2))
    log.info("wrote %s", out / "e17b.json")


if __name__ == "__main__":
    main()
