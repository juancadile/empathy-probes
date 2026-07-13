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

Integrity Repair A (2026-07-13): the audited artifact (`e17b.json`) was not
self-contained — the reported "88% of baseline margin" numbers could not be
reconstructed from the file. The result now persists direction path/SHA-256,
cell paths/hashes/families, exact baseline per-pair scores, per-pair edited
scores and deltas, the A/B option order (flip signs) for every scored pass,
model/block/seed/revision/dtype/tokenizer settings, and an environment
provenance block. Existing summary keys (mean/ci95) are unchanged for
backward compatibility. Component sets must be resolved explicitly (the old
module-level Llama defaults silently mismatched non-Llama runs).

Usage (Spark, `empathy` env):
  python -u src/e17b_null_audit.py \
    --direction results/controlled_directions_llama31_8b_it/direction_M_grouped_block15.npy \
    --component-set llama31_8b_it_grouped_2026-07-12 \
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
    from src.component_sets import ComponentSetError, resolve_component_sets
    from src.utils.run_provenance import (
        collect_run_provenance, resolve_model_and_tokenizer, sha256_file,
    )
except ModuleNotFoundError:
    from weight_orthogonalization import (
        load_pairs, orthogonalize_component, parse_component,
        restore_weights, snapshot_weights, tokenize,
    )
    from activation_patching import build_choice_prompt
    from component_sets import ComponentSetError, resolve_component_sets
    from utils.run_provenance import (
        collect_run_provenance, resolve_model_and_tokenizer, sha256_file,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("e17b")

EVAL_SETS = {
    "M_confirm": "data/contrastive_pairs/v2_1/M_confirm_templated.jsonl",
    "T_confirm": "data/contrastive_pairs/v2_1/T_confirm_templated.jsonl",
}


@torch.no_grad()
def choice_scores_fmt(model, tokenizer, pairs, seed, batch_size, max_tokens,
                      device, chat=False, return_flips=False):
    """choice_scores with optional chat-template wrapping.

    ``flips[i]`` is the A/B option order for pair ``i`` (True = pos_text was
    option B); the returned score is already flip-corrected (positive = model
    prefers pos_text). With ``return_flips`` the option order is returned for
    persistence so scores can be independently recomputed.
    """
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
    if return_flips:
        return scores, flips
    return scores


class DirectionAblator:
    """Removes fraction*projection onto `direction` from a block's output."""

    def __init__(self, model, block, direction, fraction=1.0):
        self.handle = None
        self.model, self.block = model, block
        self.d = direction  # unit, float32, on device
        self.f = fraction

    def __enter__(self):
        def hook(_m, _i, output):
            h = output[0] if isinstance(output, tuple) else output
            proj = (h.float() @ self.d)[..., None] * self.d  # (b,s,d)
            h_new = (h.float() - self.f * proj).to(h.dtype)
            if isinstance(output, tuple):
                return (h_new,) + tuple(output[1:])
            return h_new
        self.handle = self.model.model.layers[self.block].register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        self.handle.remove()


def clustered_delta(cond_scores, base_scores, families, seed=0, n_boot=5000,
                    detail=False):
    d = np.asarray(cond_scores) - np.asarray(base_scores)
    fams = np.asarray(families)
    unique = sorted(set(fams))
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        pick = rng.choice(len(unique), len(unique), replace=True)
        means.append(np.concatenate([d[fams == unique[i]] for i in pick]).mean())
    record = {"mean": float(d.mean()),
              "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}
    if detail:
        record["per_pair_delta"] = [float(x) for x in d]
        record["per_pair_scores"] = [float(x) for x in cond_scores]
    return record


def cell_provenance(pair_sets, fams, eval_sets=None):
    """Self-contained cell descriptors: path, SHA-256, rows, families."""
    prov = {}
    for name, path in (eval_sets or EVAL_SETS).items():
        prov[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "n_rows": len(pair_sets[name]),
            "families": fams[name],
        }
    return prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--revision", default=None,
                    help="explicit HF model revision (accepted reruns must pin this)")
    ap.add_argument("--tokenizer-revision", default=None,
                    help="explicit tokenizer revision (defaults to --revision)")
    ap.add_argument("--direction", required=True)
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--ablate-blocks", type=int, nargs="+", default=[15, 20, 24, 28])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ablate-fractions", type=float, nargs="+", default=None,
                    help="test 3: fractional ablation curve at --block (+3 random-dir controls at f=1)")
    ap.add_argument("--skip-test1", action="store_true",
                    help="only run activation ablation (e.g. cross-model normalization)")
    ap.add_argument("--component-set", default=None,
                    help="versioned set-of-record key from src/component_sets.py")
    ap.add_argument("--writers", default=None,
                    help="explicit spec; overrides --component-set")
    ap.add_argument("--suppressors", default=None,
                    help="explicit spec; overrides --component-set")
    ap.add_argument("--random-components", default=None,
                    help="explicit spec; overrides --component-set")
    ap.add_argument("--out", default="results/e17b_null_audit_llama31_8b_it")
    args = ap.parse_args()

    try:
        resolution = resolve_component_sets(
            roles=("positive_writers", "suppressors", "random"),
            explicit={"positive_writers": args.writers,
                      "suppressors": args.suppressors,
                      "random": args.random_components},
            set_key=args.component_set,
            model=args.model,
        )
    except ComponentSetError as exc:
        ap.error(str(exc))
    log.info("component sets resolved: %s", resolution)
    writers_spec = resolution["sets"]["positive_writers"]
    suppressors_spec = resolution["sets"]["suppressors"]
    random_spec = resolution["sets"]["random"]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.tokenizer_revision or args.revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager",
        revision=args.revision).to(device)
    model.eval()
    direction = torch.tensor(np.load(args.direction), dtype=torch.float32, device=device)
    direction /= torch.linalg.vector_norm(direction)

    pair_sets = {n: load_pairs(p) for n, p in EVAL_SETS.items()}
    fams = {n: [p["scenario_id"] for p in pairs] for n, pairs in pair_sets.items()}

    conditions = {
        "positive_writers_k2": writers_spec,
        "suppressors_k4": suppressors_spec,
        "targeted_k6": writers_spec + "," + suppressors_spec,
        "random_k6": random_spec,
    }
    results = {
        "model": args.model,
        "block": args.block,
        "component_sets": resolution,
        "conditions_spec": dict(conditions),
        "config": {
            "revision": args.revision,
            "tokenizer_revision": args.tokenizer_revision or args.revision,
            "dtype": "bfloat16",
            "attn_implementation": "eager",
            "seed": args.seed,
            "batch_size": args.batch_size,
            "max_tokens": args.max_tokens,
            "ablate_blocks": args.ablate_blocks,
            "ablate_fractions": args.ablate_fractions,
            "tokenizer": {
                "name_or_path": tokenizer.name_or_path,
                "padding_side": tokenizer.padding_side,
                "pad_token": tokenizer.pad_token,
                "truncation": True,
            },
            "prompt_format": {
                "raw": "activation_patching.build_choice_prompt (A/B forced "
                       "choice, final-token logit diff, flip-corrected)",
                "chat": "tokenizer.apply_chat_template(user=prompt, "
                        "add_generation_prompt=True)",
            },
        },
        "direction_file": {"path": args.direction,
                           "sha256": sha256_file(args.direction)},
        "cells": cell_provenance(pair_sets, fams),
        "hf_revisions": resolve_model_and_tokenizer(
            args.model, revision=args.revision or "main",
            tokenizer_revision=args.tokenizer_revision or args.revision or "main",
        ),
        "provenance": collect_run_provenance(),
    }

    # ---- test 1: chat-template sensitivity ----
    for fmt, chat in ([] if args.skip_test1 else [("raw", False), ("chat", True)]):
        base, base_flips = {}, {}
        for n, pairs in pair_sets.items():
            base[n], base_flips[n] = choice_scores_fmt(
                model, tokenizer, pairs, args.seed, args.batch_size,
                args.max_tokens, device, chat, return_flips=True)
        block_res = {"baseline": {n: {"mean": float(np.mean(v)),
                                      "per_pair_scores": [float(x) for x in v],
                                      "option_flips": base_flips[n]}
                                  for n, v in base.items()}}
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
                    deltas[n] = clustered_delta(s, base[n], fams[n],
                                                seed=args.seed, detail=True)
            finally:
                restore_weights(snap)
            block_res[cond] = deltas
            log.info("[%s] %s: dM %+0.4f CI %s | dT %+0.4f", fmt, cond,
                     deltas["M_confirm"]["mean"], deltas["M_confirm"]["ci95"],
                     deltas["T_confirm"]["mean"])
        results[f"test1_{fmt}"] = block_res

    # ---- test 2: activation-level direction ablation (unedited weights) ----
    base_raw, base_raw_flips = {}, {}
    for n, pairs in pair_sets.items():
        base_raw[n], base_raw_flips[n] = choice_scores_fmt(
            model, tokenizer, pairs, args.seed, args.batch_size,
            args.max_tokens, device, False, return_flips=True)
    results["baseline_raw"] = {
        n: {"mean": float(np.mean(v)),
            "per_pair_scores": [float(x) for x in v],
            "option_flips": base_raw_flips[n]}
        for n, v in base_raw.items()}
    test2 = {}
    for blk in args.ablate_blocks:
        with DirectionAblator(model, blk, direction):
            deltas = {}
            for n, pairs in pair_sets.items():
                s = choice_scores_fmt(model, tokenizer, pairs, args.seed,
                                      args.batch_size, args.max_tokens, device, False)
                deltas[n] = clustered_delta(s, base_raw[n], fams[n],
                                            seed=args.seed, detail=True)
        test2[f"block_{blk}"] = deltas
        log.info("[ablate B%d] dM %+0.4f CI %s | dT %+0.4f", blk,
                 deltas["M_confirm"]["mean"], deltas["M_confirm"]["ci95"],
                 deltas["T_confirm"]["mean"])
    results["test2_activation_ablation"] = test2

    # ---- test 3: fractional ablation dose curve + random-direction controls ----
    if args.ablate_fractions:
        test3 = {
            "notes": (
                "deltas are vs baseline_raw (same file); relative effect for "
                "fraction f on cell c = test3[fraction_f][c][mean] / "
                "baseline_raw[c][mean]"
            ),
        }
        for f in args.ablate_fractions:
            with DirectionAblator(model, args.block, direction, fraction=f):
                deltas = {}
                for n, pairs in pair_sets.items():
                    s = choice_scores_fmt(model, tokenizer, pairs, args.seed,
                                          args.batch_size, args.max_tokens, device, False)
                    deltas[n] = clustered_delta(s, base_raw[n], fams[n],
                                                seed=args.seed, detail=True)
            test3[f"fraction_{f}"] = deltas
            log.info("[frac %.2f] dM %+0.4f CI %s | dT %+0.4f", f,
                     deltas["M_confirm"]["mean"], deltas["M_confirm"]["ci95"],
                     deltas["T_confirm"]["mean"])
        for s_i in range(3):
            gen = torch.Generator(device="cpu").manual_seed(args.seed * 100 + s_i)
            rd = torch.randn(direction.shape[0], generator=gen).to(device)
            rd /= torch.linalg.vector_norm(rd)
            with DirectionAblator(model, args.block, rd, fraction=1.0):
                s = choice_scores_fmt(model, tokenizer, pair_sets["M_confirm"], args.seed,
                                      args.batch_size, args.max_tokens, device, False)
            d = clustered_delta(s, base_raw["M_confirm"], fams["M_confirm"],
                                seed=args.seed, detail=True)
            d["direction_seed"] = args.seed * 100 + s_i
            test3[f"random_dir_{s_i}"] = d
            log.info("[random-dir %d] dM %+0.4f CI %s", s_i, d["mean"], d["ci95"])
        results["test3_fractional_ablation"] = test3

    (out / "e17b.json").write_text(json.dumps(results, indent=2))
    log.info("wrote %s", out / "e17b.json")


if __name__ == "__main__":
    main()
