"""Causal mediation/rescue tests for the purified costly-helping circuit.

For each proposed edge, mean-ablate the source component on Cell M forced-choice
prompts, then restore the target component's baseline activation. Behavioral
recovery estimates how much of the source effect is mediated by that target.
"""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch

try:
    from src.activation_patching import build_choice_prompt, compute_means, tokenize_batch
except ModuleNotFoundError:  # Direct execution adds src/ rather than the repo root.
    from activation_patching import build_choice_prompt, compute_means, tokenize_batch


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("path-mediation")

DEFAULT_EDGES = (
    "L15H15->L19MLP,L17H13->L19MLP,L18H13->L19MLP,"
    "L19H12->L19MLP,L18H3->L19MLP,L19MLP->L20H15"
)


def parse_component(value):
    layer = int(value[1:value.index("H")]) if "H" in value else int(value[1:value.index("MLP")])
    head = int(value.split("H")[1]) if "H" in value else None
    return layer, head


def parse_edges(value):
    return [
        {"source_name": source, "source": parse_component(source),
         "target_name": target, "target": parse_component(target)}
        for source, target in (edge.split("->") for edge in value.split(","))
    ]


def mean_ablation_hook(component, means_z, means_mlp):
    layer, head = component
    if head is None:
        mean = means_mlp[layer]

        def hook(value, _hook):
            value[:] = mean.to(value.dtype)
            return value

        return f"blocks.{layer}.hook_mlp_out", hook
    mean = means_z[layer][head]

    def hook(value, _hook):
        value[:, :, head, :] = mean.to(value.dtype)
        return value

    return f"blocks.{layer}.attn.hook_z", hook


def restore_hook(component, baseline):
    layer, head = component
    if head is None:
        def hook(value, _hook):
            value[:] = baseline.to(value.dtype)
            return value

        return f"blocks.{layer}.hook_mlp_out", hook

    def hook(value, _hook):
        value[:, :, head, :] = baseline[:, :, head, :].to(value.dtype)
        return value

    return f"blocks.{layer}.attn.hook_z", hook


def cache_name(component):
    layer, head = component
    return f"blocks.{layer}.hook_mlp_out" if head is None else f"blocks.{layer}.attn.hook_z"


def choice_scores(logits, lengths, flips, tok_a, tok_b):
    scores = []
    for row, (length, flip) in enumerate(zip(lengths.tolist(), flips)):
        la = float(logits[row, length - 1, tok_a])
        lb = float(logits[row, length - 1, tok_b])
        scores.append((lb - la) if flip else (la - lb))
    return scores


def bootstrap_mean_ci(values, seed, n_bootstrap=5000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


@torch.no_grad()
def run_edge(model, prompts, flips, edge, means_z, means_mlp,
             tok_a, tok_b, batch_size, seed):
    baseline_scores, ablated_scores, rescued_scores = [], [], []
    target_name = cache_name(edge["target"])
    source_hook = mean_ablation_hook(edge["source"], means_z, means_mlp)

    for start in range(0, len(prompts), batch_size):
        chunk_prompts = prompts[start:start + batch_size]
        chunk_flips = flips[start:start + batch_size]
        tokens, mask, lengths = tokenize_batch(model, chunk_prompts)
        baseline_logits, cache = model.run_with_cache(
            tokens,
            attention_mask=mask,
            names_filter=target_name,
            return_type="logits",
        )
        ablated_logits = model.run_with_hooks(
            tokens,
            attention_mask=mask,
            fwd_hooks=[source_hook],
            return_type="logits",
        )
        target_hook = restore_hook(edge["target"], cache[target_name])
        rescued_logits = model.run_with_hooks(
            tokens,
            attention_mask=mask,
            fwd_hooks=[source_hook, target_hook],
            return_type="logits",
        )
        baseline_scores.extend(choice_scores(
            baseline_logits, lengths, chunk_flips, tok_a, tok_b
        ))
        ablated_scores.extend(choice_scores(
            ablated_logits, lengths, chunk_flips, tok_a, tok_b
        ))
        rescued_scores.extend(choice_scores(
            rescued_logits, lengths, chunk_flips, tok_a, tok_b
        ))

    baseline = np.asarray(baseline_scores)
    ablated = np.asarray(ablated_scores)
    rescued = np.asarray(rescued_scores)
    source_effect = ablated - baseline
    rescue_effect = rescued - ablated
    denominator = float(baseline.mean() - ablated.mean())
    rescue_fraction = float(rescue_effect.mean() / denominator) if abs(denominator) > 1e-8 else None
    return {
        "source": edge["source_name"],
        "target": edge["target_name"],
        "baseline_behavior": float(baseline.mean()),
        "ablated_behavior": float(ablated.mean()),
        "rescued_behavior": float(rescued.mean()),
        "source_effect": float(source_effect.mean()),
        "source_effect_95ci": bootstrap_mean_ci(source_effect, seed),
        "rescue_effect": float(rescue_effect.mean()),
        "rescue_effect_95ci": bootstrap_mean_ci(rescue_effect, seed + 1),
        "rescue_fraction": rescue_fraction,
        "per_example": {
            "baseline": baseline.tolist(),
            "ablated": ablated.tolist(),
            "rescued": rescued.tolist(),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2-9b-it")
    parser.add_argument("--pairs", default="data/contrastive_pairs/v2_1/M_templated.jsonl")
    parser.add_argument("--edges", default=DEFAULT_EDGES)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/action_path_mediation_gemma2_9b_it")
    args = parser.parse_args()

    from transformer_lens import HookedTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained_no_processing(
        args.model, dtype=torch.bfloat16, device=device
    )
    model.eval()
    pairs = [json.loads(line) for line in Path(args.pairs).read_text().splitlines()]
    rng = random.Random(args.seed)
    flips = [rng.random() < 0.5 for _ in pairs]
    prompts = [build_choice_prompt(pair, flip) for pair, flip in zip(pairs, flips)]
    tok_a = model.to_single_token("A")
    tok_b = model.to_single_token("B")

    all_texts = [text for pair in pairs for text in (pair["pos_text"], pair["neg_text"])]
    means_z, means_mlp = compute_means(model, all_texts, 21, args.batch_size)
    results = []
    for index, edge in enumerate(parse_edges(args.edges)):
        log.info("edge %s -> %s", edge["source_name"], edge["target_name"])
        results.append(run_edge(
            model, prompts, flips, edge, means_z, means_mlp,
            tok_a, tok_b, args.batch_size, args.seed + 10 * index,
        ))

    summary = {
        "model": args.model,
        "n_pairs": len(pairs),
        "intervention": "mean-ablate source, restore target baseline activation",
        "results": results,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log.info("wrote %s", out / "summary.json")


if __name__ == "__main__":
    main()
