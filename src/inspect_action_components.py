"""Inspect attention and weight structure of purified action components."""

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("inspect-components")

DEFAULT_HEADS = "17:13,18:13,19:12,20:15,15:15"
STOPWORDS = {
    "<bos>", "a", "am", "an", "and", "as", "at", "be", "before", "but",
    "by", "for", "from", "i", "in", "is", "it", "my", "now", "of", "on",
    "or", "that", "the", "this", "to", "will", "with",
}
HELP_STEMS = ("help", "assist", "pause", "support", "care", "water", "stop", "take")
TASK_STEMS = ("continue", "finish", "route", "deliver", "objective", "remain", "keep", "task")


def parse_heads(value):
    return [tuple(map(int, item.split(":"))) for item in value.split(",")]


def clean_token(tokenizer, token_id):
    text = tokenizer.decode([int(token_id)]).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text if re.search(r"[a-z0-9]", text) else None


def summarize_token_scores(scores, n_examples, limit=25):
    rows = [
        {"token": token, "attention_per_example": value / n_examples}
        for token, value in scores.items()
    ]
    return sorted(rows, key=lambda row: row["attention_per_example"], reverse=True)[:limit]


def token_differential(positive, negative, positive_n, negative_n, limit=25):
    tokens = set(positive) | set(negative)
    rows = [
        {
            "token": token,
            "positive_minus_negative_attention": (
                positive.get(token, 0) / positive_n
                - negative.get(token, 0) / negative_n
            ),
        }
        for token in tokens
    ]
    rows.sort(key=lambda row: abs(row["positive_minus_negative_attention"]), reverse=True)
    return rows[:limit]


@torch.no_grad()
def inspect_attention(model, pairs, heads, max_tokens):
    layers = sorted({layer for layer, _ in heads})
    names = {f"blocks.{layer}.attn.hook_pattern" for layer in layers}
    aggregates = {
        f"L{layer}H{head}": {
            side: {
                "n": 0,
                "prefix_mass": [],
                "tail_mass": [],
                "help_token_mass": [],
                "task_token_mass": [],
                "tokens": defaultdict(float),
                "content_tokens": defaultdict(float),
            }
            for side in ("positive", "negative")
        }
        for layer, head in heads
    }

    for pair_index, pair in enumerate(pairs):
        prefix = pair["shared_prefix"]
        for side, field in (("positive", "pos_text"), ("negative", "neg_text")):
            text = pair[field]
            encoded = model.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_tokens,
                return_offsets_mapping=True,
            )
            offsets = encoded.pop("offset_mapping")[0]
            tokens = encoded["input_ids"].to(model.cfg.device)
            attention_mask = encoded["attention_mask"].to(model.cfg.device)
            _, cache = model.run_with_cache(
                tokens,
                attention_mask=attention_mask,
                names_filter=lambda name: name in names,
                return_type=None,
            )
            tail = offsets[:, 1] > len(prefix)
            if not tail.any():
                continue
            prefix_keys = ~tail
            token_ids = tokens[0].cpu()
            for layer, head in heads:
                key = f"L{layer}H{head}"
                pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0, head].float().cpu()
                key_attention = pattern[tail].mean(0)
                record = aggregates[key][side]
                record["n"] += 1
                record["prefix_mass"].append(float(key_attention[prefix_keys].sum()))
                record["tail_mass"].append(float(key_attention[tail].sum()))
                help_mass, task_mass = 0.0, 0.0
                for position, score in enumerate(key_attention.tolist()):
                    token = clean_token(model.tokenizer, token_ids[position])
                    if token:
                        record["tokens"][token] += score
                        if token not in STOPWORDS and len(token) > 1:
                            record["content_tokens"][token] += score
                        if any(stem in token for stem in HELP_STEMS):
                            help_mass += score
                        if any(stem in token for stem in TASK_STEMS):
                            task_mass += score
                record["help_token_mass"].append(help_mass)
                record["task_token_mass"].append(task_mass)
        if (pair_index + 1) % 10 == 0:
            log.info("attention %d/%d pairs", pair_index + 1, len(pairs))

    summary = {}
    for key, sides in aggregates.items():
        summary[key] = {}
        for side, record in sides.items():
            summary[key][side] = {
                "n": record["n"],
                "mean_prefix_attention": float(np.mean(record["prefix_mass"])),
                "mean_tail_attention": float(np.mean(record["tail_mass"])),
                "mean_help_token_attention": float(np.mean(record["help_token_mass"])),
                "mean_task_token_attention": float(np.mean(record["task_token_mass"])),
                "top_source_tokens": summarize_token_scores(
                    record["tokens"], record["n"]
                ),
                "top_content_source_tokens": summarize_token_scores(
                    record["content_tokens"], record["n"]
                ),
            }
        summary[key]["positive_minus_negative_prefix_attention"] = (
            summary[key]["positive"]["mean_prefix_attention"]
            - summary[key]["negative"]["mean_prefix_attention"]
        )
        summary[key]["content_token_differential"] = token_differential(
            sides["positive"]["content_tokens"],
            sides["negative"]["content_tokens"],
            sides["positive"]["n"],
            sides["negative"]["n"],
        )
    return summary


def percentile(values, value):
    values = np.asarray(values)
    return float(100 * np.mean(values <= value))


@torch.no_grad()
def inspect_weights(model, direction, heads, mlp_layer):
    direction = torch.tensor(direction, dtype=torch.float32, device=model.cfg.device)
    selected_layers = {layer for layer, _ in heads}
    head_rows = []
    for layer in range(21):
        for head in range(model.cfg.n_heads):
            w_o = model.W_O[layer, head].float()  # (d_head, d_model)
            projection_fraction = None
            if layer in selected_layers:
                q = torch.linalg.qr(w_o.T, mode="reduced").Q
                projection_fraction = float((q.T @ direction).pow(2).sum())
            normalized_write_gain = float(torch.linalg.vector_norm(w_o @ direction) / torch.linalg.matrix_norm(w_o))
            head_rows.append({
                "component": f"L{layer}H{head}",
                "layer": layer,
                "head": head,
                "projection_fraction": projection_fraction,
                "normalized_write_gain": normalized_write_gain,
            })
    fractions = [
        row["projection_fraction"]
        for row in head_rows
        if row["projection_fraction"] is not None
    ]
    gains = [row["normalized_write_gain"] for row in head_rows]
    selected = {}
    for layer, head in heads:
        row = next(r for r in head_rows if r["layer"] == layer and r["head"] == head)
        selected[row["component"]] = {
            **row,
            "projection_fraction_percentile": percentile(fractions, row["projection_fraction"]),
            "write_gain_percentile": percentile(gains, row["normalized_write_gain"]),
        }

    w_out = model.W_out[mlp_layer].float()  # (d_mlp, d_model)
    output_alignment = w_out @ direction
    abs_alignment = output_alignment.abs()
    top = torch.topk(abs_alignment, k=min(50, len(abs_alignment)))
    total = float(abs_alignment.sum())
    mlp = {
        "component": f"L{mlp_layer}MLP",
        "top10_abs_alignment_share": float(top.values[:10].sum()) / total,
        "top50_abs_alignment_share": float(top.values.sum()) / total,
        "top_output_channels": [
            {
                "channel": int(index),
                "signed_output_alignment": float(output_alignment[index]),
                "abs_alignment": float(value),
            }
            for value, index in zip(top.values, top.indices)
        ],
    }
    return {"selected_heads": selected, "mlp": mlp}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2-9b-it")
    parser.add_argument("--pairs", default="data/contrastive_pairs/v2_1/M_templated.jsonl")
    parser.add_argument("--direction", required=True)
    parser.add_argument("--heads", default=DEFAULT_HEADS)
    parser.add_argument("--mlp-layer", type=int, default=19)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--out", default="results/action_component_inspection_gemma2_9b_it")
    args = parser.parse_args()

    from transformer_lens import HookedTransformer

    heads = parse_heads(args.heads)
    pairs = [json.loads(line) for line in Path(args.pairs).read_text().splitlines()]
    direction = np.load(args.direction)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained_no_processing(
        args.model, dtype=torch.bfloat16, device=device
    )
    model.eval()
    summary = {
        "model": args.model,
        "direction": args.direction,
        "heads": [f"L{layer}H{head}" for layer, head in heads],
        "n_pairs": len(pairs),
        "attention": inspect_attention(model, pairs, heads, args.max_tokens),
        "weights": inspect_weights(model, direction, heads, args.mlp_layer),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log.info("wrote %s", out / "summary.json")


if __name__ == "__main__":
    main()
