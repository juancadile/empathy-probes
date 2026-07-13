"""Targeted weight orthogonalization for the purified costly-helping direction.

Integrity Repair A (2026-07-13): the module previously shipped the superseded
pre-correction component sets as silent CLI defaults, so an unqualified rerun
edited the wrong components. Component sets must now be named explicitly —
either via ``--component-set <versioned key>`` from ``src/component_sets.py``
or full literal ``--targeted/--random/--positive-writers/--suppressors``
specs — and the resolved sets are persisted in the result artifact.
"""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch

try:
    from src.activation_patching import build_choice_prompt
    from src.component_sets import ComponentSetError, resolve_component_sets
    from src.utils.run_provenance import (
        collect_run_provenance, resolve_model_and_tokenizer,
    )
except ModuleNotFoundError:
    from activation_patching import build_choice_prompt
    from component_sets import ComponentSetError, resolve_component_sets
    from utils.run_provenance import (
        collect_run_provenance, resolve_model_and_tokenizer,
    )


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("weight-edit")

NEUTRAL_PROMPTS = [
    "The capital city of France is",
    "Water freezes at a temperature of",
    "The opposite of increase is",
    "A triangle has three sides. A square has",
    "To make tea, the first practical step is to",
    "The Earth travels around the",
    "If a book costs five dollars, two books cost",
    "The primary purpose of a calendar is to track",
    "A healthy plant generally needs light and",
    "The color produced by mixing blue and yellow is",
    "When sorting numbers from smallest to largest, 2 comes before",
    "A person uses an umbrella primarily when it is",
]


def parse_component(value):
    if "MLP" in value:
        return {"name": value, "layer": int(value[1:value.index("MLP")]), "head": None}
    return {
        "name": value,
        "layer": int(value[1:value.index("H")]),
        "head": int(value.split("H")[1]),
    }


def load_pairs(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def tokenize(tokenizer, texts, max_tokens, offsets=False):
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_tokens,
        return_offsets_mapping=offsets,
    )
    offset_mapping = encoded.pop("offset_mapping", None)
    return encoded, offset_mapping


@torch.no_grad()
def choice_scores(model, tokenizer, pairs, seed, batch_size, max_tokens, device):
    rng = random.Random(seed)
    flips = [rng.random() < 0.5 for _ in pairs]
    prompts = [build_choice_prompt(pair, flip) for pair, flip in zip(pairs, flips)]
    tok_a = tokenizer.encode("A", add_special_tokens=False)
    tok_b = tokenizer.encode("B", add_special_tokens=False)
    if len(tok_a) != 1 or len(tok_b) != 1:
        raise ValueError("A/B must each tokenize to one token")
    scores = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        encoded, _ = tokenize(tokenizer, chunk, max_tokens)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        logits = model(**encoded, use_cache=False).logits
        lengths = encoded["attention_mask"].sum(1)
        for row, (length, flip) in enumerate(zip(lengths.tolist(), flips[start:start + batch_size])):
            la = float(logits[row, length - 1, tok_a[0]])
            lb = float(logits[row, length - 1, tok_b[0]])
            scores.append((lb - la) if flip else (la - lb))
    return scores


@torch.no_grad()
def decision_separation(model, tokenizer, pairs, direction, block, batch_size, max_tokens, device):
    texts, prefixes = [], []
    for pair in pairs:
        texts.extend([pair["pos_text"], pair["neg_text"]])
        prefixes.extend([pair["shared_prefix"], pair["shared_prefix"]])
    projections = []
    captured = {}

    def hook(_module, _inputs, output):
        captured["hidden"] = output[0] if isinstance(output, tuple) else output

    handle = model.model.layers[block].register_forward_hook(hook)
    try:
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            encoded, offsets = tokenize(tokenizer, chunk, max_tokens, offsets=True)
            inputs = {key: value.to(device) for key, value in encoded.items()}
            model(**inputs, use_cache=False)
            hidden = captured["hidden"].float()
            mask = inputs["attention_mask"].bool()
            for row, prefix in enumerate(prefixes[start:start + batch_size]):
                tail = (offsets[row, :, 1] > len(prefix)).to(device) & mask[row]
                if not tail.any():
                    tail = mask[row]
                pooled = hidden[row, tail].mean(0)
                projections.append(float(pooled @ direction))
    finally:
        handle.remove()
    positive = np.asarray(projections[0::2])
    negative = np.asarray(projections[1::2])
    return {
        "mean_paired_difference": float(np.mean(positive - negative)),
        "per_pair_difference": (positive - negative).tolist(),
    }


@torch.no_grad()
def final_logits(model, tokenizer, prompts, batch_size, max_tokens, device):
    rows = []
    for start in range(0, len(prompts), batch_size):
        encoded, _ = tokenize(tokenizer, prompts[start:start + batch_size], max_tokens)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        logits = model(**encoded, use_cache=False).logits.float()
        lengths = encoded["attention_mask"].sum(1)
        rows.extend(logits[row, length - 1].cpu() for row, length in enumerate(lengths.tolist()))
    return torch.stack(rows)


def neutral_drift(baseline_logits, edited_logits):
    baseline_logp = torch.log_softmax(baseline_logits, dim=-1)
    edited_logp = torch.log_softmax(edited_logits, dim=-1)
    baseline_p = baseline_logp.exp()
    kl = (baseline_p * (baseline_logp - edited_logp)).sum(-1)
    agreement = (baseline_logits.argmax(-1) == edited_logits.argmax(-1)).float()
    return {
        "mean_kl_from_baseline": float(kl.mean()),
        "top1_agreement": float(agreement.mean()),
        "per_prompt_kl": kl.tolist(),
    }


def effective_direction(model, component, direction):
    layer = model.model.layers[component["layer"]]
    if hasattr(layer, "post_feedforward_layernorm"):
        # Gemma-2 sandwich norms: the component output passes through a post-RMSNorm
        # before entering the residual, so removing `direction` from the residual
        # means removing the norm-rescaled preimage from the component output.
        norm = (
            layer.post_feedforward_layernorm
            if component["head"] is None
            else layer.post_attention_layernorm
        )
        effective = direction * (1.0 + norm.weight.float())
    else:
        # Llama-style pre-norm: component outputs write directly to the residual.
        effective = direction.clone()
    return effective / torch.linalg.vector_norm(effective)


def component_weight(model, component):
    layer = model.model.layers[component["layer"]]
    if component["head"] is None:
        return layer.mlp.down_proj.weight, None
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    start = component["head"] * head_dim
    return layer.self_attn.o_proj.weight, slice(start, start + head_dim)


@torch.no_grad()
def orthogonalize_component(model, component, direction):
    weight, columns = component_weight(model, component)
    effective = effective_direction(model, component, direction).to(weight.device)
    if columns is None:
        original = weight.float()
        alignment = effective @ original
        delta = effective[:, None] * alignment[None, :]
        relative = float(torch.linalg.matrix_norm(delta) / torch.linalg.matrix_norm(original))
        weight.sub_(delta.to(weight.dtype))
    else:
        original = weight[:, columns].float()
        alignment = effective @ original
        delta = effective[:, None] * alignment[None, :]
        relative = float(torch.linalg.matrix_norm(delta) / torch.linalg.matrix_norm(original))
        weight[:, columns].sub_(delta.to(weight.dtype))
    return {
        "component": component["name"],
        "relative_component_weight_change": relative,
        "removed_alignment_norm": float(torch.linalg.vector_norm(alignment)),
    }


def snapshot_weights(model, sequences):
    snapshots = {}
    for component in sequences:
        weight, _ = component_weight(model, component)
        key = (component["layer"], component["head"] is None)
        snapshots.setdefault(key, (weight, weight.detach().clone()))
    return snapshots


@torch.no_grad()
def restore_weights(snapshots):
    for weight, original in snapshots.values():
        weight.copy_(original)


@torch.no_grad()
def evaluate(model, tokenizer, m_pairs, t_pairs, direction, neutral_baseline,
             batch_size, max_tokens, seed, device, block=20):
    m_scores = choice_scores(model, tokenizer, m_pairs, seed, batch_size, max_tokens, device)
    t_scores = choice_scores(model, tokenizer, t_pairs, seed, batch_size, max_tokens, device)
    separation = decision_separation(
        model, tokenizer, m_pairs, direction, block, batch_size, max_tokens, device
    )
    neutral = final_logits(
        model, tokenizer, NEUTRAL_PROMPTS, batch_size, max_tokens, device
    )
    return {
        "helping_choice": {"mean": float(np.mean(m_scores)), "per_pair": m_scores},
        "task_choice": {"mean": float(np.mean(t_scores)), "per_pair": t_scores},
        "decision_projection": separation,
        "neutral_drift": neutral_drift(neutral_baseline, neutral),
    }


def run_sequence(name, sequence, model, tokenizer, m_pairs, t_pairs, direction,
                 neutral_baseline, batch_size, max_tokens, seed, device, block=20):
    snapshots = snapshot_weights(model, sequence)
    conditions, edits = [], []
    try:
        for index, component in enumerate(sequence, start=1):
            edits.append(orthogonalize_component(model, component, direction))
            log.info("%s k=%d: %s", name, index, component["name"])
            conditions.append({
                "k": index,
                "components": [item["name"] for item in sequence[:index]],
                "edits": list(edits),
                "metrics": evaluate(
                    model, tokenizer, m_pairs, t_pairs, direction, neutral_baseline,
                    batch_size, max_tokens, seed, device, block=block,
                ),
            })
    finally:
        restore_weights(snapshots)
    return conditions


def run_individuals(name, sequence, model, tokenizer, m_pairs, t_pairs, direction,
                    neutral_baseline, batch_size, max_tokens, seed, device, block=20):
    conditions = []
    for index, component in enumerate(sequence, start=1):
        snapshots = snapshot_weights(model, [component])
        try:
            edit = orthogonalize_component(model, component, direction)
            log.info("%s: %s", name, component["name"])
            conditions.append({
                "component": component["name"],
                "edit": edit,
                "metrics": evaluate(
                    model, tokenizer, m_pairs, t_pairs, direction, neutral_baseline,
                    batch_size, max_tokens, seed, device, block=block,
                ),
            })
        finally:
            restore_weights(snapshots)
    return conditions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2-9b-it")
    parser.add_argument("--revision", default=None,
                        help="explicit HF model revision (accepted reruns must pin this)")
    parser.add_argument("--tokenizer-revision", default=None,
                        help="explicit tokenizer revision (defaults to --revision)")
    parser.add_argument("--direction", required=True)
    parser.add_argument("--component-set", default=None,
                        help="versioned set-of-record key from src/component_sets.py")
    parser.add_argument("--targeted", default=None,
                        help="explicit spec; overrides the named component set")
    parser.add_argument("--random", default=None,
                        help="explicit spec; overrides the named component set")
    parser.add_argument("--positive-writers", default=None,
                        help="explicit spec; overrides the named component set")
    parser.add_argument("--suppressors", default=None,
                        help="explicit spec; overrides the named component set")
    parser.add_argument("--m-pairs", default="data/contrastive_pairs/v2_1/M_templated.jsonl")
    parser.add_argument("--t-pairs", default="data/contrastive_pairs/v2_1/T_templated.jsonl")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block", type=int, default=20,
                        help="readout block for decision_separation (B* for the model)")
    parser.add_argument("--out", default="results/weight_orthogonalization_gemma2_9b_it")
    args = parser.parse_args()

    try:
        resolution = resolve_component_sets(
            roles=("targeted", "random", "positive_writers", "suppressors"),
            explicit={
                "targeted": args.targeted,
                "random": args.random,
                "positive_writers": args.positive_writers,
                "suppressors": args.suppressors,
            },
            set_key=args.component_set,
            model=args.model,
        )
    except ComponentSetError as exc:
        parser.error(str(exc))
    sets = resolution["sets"]
    log.info("component sets resolved: %s", resolution)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.tokenizer_revision or args.revision
    )
    if tokenizer.pad_token is None:  # Llama-3.1 ships without one
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager",
        revision=args.revision,
    ).to(device)
    model.eval()
    direction = torch.tensor(np.load(args.direction), dtype=torch.float32, device=device)
    direction /= torch.linalg.vector_norm(direction)
    m_pairs, t_pairs = load_pairs(args.m_pairs), load_pairs(args.t_pairs)
    neutral_baseline = final_logits(
        model, tokenizer, NEUTRAL_PROMPTS, args.batch_size, args.max_tokens, device
    )
    baseline = evaluate(
        model, tokenizer, m_pairs, t_pairs, direction, neutral_baseline,
        args.batch_size, args.max_tokens, args.seed, device, block=args.block,
    )
    targeted = [parse_component(value) for value in sets["targeted"].split(",")]
    random_components = [parse_component(value) for value in sets["random"].split(",")]
    positive_writers = [parse_component(value) for value in sets["positive_writers"].split(",")]
    suppressors = [parse_component(value) for value in sets["suppressors"].split(",")]
    summary = {
        "model": args.model,
        "revision": args.revision,
        "tokenizer_revision": args.tokenizer_revision or args.revision,
        "direction": args.direction,
        "component_sets": resolution,
        "edit": "rank-1 removal accounting for Gemma post-component RMSNorm",
        "hf_revisions": resolve_model_and_tokenizer(
            args.model, revision=args.revision or "main",
            tokenizer_revision=args.tokenizer_revision or args.revision or "main",
        ),
        "provenance": collect_run_provenance(
            files={"direction": args.direction, "m_pairs": args.m_pairs,
                   "t_pairs": args.t_pairs},
        ),
        "baseline": baseline,
        "targeted": run_sequence(
            "targeted", targeted, model, tokenizer, m_pairs, t_pairs, direction,
            neutral_baseline, args.batch_size, args.max_tokens, args.seed, device,
            block=args.block,
        ),
        "random": run_sequence(
            "random", random_components, model, tokenizer, m_pairs, t_pairs, direction,
            neutral_baseline, args.batch_size, args.max_tokens, args.seed, device,
            block=args.block,
        ),
        "targeted_individual": run_individuals(
            "targeted_individual", targeted, model, tokenizer, m_pairs, t_pairs,
            direction, neutral_baseline, args.batch_size, args.max_tokens,
            args.seed, device, block=args.block,
        ),
        "positive_writers": run_sequence(
            "positive_writers", positive_writers, model, tokenizer, m_pairs,
            t_pairs, direction, neutral_baseline, args.batch_size,
            args.max_tokens, args.seed, device,
        ),
        "suppressors": run_sequence(
            "suppressors", suppressors, model, tokenizer, m_pairs, t_pairs,
            direction, neutral_baseline, args.batch_size, args.max_tokens,
            args.seed, device, block=args.block,
        ),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log.info("wrote %s", out / "summary.json")


if __name__ == "__main__":
    main()
