"""Norm-matched random controls for the weight-orthogonalization result.

Publication gate 1 (after 2ed963d): Sol's random controls edited *different*
components, whose alignment with the direction — and hence edit magnitude — is
naturally tiny (~2% relative change). This leaves open the confound that any
equally large edit to the *same* weights would reproduce the helping effect.

Control here: for each seed, edit the SAME components (positive writers
L19MLP+L20H15; separately suppressors) against a RANDOM unit direction, with
the rank-1 delta SCALED so its Frobenius norm exactly matches the targeted
edit's delta norm for that component. Same weights, same damage size, random
content. The targeted helping deltas (-0.206 / +0.263) should fall far outside
the null distribution if the effect is direction-specific.

Usage (Spark, `empathy` env):
  python -u src/norm_matched_controls.py \
    --direction results/controlled_directions_gemma2_9b_it/direction_M_block20.npy \
    --out results/norm_matched_controls_gemma2_9b_it
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

try:
    from src.weight_orthogonalization import (
        NEUTRAL_PROMPTS, POSITIVE_WRITERS, SUPPRESSORS,
        component_weight, effective_direction, evaluate, final_logits,
        load_pairs, orthogonalize_component, parse_component, restore_weights,
        snapshot_weights,
    )
except ModuleNotFoundError:
    from weight_orthogonalization import (
        NEUTRAL_PROMPTS, POSITIVE_WRITERS, SUPPRESSORS,
        component_weight, effective_direction, evaluate, final_logits,
        load_pairs, orthogonalize_component, parse_component, restore_weights,
        snapshot_weights,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("norm-matched")


@torch.no_grad()
def targeted_delta_norm(model, component, direction):
    """Frobenius norm of the rank-1 delta the targeted edit would apply."""
    weight, columns = component_weight(model, component)
    effective = effective_direction(model, component, direction).to(weight.device)
    original = (weight if columns is None else weight[:, columns]).float()
    alignment = effective @ original
    return float(torch.linalg.vector_norm(alignment))  # ||delta||_F = ||e||*||align||


@torch.no_grad()
def apply_norm_matched_random(model, component, direction, target_norm, generator,
                              rand_vec=None):
    """Rank-1 edit along a random unit direction, scaled to target_norm.

    Pass rand_vec to share ONE random vector across the components of a set
    (procedure-matched to the targeted edit, which removes the same direction
    from every component); default draws a fresh vector from `generator`.
    """
    weight, columns = component_weight(model, component)
    d_model = weight.shape[0]
    rand = (torch.randn(d_model, generator=generator, dtype=torch.float32)
            if rand_vec is None else rand_vec.float().cpu())
    rand = rand.to(weight.device)
    # procedure-matched: pass the random vector through the same RMS-weight adjustment
    rand_effective = effective_direction(model, component, rand)
    original = (weight if columns is None else weight[:, columns]).float()
    alignment = rand_effective @ original
    norm = float(torch.linalg.vector_norm(alignment))
    scale = target_norm / max(norm, 1e-12)
    delta = (rand_effective[:, None] * alignment[None, :]) * scale
    if columns is None:
        weight.sub_(delta.to(weight.dtype))
    else:
        weight[:, columns].sub_(delta.to(weight.dtype))
    return {"component": component["name"], "scale": scale,
            "applied_delta_norm": target_norm}


def summarize(name, targeted_delta, null_deltas):
    null = np.asarray(null_deltas)
    z = (targeted_delta - null.mean()) / max(null.std(ddof=1), 1e-9)
    frac_beyond = float(np.mean(np.abs(null) >= abs(targeted_delta)))
    return {"set": name, "targeted_helping_delta": targeted_delta,
            "null_mean": float(null.mean()), "null_std": float(null.std(ddof=1)),
            "null_values": null_deltas, "z_score": float(z),
            "fraction_null_as_extreme": frac_beyond}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2-9b-it")
    parser.add_argument("--direction", required=True)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--m-pairs", default="data/contrastive_pairs/v2_1/M_templated.jsonl")
    parser.add_argument("--t-pairs", default="data/contrastive_pairs/v2_1/T_templated.jsonl")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block", type=int, default=20)
    parser.add_argument("--reference", default="results/weight_orthogonalization_gemma2_9b_it/analysis.json",
                        help="pilot analysis.json for context deltas; pass '' for re-derived sets")
    parser.add_argument("--writers", default=POSITIVE_WRITERS,
                        help="override positive_writers component set")
    parser.add_argument("--suppressors", default=SUPPRESSORS,
                        help="override suppressors component set")
    parser.add_argument("--out", default="results/norm_matched_controls_gemma2_9b_it")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:  # Llama-3.1 ships without one
        tokenizer.pad_token = tokenizer.eos_token
    # CRITICAL: choice_scores indexes logits at attention_mask.sum(1)-1, which
    # is only the final real token under RIGHT padding. Gemma defaults left.
    # (Absence of this line mis-indexed E14/E14b readouts — see log E23/E14d.)
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device)
    model.eval()

    direction = torch.tensor(np.load(args.direction), dtype=torch.float32, device=device)
    m_pairs, t_pairs = load_pairs(args.m_pairs), load_pairs(args.t_pairs)

    neutral_baseline = final_logits(model, tokenizer, NEUTRAL_PROMPTS,
                                    args.batch_size, args.max_tokens, device)
    baseline = evaluate(model, tokenizer, m_pairs, t_pairs, direction, neutral_baseline,
                        args.batch_size, args.max_tokens, args.seed, device, block=args.block)
    base_help = baseline["helping_choice"]["mean"]
    log.info("baseline helping %.4f", base_help)

    if args.reference:
        reference = json.load(open(args.reference))
        pilot_deltas = {
            "positive_writers": reference["conditions_vs_baseline"]["positive_writers_k2"]["helping"]["mean"],
            "suppressors": reference["conditions_vs_baseline"]["suppressors_k4"]["helping"]["mean"],
        }
    else:  # re-derived sets have no pilot reference; in-run deltas are the anchor
        pilot_deltas = {"positive_writers": float("nan"), "suppressors": float("nan")}

    sets = {"positive_writers": [parse_component(v) for v in args.writers.split(",")],
            "suppressors": [parse_component(v) for v in args.suppressors.split(",")]}

    # recompute targeted deltas IN-RUN so the z-score compares like with like
    # (the pilot's baseline may differ in loading config / aggregation)
    ref_deltas = {}
    for set_name, components in sets.items():
        snapshots = snapshot_weights(model, components)
        try:
            for c in components:
                orthogonalize_component(model, c, direction)
            metrics = evaluate(model, tokenizer, m_pairs, t_pairs, direction,
                               neutral_baseline, args.batch_size, args.max_tokens,
                               args.seed, device, block=args.block)
        finally:
            restore_weights(snapshots)
        ref_deltas[set_name] = metrics["helping_choice"]["mean"] - base_help
        log.info("%s targeted (in-run): helping delta %+.4f (pilot: %+.4f)",
                 set_name, ref_deltas[set_name], pilot_deltas[set_name])

    results = {"baseline_helping": base_help, "n_seeds": args.n_seeds, "sets": {}}
    for set_name, components in sets.items():
        target_norms = {c["name"]: targeted_delta_norm(model, c, direction) for c in components}
        log.info("%s targeted delta norms: %s", set_name,
                 {k: round(v, 4) for k, v in target_norms.items()})
        null_deltas, conditions = [], []
        for s in range(args.n_seeds):
            generator = torch.Generator().manual_seed(args.seed * 1000 + s)
            snapshots = snapshot_weights(model, components)
            try:
                edits = [apply_norm_matched_random(model, c, direction,
                                                   target_norms[c["name"]], generator)
                         for c in components]
                metrics = evaluate(model, tokenizer, m_pairs, t_pairs, direction,
                                   neutral_baseline, args.batch_size, args.max_tokens,
                                   args.seed, device, block=args.block)
            finally:
                restore_weights(snapshots)
            delta = metrics["helping_choice"]["mean"] - base_help
            null_deltas.append(delta)
            conditions.append({"seed": s, "edits": edits,
                               "helping_delta": delta,
                               "task_delta": metrics["task_choice"]["mean"] - baseline["task_choice"]["mean"],
                               "neutral_kl": metrics["neutral_drift"]["mean_kl_from_baseline"]})
            log.info("%s seed %d: helping delta %+.4f", set_name, s, delta)
        results["sets"][set_name] = {
            "summary": summarize(set_name, ref_deltas[set_name], null_deltas),
            "pilot_helping_delta": pilot_deltas[set_name],
            "target_delta_norms": target_norms,
            "conditions": conditions,
        }
        log.info("%s: targeted %+0.4f | null %+0.4f ± %.4f | z=%.2f",
                 set_name, ref_deltas[set_name],
                 results["sets"][set_name]["summary"]["null_mean"],
                 results["sets"][set_name]["summary"]["null_std"],
                 results["sets"][set_name]["summary"]["z_score"])

    with open(out / "norm_matched_controls.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info("wrote %s", out / "norm_matched_controls.json")


if __name__ == "__main__":
    main()
