"""E17 stage 3: Llama-3.1-8B weight-edit evaluation (amendment v3 protocol).

Order of operations (fixed by pre-registration + amendments; see EXPERIMENT_LOG):
  gate A  certify the frozen direction on confirmatory M: tail-projection AUROC
          at B* must be >= 0.85 (else stop and report)
  gate B  record baseline forced-choice headroom on confirmatory M
  edits   conditions = baseline; positive_writers k=1..2; suppressors k=1..4;
          targeted k=6; random k=6; individual components — every condition
          evaluated by choice_scores on {M_confirm (PRIMARY), T_confirm,
          D, G, H, dev-M, dev-T} + decision_separation on M_confirm
          (manipulation check) + neutral drift
  checks  realized post-cast delta + idempotence (re-apply must be a near-no-op)
  stats   family-clustered bootstrap deltas vs baseline; pre-registered
          selectivity ratios (|dM_confirm| >= 3x each control)

Usage (Spark, `empathy` env):
  python -u src/e17_stage3.py \
    --direction results/controlled_directions_llama31_8b_it/direction_M_grouped_block15.npy \
    --block 15 --out results/e17_stage3_llama31_8b_it
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

try:
    from src.weight_orthogonalization import (
        NEUTRAL_PROMPTS, choice_scores, decision_separation, final_logits,
        load_pairs, neutral_drift, orthogonalize_component, parse_component,
        restore_weights, snapshot_weights, component_weight, effective_direction,
        tokenize,
    )
except ModuleNotFoundError:
    from weight_orthogonalization import (
        NEUTRAL_PROMPTS, choice_scores, decision_separation, final_logits,
        load_pairs, neutral_drift, orthogonalize_component, parse_component,
        restore_weights, snapshot_weights, component_weight, effective_direction,
        tokenize,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("e17-stage3")

# Llama-3.1-8B defaults (amendment v3); override via CLI for other models,
# e.g. E14c Gemma retro-eval with its E13 components
POSITIVE_WRITERS = "L15MLP,L12H4"
SUPPRESSORS = "L15H6,L14H27,L12H20,L11MLP"
RANDOM = "L1MLP,L12H15,L15H17,L14H6,L12H31,L2MLP"

EVAL_SETS = {
    "M_confirm": "data/contrastive_pairs/v2_1/M_confirm_templated.jsonl",   # PRIMARY
    "T_confirm": "data/contrastive_pairs/v2_1/T_confirm_templated.jsonl",   # negative control
    "D": "data/contrastive_pairs/v2_1_audited/D.jsonl",                     # warmth control
    "G": "data/contrastive_pairs/v2_1_audited/G.jsonl",                     # motive control
    "H": "data/contrastive_pairs/v2_1_audited/H.jsonl",                     # character control
    "M_dev": "data/contrastive_pairs/v2_1/M_templated.jsonl",               # comparability only
    "T_dev": "data/contrastive_pairs/v2_1/T_templated.jsonl",
}


@torch.no_grad()
def tail_projections(model, tokenizer, pairs, direction, block, batch_size,
                     max_tokens, device):
    """Per-side tail-pooled projections at `block` (for AUROC certification)."""
    texts, prefixes = [], []
    for pair in pairs:
        texts.extend([pair["pos_text"], pair["neg_text"]])
        prefixes.extend([pair["shared_prefix"], pair["shared_prefix"]])
    captured = {}

    def hook(_m, _i, output):
        captured["h"] = output[0] if isinstance(output, tuple) else output

    handle = model.model.layers[block].register_forward_hook(hook)
    projections, excluded = [], 0
    try:
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            encoded, offsets = tokenize(tokenizer, chunk, max_tokens, offsets=True)
            inputs = {k: v.to(device) for k, v in encoded.items()}
            model(**inputs, use_cache=False)
            hidden = captured["h"].float()
            mask = inputs["attention_mask"].bool()
            for row, prefix in enumerate(prefixes[start:start + batch_size]):
                tail = (offsets[row, :, 1] > len(prefix)).to(device) & mask[row]
                if not tail.any():  # amendment 6: exclude, never silently pool
                    excluded += 1
                    projections.append(np.nan)
                    continue
                projections.append(float(hidden[row, tail].mean(0) @ direction))
    finally:
        handle.remove()
    return np.asarray(projections), excluded


def auroc_from_projections(proj):
    from sklearn.metrics import roc_auc_score
    pos, neg = proj[0::2], proj[1::2]
    keep = ~(np.isnan(pos) | np.isnan(neg))
    labels = np.concatenate([np.ones(keep.sum()), np.zeros(keep.sum())])
    return float(roc_auc_score(labels, np.concatenate([pos[keep], neg[keep]])))


def evaluate_all(model, tokenizer, pair_sets, direction, block, batch_size,
                 max_tokens, seed, device, neutral_baseline):
    out = {}
    for name, pairs in pair_sets.items():
        scores = choice_scores(model, tokenizer, pairs, seed, batch_size, max_tokens, device)
        out[name] = {"mean": float(np.mean(scores)), "per_pair": scores}
    out["decision_projection_M_confirm"] = decision_separation(
        model, tokenizer, pair_sets["M_confirm"], direction, block,
        batch_size, max_tokens, device)
    neutral = final_logits(model, tokenizer, NEUTRAL_PROMPTS, batch_size, max_tokens, device)
    out["neutral_drift"] = neutral_drift(neutral_baseline, neutral)
    return out


def clustered_delta(cond_scores, base_scores, families, seed=0, n_boot=5000):
    """Family-clustered bootstrap CI of mean(cond - base)."""
    d = np.asarray(cond_scores) - np.asarray(base_scores)
    fams = np.asarray(families)
    unique = sorted(set(fams))
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        pick = rng.choice(len(unique), len(unique), replace=True)
        sel = np.concatenate([d[fams == unique[i]] for i in pick])
        means.append(sel.mean())
    return {"mean": float(d.mean()),
            "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "n_families": len(unique)}


def apply_and_check(model, component, direction):
    """Edit + realized-delta / idempotence record (amendment 6)."""
    first = orthogonalize_component(model, component, direction)
    weight, cols = component_weight(model, component)
    eff = effective_direction(model, component, direction).to(weight.device)
    residual_alignment = float(torch.linalg.vector_norm(
        eff @ (weight if cols is None else weight[:, cols]).float()))
    second = orthogonalize_component(model, component, direction)  # near-no-op if clean
    return {**first,
            "post_edit_residual_alignment": residual_alignment,
            "idempotence_second_removal_norm": second["removed_alignment_norm"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--direction", required=True)
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--auroc-gate", type=float, default=0.85)
    ap.add_argument("--writers", default=POSITIVE_WRITERS)
    ap.add_argument("--suppressors", default=SUPPRESSORS)
    ap.add_argument("--random-components", default=RANDOM)
    ap.add_argument("--out", default="results/e17_stage3_llama31_8b_it")
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

    pair_sets = {name: load_pairs(path) for name, path in EVAL_SETS.items()}
    # free-form cells (D/G/H) lack shared_prefix; build_choice_prompt slices by
    # it, so synthesize the longest common prefix backed off to a word boundary
    # (empty prefix -> options are the full texts, which is the intended format)
    import os.path
    for pairs in pair_sets.values():
        for p in pairs:
            if "shared_prefix" not in p:
                lcp = os.path.commonprefix([p["pos_text"], p["neg_text"]])
                p["shared_prefix"] = lcp[:lcp.rfind(" ") + 1] if " " in lcp else ""
    families = {name: [p.get("scenario_id", str(i)) for i, p in enumerate(pairs)]
                for name, pairs in pair_sets.items()}
    for name, pairs in pair_sets.items():
        log.info("eval set %s: %d pairs, %d families", name, len(pairs), len(set(families[name])))

    results = {"model": args.model, "direction": args.direction, "block": args.block,
               "sets": {k: len(v) for k, v in pair_sets.items()}}

    # ---- gate A: representation certification on confirmatory M ----
    proj, excluded = tail_projections(model, tokenizer, pair_sets["M_confirm"],
                                      direction, args.block, args.batch_size,
                                      args.max_tokens, device)
    gate_auroc = auroc_from_projections(proj)
    results["gate_A"] = {"confirm_M_auroc": gate_auroc, "tail_excluded": excluded,
                         "threshold": args.auroc_gate, "pass": gate_auroc >= args.auroc_gate}
    log.info("GATE A: confirmatory-M AUROC %.4f (excluded %d) -> %s",
             gate_auroc, excluded, "PASS" if gate_auroc >= args.auroc_gate else "FAIL")
    if gate_auroc < args.auroc_gate:
        (out / "stage3.json").write_text(json.dumps(results, indent=2))
        log.error("gate A failed; stopping before any edit evaluation")
        return

    # ---- gate B + baseline ----
    neutral_baseline = final_logits(model, tokenizer, NEUTRAL_PROMPTS,
                                    args.batch_size, args.max_tokens, device)
    baseline = evaluate_all(model, tokenizer, pair_sets, direction, args.block,
                            args.batch_size, args.max_tokens, args.seed, device,
                            neutral_baseline)
    results["baseline"] = baseline
    log.info("GATE B: baseline M_confirm choice mean %+.4f | T_confirm %+.4f",
             baseline["M_confirm"]["mean"], baseline["T_confirm"]["mean"])

    # ---- edit conditions ----
    def run_sequence(name, components):
        conditions = []
        snapshots = snapshot_weights(model, components)
        try:
            edits = []
            for k, comp in enumerate(components, start=1):
                edits.append(apply_and_check(model, comp, direction))
                log.info("%s k=%d: %s", name, k, comp["name"])
                conditions.append({
                    "k": k, "components": [c["name"] for c in components[:k]],
                    "edits": list(edits),
                    "metrics": evaluate_all(model, tokenizer, pair_sets, direction,
                                            args.block, args.batch_size,
                                            args.max_tokens, args.seed, device,
                                            neutral_baseline)})
        finally:
            restore_weights(snapshots)
        return conditions

    def run_individuals(components):
        conds = []
        for comp in components:
            snapshots = snapshot_weights(model, [comp])
            try:
                edit = apply_and_check(model, comp, direction)
                log.info("individual: %s", comp["name"])
                conds.append({"component": comp["name"], "edit": edit,
                              "metrics": evaluate_all(model, tokenizer, pair_sets,
                                                      direction, args.block,
                                                      args.batch_size, args.max_tokens,
                                                      args.seed, device, neutral_baseline)})
            finally:
                restore_weights(snapshots)
        return conds

    writers = [parse_component(v) for v in args.writers.split(",")]
    sups = [parse_component(v) for v in args.suppressors.split(",")]
    targeted = writers + sups
    rand = [parse_component(v) for v in args.random_components.split(",")]

    results["positive_writers"] = run_sequence("positive_writers", writers)
    results["suppressors"] = run_sequence("suppressors", sups)
    results["targeted"] = run_sequence("targeted", targeted)
    results["random"] = run_sequence("random", rand)
    results["individuals"] = run_individuals(targeted)

    # ---- pre-registered deltas + selectivity ratios ----
    analysis = {}
    for cond_name, cond_key, k in [("positive_writers_k2", "positive_writers", 2),
                                   ("suppressors_k4", "suppressors", 4),
                                   ("targeted_k6", "targeted", 6),
                                   ("random_k6", "random", 6)]:
        cond = next(c for c in results[cond_key] if c["k"] == k)
        deltas = {}
        for s in EVAL_SETS:
            deltas[s] = clustered_delta(cond["metrics"][s]["per_pair"],
                                        baseline[s]["per_pair"], families[s],
                                        seed=args.seed)
        m = abs(deltas["M_confirm"]["mean"])
        ratios = {s: (m / abs(deltas[s]["mean"]) if deltas[s]["mean"] != 0 else float("inf"))
                  for s in ["T_confirm", "D", "G", "H"]}
        analysis[cond_name] = {"deltas": deltas, "selectivity_ratios": ratios,
                               "selective_3x": all(r >= 3 for r in ratios.values()),
                               "partial_2x": all(r >= 2 for r in ratios.values())}
        log.info("%s: dM_confirm %+.4f | ratios %s", cond_name,
                 deltas["M_confirm"]["mean"],
                 {s: round(r, 2) for s, r in ratios.items()})
    results["analysis"] = analysis

    (out / "stage3.json").write_text(json.dumps(results, indent=2))
    log.info("wrote %s", out / "stage3.json")


if __name__ == "__main__":
    main()
