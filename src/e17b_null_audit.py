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
import hashlib
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch

try:
    from src.weight_orthogonalization import (
        assert_restored, load_pairs, orthogonalize_component, parse_component,
        restore_weights, snapshot_weights, tokenize,
    )
    from src.activation_patching import build_choice_prompt
    from src.component_sets import ComponentSetError, resolve_component_sets
    from src.utils.evidence_run import (
        EvidenceRunError, atomic_write_json, evaluate_run_contract,
        revalidate_run_contract,
    )
    from src.utils.run_provenance import (
        collect_run_provenance, resolve_hf_commit, resolve_model_and_tokenizer,
        sha256_file,
    )
except ModuleNotFoundError:
    from weight_orthogonalization import (
        assert_restored, load_pairs, orthogonalize_component, parse_component,
        restore_weights, snapshot_weights, tokenize,
    )
    from activation_patching import build_choice_prompt
    from component_sets import ComponentSetError, resolve_component_sets
    from utils.evidence_run import (
        EvidenceRunError, atomic_write_json, evaluate_run_contract,
        revalidate_run_contract,
    )
    from utils.run_provenance import (
        collect_run_provenance, resolve_hf_commit, resolve_model_and_tokenizer,
        sha256_file,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("e17b")

EVAL_SETS = {
    "M_confirm": "data/contrastive_pairs/v2_1/M_confirm_templated.jsonl",
    "T_confirm": "data/contrastive_pairs/v2_1/T_confirm_templated.jsonl",
}

# ---------------------------------------------------------------------------
# Random-direction null protocol (Integrity Repair A QA Q7 / Gate 0C)
# ---------------------------------------------------------------------------

#: Readouts the repaired protocol can score. ``raw_ab_dual`` is the Gate-0C
#: primary; chat and scaffold-free continuation are format sensitivities.
NULL_READOUTS = ("raw_ab_dual", "chat_ab_dual", "continuation")

#: Frozen Gate-0C reconstruction protocol (GATE0C_RECONSTRUCTION_PREREG
#: part 2). Accepted mode implements EXACTLY this; a generic ``>=39`` check
#: or M-only raw scoring does not satisfy the preregistered test.
GATE0C_NULL_PROTOCOL = {
    "name": "gate0c_fractional_null_2026_07_13_v1",
    "model": "google/gemma-2-9b-it",
    "component_set": "gemma2_9b_it_resid_2026-07-12",
    "direction": {
        "path": ("results/controlled_directions_gemma2_9b_it/"
                 "direction_M_resid_block20.npy"),
        "sha256": "1b6d692e0e933d76c15f722fe996d01e1ca2ee8c8b72f19daf2f122cda294b42",
    },
    "block": 20,
    "cell_manifests": {
        "M_confirm": {
            "path": "data/contrastive_pairs/v2_1/M_confirm_templated.jsonl",
            "sha256": "2e4bcd4d59538f0e5316b23f55247418636deca2b4df0d72cb6cca690ad847d4",
            "families": ("mc_archive", "mc_convoy", "mc_garden", "mc_kitchen",
                         "mc_stream", "mc_warehouse"),
        },
        "T_confirm": {
            "path": "data/contrastive_pairs/v2_1/T_confirm_templated.jsonl",
            "sha256": "33bed6a287d71e6515838c94027ec5a754e5bd2fb86970b29ca4c966225300e4",
            "families": ("tc_bikes", "tc_recipes", "tc_seedbank",
                         "tc_telescope", "tc_transcribe"),
        },
    },
    "n_random_directions": 39,
    "master_seed": 2490282906,
    "fractions": (0.0, 0.25, 0.5, 0.75, 1.0),
    "full_ablation_fraction": 1.0,
    "readouts": NULL_READOUTS,
    "cells": ("M_confirm", "T_confirm"),
    "min_attainable_plus_one_p": 1.0 / 40,
    "seed_stream": ("numpy.random.Generator(PCG64(master_seed))"
                    ".integers(0, 2**32, dtype=uint32) consumed in order"),
    "direction_generation": (
        "per child seed: fresh PCG64 generator, d_model standard-normal "
        "float64 on CPU, single cast to float32 host dtype, normalize to "
        "unit L2 norm, then transfer to device"),
    "statistic": ("Z = -(mean_margin_fraction1 - mean_margin_fraction0); "
                  "one-sided plus-one empirical p vs the 39 nulls"),
    "dose_monotonicity_tolerance": 0.02,
    "descriptive_min_selectivity_ratio": 2.0,
}

#: Minimum accepted control count, justified by the preregistered inferential
#: resolution: 39 nulls give a minimum attainable plus-one p of 1/40 = .025.
MIN_ACCEPTED_RANDOM_DIRECTIONS = GATE0C_NULL_PROTOCOL["n_random_directions"]
SMOKE_RANDOM_DIRECTIONS = 3


class NullProtocolError(ValueError):
    """The requested random-direction protocol is not evidence-eligible."""


def derive_child_seeds(master_seed, n):
    """Deterministic child-seed stream, independent of batch partitioning.

    Consumes ``Generator(PCG64(master_seed)).integers(0, 2**32,
    dtype=uint32)`` one value at a time, so the first k seeds are identical
    no matter how many are ultimately drawn or how work is chunked.
    """
    gen = np.random.Generator(np.random.PCG64(int(master_seed)))
    return [int(gen.integers(0, 2 ** 32, dtype=np.uint32)) for _ in range(n)]


def isotropic_unit_direction(child_seed, d_model):
    """Gate-0C random direction: fresh PCG64, float64 normals, single cast
    to float32, unit L2 normalization. Returns (vector, derivation record)."""
    gen = np.random.Generator(np.random.PCG64(int(child_seed)))
    raw64 = gen.standard_normal(int(d_model))  # float64 on CPU
    pre_cast_norm = float(np.linalg.norm(raw64))
    raw32 = raw64.astype(np.float32)  # cast ONCE to the target host dtype
    post_cast_norm = float(np.linalg.norm(raw32))
    unit = raw32 / np.float32(np.linalg.norm(raw32))
    record = {
        "seed": int(child_seed),
        "d_model": int(d_model),
        "pre_cast_norm_float64": pre_cast_norm,
        "post_cast_norm_float32": post_cast_norm,
        "unit_direction_sha256": hashlib.sha256(unit.tobytes()).hexdigest(),
    }
    return unit, record


def validate_null_protocol(run_mode, n_random_directions, master_seed,
                           fractions, readouts, cells):
    """Validate + freeze the random-direction protocol BEFORE any model work.

    Accepted mode implements the frozen Gate-0C reconstruction exactly;
    everything else is persisted as non-evidential. Returns the protocol
    block persisted in the artifact.
    """
    frozen = GATE0C_NULL_PROTOCOL
    if run_mode == "accepted":
        if n_random_directions is None:
            raise NullProtocolError(
                "accepted mode requires an explicit --n-random-directions "
                "(omitted count is rejected; the frozen Gate-0C protocol "
                f"uses exactly {frozen['n_random_directions']})")
        if n_random_directions != frozen["n_random_directions"]:
            raise NullProtocolError(
                f"accepted mode requires exactly "
                f"{frozen['n_random_directions']} isotropic unit controls "
                f"(minimum plus-one p {frozen['min_attainable_plus_one_p']}); "
                f"got {n_random_directions} (three smoke draws are "
                "exploratory-only)")
        if master_seed != frozen["master_seed"]:
            raise NullProtocolError(
                f"accepted mode requires master seed "
                f"{frozen['master_seed']}; got {master_seed}")
        if tuple(fractions or ()) != frozen["fractions"]:
            raise NullProtocolError(
                f"accepted mode requires fractions {frozen['fractions']}; "
                f"got {tuple(fractions or ())}")
        if tuple(readouts or ()) != frozen["readouts"]:
            raise NullProtocolError(
                f"accepted mode requires readouts {frozen['readouts']} "
                f"(raw dual-order primary + chat dual-order + scaffold-free "
                f"continuation); got {tuple(readouts or ())}")
        if tuple(cells or ()) != frozen["cells"]:
            raise NullProtocolError(
                f"accepted mode scores the same M and T cells as the target "
                f"ablation: {frozen['cells']}; got {tuple(cells or ())}")
    for readout in readouts or ():
        if readout not in NULL_READOUTS:
            raise NullProtocolError(
                f"unknown readout {readout!r}; known: {NULL_READOUTS}")
    n = (SMOKE_RANDOM_DIRECTIONS if n_random_directions is None
         else n_random_directions)
    smoke = n < MIN_ACCEPTED_RANDOM_DIRECTIONS
    protocol = {
        "run_mode": run_mode,
        "n_random_directions": n,
        "master_seed": master_seed,
        "fractions": list(fractions or ()),
        "readouts": list(readouts or ()),
        "cells": list(cells or ()),
        "min_attainable_plus_one_p": 1.0 / (n + 1),
        "seed_stream": frozen["seed_stream"],
        "direction_generation": frozen["direction_generation"],
        "smoke": smoke,
    }
    if smoke or run_mode != "accepted":
        protocol["evidence_note"] = (
            "NON-EVIDENTIAL "
            + ("SMOKE " if smoke else "")
            + "protocol: this artifact cannot support the preregistered "
            "fractional-ablation inference (accepted mode requires the "
            f"frozen Gate-0C protocol: {frozen['n_random_directions']} "
            f"controls, master seed {frozen['master_seed']}, fractions "
            f"{frozen['fractions']}, readouts {frozen['readouts']} on cells "
            f"{frozen['cells']})")
    return protocol


def validate_gate0c_binding(*, protocol_name, model, component_set,
                            direction_path, block, repo_root=None):
    """Bind accepted Gate-0C to the preregistered Gemma experiment bytes."""
    frozen = GATE0C_NULL_PROTOCOL
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
    direction_arg = Path(direction_path)
    if direction_arg.is_absolute():
        try:
            direction_arg = direction_arg.resolve().relative_to(root.resolve())
        except ValueError:
            pass
    observed = {
        "protocol_name": protocol_name, "model": model,
        "component_set": component_set, "block": block,
        "direction_path": direction_arg.as_posix(),
    }
    expected = {
        "protocol_name": frozen["name"], "model": frozen["model"],
        "component_set": frozen["component_set"], "block": frozen["block"],
        "direction_path": frozen["direction"]["path"],
    }
    if observed != expected:
        raise NullProtocolError(
            f"Gate-0C experiment binding mismatch: expected {expected}, "
            f"observed {observed}")
    direction = root / frozen["direction"]["path"]
    if not direction.is_file() or sha256_file(direction) != frozen["direction"]["sha256"]:
        raise NullProtocolError("Gate-0C direction path/hash mismatch")
    realized = {}
    for cell, spec in frozen["cell_manifests"].items():
        path = root / spec["path"]
        if not path.is_file() or sha256_file(path) != spec["sha256"]:
            raise NullProtocolError(f"Gate-0C {cell} path/hash mismatch")
        rows = load_pairs(path)
        families = tuple(sorted({row["scenario_id"] for row in rows}))
        if families != spec["families"]:
            raise NullProtocolError(f"Gate-0C {cell} family manifest mismatch")
        realized[cell] = {"path": spec["path"], "sha256": spec["sha256"],
                          "families": list(families), "n_rows": len(rows)}
    return {"name": frozen["name"], "binding": expected,
            "direction_sha256": frozen["direction"]["sha256"],
            "cells": realized}


def signed_full_ablation_effect(fraction0_scores, fraction1_scores):
    """Gate-0C primary statistic: Z = -(mean_f1 - mean_f0); positive Z means
    removing the tested subspace reduces the costly-helping margin."""
    return float(-(np.mean(fraction1_scores) - np.mean(fraction0_scores)))


def plus_one_rank_p(z_target, z_nulls):
    """One-sided empirical plus-one p: target vs identically computed nulls."""
    n_ge = sum(1 for z in z_nulls if z >= z_target)
    return (1 + n_ge) / (len(z_nulls) + 1)


def analyze_fractional_protocol(protocol_scores, families, readouts,
                                *, bootstrap_seed=946441184,
                                n_boot=5000):
    """Pure Gate-0C inference and robustness gates from persisted scores."""
    fractions = protocol_scores["target_fractions"]
    if not {"0.0", "1.0"} <= set(fractions):
        raise NullProtocolError("fractional analysis requires f=0 and f=1")
    tolerance = GATE0C_NULL_PROTOCOL["dose_monotonicity_tolerance"]
    ordered = [str(f) for f in GATE0C_NULL_PROTOCOL["fractions"]]
    analysis = {"readouts": {}, "all_required_gates_pass": True}
    for readout in readouts:
        per_cell = {}
        for cell in GATE0C_NULL_PROTOCOL["cells"]:
            f0 = np.asarray(fractions["0.0"][readout][cell]["scores"], dtype=float)
            f1 = np.asarray(fractions["1.0"][readout][cell]["scores"], dtype=float)
            fam = np.asarray(families[cell])
            if not (len(f0) == len(f1) == len(fam)):
                raise NullProtocolError(f"unaligned scores/families for {readout}/{cell}")
            z_target = signed_full_ablation_effect(f0, f1)
            z_nulls = [signed_full_ablation_effect(
                f0, entry["scores"][readout][cell]["scores"])
                for entry in protocol_scores["random"].values()]
            family_effects = {
                name: signed_full_ablation_effect(f0[fam == name], f1[fam == name])
                for name in sorted(set(fam))}
            names = sorted(family_effects)
            rng = np.random.default_rng(bootstrap_seed)
            boot = [float(np.mean([family_effects[names[i]] for i in
                                   rng.integers(0, len(names), len(names))]))
                    for _ in range(n_boot)]
            lofo = {
                omitted: float(np.mean([value for name, value in family_effects.items()
                                        if name != omitted]))
                for omitted in names}
            means = [float(np.mean(
                fractions[key][readout][cell]["scores"])) for key in ordered]
            monotone = all(b <= a + tolerance for a, b in zip(means, means[1:]))
            gates = {
                "target_greater_than_all_nulls": bool(z_nulls and z_target > max(z_nulls)),
                "family_ci_lower_gt_zero": bool(np.percentile(boot, 2.5) > 0),
                "lofo_all_positive": bool(lofo and min(lofo.values()) > 0),
                "dose_nonincreasing_with_tolerance": monotone,
            }
            per_cell[cell] = {
                "Z_target": z_target, "Z_nulls": z_nulls,
                "plus_one_p_one_sided": plus_one_rank_p(z_target, z_nulls),
                "family_effects": family_effects,
                "family_bootstrap_ci95": [float(np.percentile(boot, 2.5)),
                                           float(np.percentile(boot, 97.5))],
                "lofo_effects": lofo, "dose_mean_margins": means,
                "dose_monotonicity_tolerance": tolerance, "gates": gates,
            }
        m = per_cell["M_confirm"]["Z_target"]
        t = per_cell["T_confirm"]["Z_target"]
        ratio = abs(m) / max(abs(t), 1e-12)
        per_cell["descriptive_selectivity"] = {
            "abs_M_over_abs_T": ratio,
            "threshold": GATE0C_NULL_PROTOCOL["descriptive_min_selectivity_ratio"],
            "passes": ratio >= GATE0C_NULL_PROTOCOL[
                "descriptive_min_selectivity_ratio"],
            "interpretation": "descriptive gate, not an equivalence claim",
        }
        required = per_cell["M_confirm"]["gates"]
        readout_pass = (all(required.values())
                        and per_cell["descriptive_selectivity"]["passes"])
        per_cell["required_M_gates_pass"] = readout_pass
        analysis["all_required_gates_pass"] &= readout_pass
        analysis["readouts"][readout] = per_cell
    return analysis


def run_fractional_protocol(*, score_fn, ablator_factory, cells, readouts,
                            fractions, random_directions,
                            full_ablation_fraction=1.0):
    """Orchestrate the fractional protocol so nulls mirror the target exactly.

    ``score_fn(readout, cell)`` scores one cell under one readout with the
    CURRENTLY installed hooks; ``ablator_factory(direction, fraction)``
    returns a context manager installing the ablation ("target" selects the
    tested direction). ``random_directions`` is a list of
    ``(label, vector, derivation_record)``. Every random direction is scored
    on the SAME cells and readouts as the target ablation (QA Q7) at the
    full-ablation fraction. Fraction 0.0 is the no-hook baseline.
    """
    def score_all():
        return {readout: {cell: score_fn(readout, cell) for cell in cells}
                for readout in readouts}

    result = {"baseline": score_all(), "target_fractions": {}, "random": {}}
    for fraction in fractions:
        if fraction == 0.0:
            result["target_fractions"]["0.0"] = result["baseline"]
            continue
        with ablator_factory("target", fraction):
            result["target_fractions"][str(fraction)] = score_all()
    for label, vector, derivation in random_directions:
        with ablator_factory(vector, full_ablation_fraction):
            scores = score_all()
        result["random"][label] = {"derivation": derivation,
                                   "fraction": full_ablation_fraction,
                                   "scores": scores}
    return result


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


@torch.no_grad()
def choice_scores_dual_order(model, tokenizer, pairs, batch_size, max_tokens,
                             device, chat=False):
    """Dual-order A/B (Gate 0B/0C protocol): score BOTH option orders for
    every pair and average the flip-corrected logit differences. Returns
    (averaged scores, {"order_pos_is_A": [...], "order_pos_is_B": [...]}).
    """
    def fixed_order(flip):
        prompts = [build_choice_prompt(pair, flip) for pair in pairs]
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
            for row, length in enumerate(lengths.tolist()):
                la = float(logits[row, length - 1, tok_a[0]])
                lb = float(logits[row, length - 1, tok_b[0]])
                scores.append((lb - la) if flip else (la - lb))
        return scores

    pos_is_a = fixed_order(False)
    pos_is_b = fixed_order(True)
    averaged = [(a + b) / 2.0 for a, b in zip(pos_is_a, pos_is_b)]
    return averaged, {"order_pos_is_A": pos_is_a, "order_pos_is_B": pos_is_b}


@torch.no_grad()
def continuation_scores(model, tokenizer, pairs, batch_size, max_tokens,
                        device):
    """Scaffold-free continuation likelihood (E26 readout 3): mean per-token
    logprob of each branch's decision tail given the shared prefix;
    score = lp(pos_tail) - lp(neg_tail)."""
    def tail_lp(texts, prefixes):
        lps = []
        for start in range(0, len(texts), batch_size):
            enc, offs = tokenize(tokenizer, texts[start:start + batch_size],
                                 max_tokens, offsets=True)
            inputs = {k: v.to(device) for k, v in enc.items()}
            logits = model(**inputs, use_cache=False).logits.float()
            logp = torch.log_softmax(logits, -1)
            for row in range(len(inputs["input_ids"])):
                prefix = prefixes[start + row]
                mask = ((offs[row, :, 1] > len(prefix))
                        & enc["attention_mask"][row].bool())
                idx = mask.nonzero().squeeze(-1)
                idx = idx[idx > 0]
                tokens = inputs["input_ids"][row, idx]
                lp = logp[row, idx - 1].gather(-1, tokens[:, None]).squeeze(-1)
                lps.append(float(lp.mean()))
        return np.asarray(lps)

    pos = tail_lp([p["pos_text"] for p in pairs],
                  [p["shared_prefix"] for p in pairs])
    neg = tail_lp([p["neg_text"] for p in pairs],
                  [p["shared_prefix"] for p in pairs])
    return (pos - neg).tolist()


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


def validate_e17b_artifact(payload):
    required = {"model", "block", "run_contract", "component_sets",
                "direction_file", "cells", "provenance"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"incomplete e17b artifact: missing {missing}")
    if payload["run_contract"].get("run_mode") == "accepted":
        test3 = payload.get("test3_fractional_ablation") or {}
        if not test3.get("gate0c_analysis", {}).get("all_required_gates_pass"):
            raise ValueError("accepted e17b artifact failed Gate-0C analysis gates")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--revision", default=None,
                    help="explicit HF model revision (accepted reruns must pin this)")
    ap.add_argument("--tokenizer-revision", default=None,
                    help="explicit tokenizer revision (defaults to --revision)")
    ap.add_argument("--direction", required=True)
    ap.add_argument("--protocol", default=None,
                    help="named frozen protocol required in accepted mode")
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--ablate-blocks", type=int, nargs="+", default=[15, 20, 24, 28])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ablate-fractions", type=float, nargs="+", default=None,
                    help="test 3: fractional ablation curve at --block with "
                         "random-direction controls at f=1 (accepted mode "
                         "requires the frozen Gate-0C fractions 0 .25 .5 .75 1)")
    ap.add_argument("--n-random-directions", type=int, default=None,
                    help="random-direction control count; REQUIRED in "
                         "accepted mode (frozen Gate-0C value: 39). Omitted "
                         "= 3-draw exploratory smoke, labeled non-evidential")
    ap.add_argument("--null-master-seed", type=int, default=None,
                    help="master seed for the child-seed stream (frozen "
                         "Gate-0C value: 2490282906); default derives the "
                         "legacy smoke seeds from --seed")
    ap.add_argument("--readouts", nargs="+", default=["raw_ab_dual"],
                    choices=list(NULL_READOUTS),
                    help="test-3 readouts; accepted mode requires all of "
                         "raw_ab_dual chat_ab_dual continuation")
    ap.add_argument("--run-mode", choices=("accepted", "exploratory"),
                    default="exploratory",
                    help="accepted = evidence-eligible (frozen Gate-0C null "
                         "protocol, pinned immutable revisions, fresh output, "
                         "clean source)")
    ap.add_argument("--allow-direction-mismatch", action="store_true",
                    help="EXPLORATORY-ONLY override; persisted as "
                         "non-confirmatory")
    ap.add_argument("--allowed-dirty", action="append", default=[],
                    help="explicit source-binding exclusion rule (glob) for "
                         "accepted mode; persisted")
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
            direction_path=args.direction,
            run_mode=args.run_mode,
            allow_direction_mismatch=args.allow_direction_mismatch,
        )
    except ComponentSetError as exc:
        ap.error(str(exc))
    log.info("component sets resolved: %s", resolution)
    writers_spec = resolution["sets"]["positive_writers"]
    suppressors_spec = resolution["sets"]["suppressors"]
    random_spec = resolution["sets"]["random"]

    gate0c_binding = None
    if args.run_mode == "accepted":
        try:
            gate0c_binding = validate_gate0c_binding(
                protocol_name=args.protocol, model=args.model,
                component_set=args.component_set,
                direction_path=args.direction, block=args.block)
        except NullProtocolError as exc:
            ap.error(str(exc))

    # QA Q7: freeze + validate the random-direction protocol BEFORE any
    # model work (accepted mode implements the Gate-0C reconstruction only)
    null_protocol = None
    if args.ablate_fractions or args.run_mode == "accepted":
        try:
            null_protocol = validate_null_protocol(
                args.run_mode, args.n_random_directions,
                args.null_master_seed, args.ablate_fractions,
                args.readouts, tuple(EVAL_SETS))
        except NullProtocolError as exc:
            ap.error(str(exc))
        if null_protocol.get("evidence_note"):
            log.warning("%s", null_protocol["evidence_note"])

    out = Path(args.out)
    try:
        contract = evaluate_run_contract(
            args.run_mode,
            revisions={
                "model": {
                    "requested": args.revision,
                    "resolution": resolve_hf_commit(
                        args.model, revision=args.revision or "main"),
                },
                "tokenizer": {
                    "requested": args.tokenizer_revision or args.revision,
                    "resolution": resolve_hf_commit(
                        args.model,
                        revision=(args.tokenizer_revision or args.revision
                                  or "main")),
                },
            },
            output_paths=(out / "e17b.json",),
            source_rules=args.allowed_dirty,
            input_paths=(args.direction, *EVAL_SETS.values()),
        )
    except EvidenceRunError as exc:
        ap.error(str(exc))
    if contract.get("warning"):
        log.warning("%s", contract["warning"])

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
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
        "run_contract": contract,
        "component_sets": resolution,
        "gate0c_binding": gate0c_binding,
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
            deltas["restoration"] = assert_restored(snap)
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

    # ---- test 3: fractional ablation dose curve + random-direction controls
    # (QA Q7): the nulls score the SAME M and T cells and the SAME readouts
    # as the target ablation; count/seeds/derivation are frozen + persisted.
    if args.ablate_fractions:
        def score_readout(readout, cell):
            pairs = pair_sets[cell]
            if readout in ("raw_ab_dual", "chat_ab_dual"):
                averaged, per_order = choice_scores_dual_order(
                    model, tokenizer, pairs, args.batch_size,
                    args.max_tokens, device, chat=readout == "chat_ab_dual")
                return {"scores": averaged, "per_order": per_order}
            if readout == "continuation":
                return {"scores": continuation_scores(
                    model, tokenizer, pairs, args.batch_size,
                    args.max_tokens, device)}
            raise ValueError(f"unknown readout {readout!r}")

        n_random = null_protocol["n_random_directions"]
        if args.null_master_seed is not None:
            child_seeds = derive_child_seeds(args.null_master_seed, n_random)
            seed_method = "pcg64_uint32_stream_from_master_seed"
        else:  # legacy smoke derivation, exploratory only
            child_seeds = [args.seed * 100 + i for i in range(n_random)]
            seed_method = "legacy_smoke_seed*100+i"
        seed_derivation = {"master_seed": args.null_master_seed,
                           "method": seed_method,
                           "child_seeds": child_seeds}
        d_model = int(direction.shape[0])
        random_directions = []
        for i, child in enumerate(child_seeds):
            vec_np, derivation = isotropic_unit_direction(child, d_model)
            random_directions.append(
                (f"random_dir_{i}",
                 torch.from_numpy(vec_np).to(device), derivation))

        protocol_scores = run_fractional_protocol(
            score_fn=score_readout,
            ablator_factory=lambda vec, frac: DirectionAblator(
                model, args.block,
                direction if isinstance(vec, str) else vec, fraction=frac),
            cells=tuple(EVAL_SETS),
            readouts=tuple(args.readouts),
            fractions=tuple(args.ablate_fractions),
            random_directions=random_directions,
            full_ablation_fraction=1.0,
        )

        test3 = {
            "protocol": null_protocol,
            "seed_derivation": seed_derivation,
            "per_pair_scores": protocol_scores,
            "notes": (
                "per_pair_scores holds every per-pair score for every cell x "
                "readout x fraction and for each random direction at f=1; "
                "fraction_<f> keys keep the legacy clustered summaries on "
                "the primary readout"
            ),
        }
        primary = ("raw_ab_dual" if "raw_ab_dual" in args.readouts
                   else args.readouts[0])
        test3["primary_readout"] = primary
        base_primary = {c: protocol_scores["baseline"][primary][c]["scores"]
                        for c in EVAL_SETS}
        for fraction_key, per_readout in protocol_scores["target_fractions"].items():
            deltas = {c: clustered_delta(per_readout[primary][c]["scores"],
                                         base_primary[c], fams[c],
                                         seed=args.seed, detail=True)
                      for c in EVAL_SETS}
            test3[f"fraction_{fraction_key}"] = deltas
            log.info("[frac %s] dM %+0.4f CI %s | dT %+0.4f", fraction_key,
                     deltas["M_confirm"]["mean"], deltas["M_confirm"]["ci95"],
                     deltas["T_confirm"]["mean"])

        fraction_keys = set(protocol_scores["target_fractions"])
        if {"0.0", "1.0"} <= fraction_keys:
            analysis = {}
            for readout in args.readouts:
                analysis[readout] = {}
                for cell in EVAL_SETS:
                    f0 = protocol_scores["target_fractions"]["0.0"][readout][cell]["scores"]
                    f1 = protocol_scores["target_fractions"]["1.0"][readout][cell]["scores"]
                    z_target = signed_full_ablation_effect(f0, f1)
                    z_nulls = [
                        signed_full_ablation_effect(
                            f0, entry["scores"][readout][cell]["scores"])
                        for entry in protocol_scores["random"].values()
                    ]
                    analysis[readout][cell] = {
                        "statistic": GATE0C_NULL_PROTOCOL["statistic"],
                        "Z_target": z_target,
                        "Z_nulls": z_nulls,
                        "plus_one_p_one_sided": plus_one_rank_p(z_target,
                                                                z_nulls),
                        "min_attainable_p": 1.0 / (len(z_nulls) + 1),
                    }
            test3["signed_effect_analysis"] = analysis
            test3["gate0c_analysis"] = analyze_fractional_protocol(
                protocol_scores, fams, args.readouts)
        results["test3_fractional_ablation"] = test3

    try:
        revalidate_run_contract(
            contract, source_rules=args.allowed_dirty,
            input_paths=(args.direction, *EVAL_SETS.values()))
    except EvidenceRunError as exc:
        raise SystemExit(str(exc))
    atomic_write_json(out / "e17b.json", results,
                      require_fresh=True, validate_fn=validate_e17b_artifact)
    log.info("wrote %s", out / "e17b.json")


if __name__ == "__main__":
    main()
