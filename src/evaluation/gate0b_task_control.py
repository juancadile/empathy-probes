"""Gate 0B Part 1: repaired task-control replacement.

Scores the repaired and historical T-confirm batteries in the same process as
the exhausted M-confirm diagnostic denominator. Accepted mode is fixed to the
current Gemma direction/component set and immutable model revision.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from src.component_sets import ComponentSetError, resolve_component_sets
    from src.e17b_null_audit import (
        choice_scores_dual_order, continuation_scores,
    )
    from src.utils.evidence_run import (
        EvidenceRunError, atomic_write_json, evaluate_run_contract,
        revalidate_run_contract,
    )
    from src.utils.run_provenance import (
        collect_run_provenance, resolve_hf_commit,
        resolve_model_and_tokenizer, sha256_file,
    )
    from src.weight_orthogonalization import (
        assert_restored, load_pairs, orthogonalize_component,
        parse_component, restore_weights, snapshot_weights,
    )
except ModuleNotFoundError:
    from component_sets import ComponentSetError, resolve_component_sets
    from e17b_null_audit import choice_scores_dual_order, continuation_scores
    from utils.evidence_run import (
        EvidenceRunError, atomic_write_json, evaluate_run_contract,
        revalidate_run_contract,
    )
    from utils.run_provenance import (
        collect_run_provenance, resolve_hf_commit,
        resolve_model_and_tokenizer, sha256_file,
    )
    from weight_orthogonalization import (
        assert_restored, load_pairs, orthogonalize_component,
        parse_component, restore_weights, snapshot_weights,
    )


PROTOCOL = {
    "name": "gate0b_task_control_2026_07_13_v1",
    "model": "google/gemma-2-9b-it",
    "revision": "11c9b309abf73637e4b6f9a3fa1e92e615547819",
    "component_set": "gemma2_9b_it_resid_2026-07-12",
    "direction": (
        "results/controlled_directions_gemma2_9b_it/"
        "direction_M_resid_block20.npy"),
    "direction_sha256": (
        "1b6d692e0e933d76c15f722fe996d01e1ca2ee8c8b72f19daf2f122cda294b42"),
    "cells": {
        "M_confirm": {
            "path": "data/contrastive_pairs/v2_1/M_confirm_templated.jsonl",
            "sha256": "2e4bcd4d59538f0e5316b23f55247418636deca2b4df0d72cb6cca690ad847d4",
            "n_rows": 48,
        },
        "T_confirm_repaired": {
            "path": "data/contrastive_pairs/v2_1/T_confirm_templated.jsonl",
            "sha256": "33bed6a287d71e6515838c94027ec5a754e5bd2fb86970b29ca4c966225300e4",
            "n_rows": 40,
        },
        "T_confirm_historical": {
            "path": (
                "data/contrastive_pairs/v2_1/historical/"
                "T_confirm_templated.jsonl.c2740918/T_confirm_templated.jsonl"),
            "sha256": "c274091806e1c9fbccbf547971692e3f8460293da3686069f1cdef1fe79d050c",
            "n_rows": 40,
        },
    },
    "readouts": ("raw_ab_dual", "continuation", "chat_ab_dual"),
    "primary_readouts": ("raw_ab_dual", "continuation"),
    "bootstrap_seed": 3114916352,
    "bootstrap_draws": 10_000,
    "writer_abs_t_over_m_threshold": 1.0 / 3.0,
}


def family_summary(condition_scores, baseline_scores, families, *, seed,
                   n_boot):
    """Family-level effect, bootstrap, and LOFO from persisted pair scores."""
    delta = np.asarray(condition_scores, dtype=float) - np.asarray(
        baseline_scores, dtype=float)
    fam = np.asarray(families)
    names = sorted(set(fam))
    effects = {
        name: float(delta[fam == name].mean()) for name in names
    }
    values = np.asarray([effects[name] for name in names], dtype=float)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    boot = [float(values[rng.integers(0, len(values), len(values))].mean())
            for _ in range(n_boot)]
    lofo = {
        omitted: float(np.mean([value for name, value in effects.items()
                                if name != omitted]))
        for omitted in names
    }
    return {
        "mean": float(values.mean()),
        "family_effects": effects,
        "family_bootstrap_ci95": [float(np.percentile(boot, 2.5)),
                                    float(np.percentile(boot, 97.5))],
        "lofo_effects": lofo,
        "per_pair_delta": delta.tolist(),
    }


def analyze_gate0b(scores, families):
    """Apply the frozen writer selectivity rule; suppressors are descriptive."""
    summaries = {}
    seed = PROTOCOL["bootstrap_seed"]
    n_boot = PROTOCOL["bootstrap_draws"]
    for readout in PROTOCOL["readouts"]:
        summaries[readout] = {}
        for condition in ("positive_writers_k2", "suppressors_k4"):
            summaries[readout][condition] = {}
            for cell in PROTOCOL["cells"]:
                summaries[readout][condition][cell] = family_summary(
                    scores[condition][readout][cell]["scores"],
                    scores["baseline"][readout][cell]["scores"],
                    families[cell], seed=seed, n_boot=n_boot)

    threshold = PROTOCOL["writer_abs_t_over_m_threshold"]
    primary = {}
    for readout in PROTOCOL["primary_readouts"]:
        writer = summaries[readout]["positive_writers_k2"]
        m = writer["M_confirm"]["mean"]
        t = writer["T_confirm_repaired"]["mean"]
        ratio = abs(t) / max(abs(m), 1e-12)
        primary[readout] = {
            "M_confirm_effect": m,
            "T_confirm_repaired_effect": t,
            "abs_T_over_abs_M": ratio,
            "threshold": threshold,
            "pass": bool(ratio < threshold),
        }
    return {
        "summaries": summaries,
        "writer_selectivity": primary,
        "all_required_gates_pass": bool(
            all(entry["pass"] for entry in primary.values())),
        "claim_ceiling": (
            "magnitude selectivity on repaired T-confirm relative to the "
            "simultaneous exhausted M-confirm diagnostic; not task "
            "equivalence, fresh writer confirmation, or construct purity"),
    }


def validate_artifact(payload):
    required = {"schema", "protocol", "run_contract", "component_sets",
                "direction", "cells", "scores", "analysis", "provenance"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"incomplete Gate 0B artifact: missing {missing}")
    if payload["run_contract"].get("run_mode") == "accepted":
        if payload["protocol"].get("name") != PROTOCOL["name"]:
            raise ValueError("accepted Gate 0B protocol mismatch")
        verdict = payload["analysis"].get("all_required_gates_pass")
        if not isinstance(verdict, bool):
            raise ValueError("accepted artifact lacks a boolean gate verdict")
        for condition in ("baseline", "positive_writers_k2", "suppressors_k4"):
            for readout in PROTOCOL["readouts"]:
                for cell, spec in PROTOCOL["cells"].items():
                    record = payload["scores"][condition][readout][cell]
                    if len(record.get("scores", [])) != spec["n_rows"]:
                        raise ValueError(
                            f"incomplete scores for {condition}/{readout}/{cell}")


def _validate_binding(args):
    if args.run_mode != "accepted":
        return
    expected = {
        "model": PROTOCOL["model"], "revision": PROTOCOL["revision"],
        "tokenizer_revision": PROTOCOL["revision"],
        "component_set": PROTOCOL["component_set"],
        "direction": PROTOCOL["direction"],
    }
    observed = {
        "model": args.model, "revision": args.revision,
        "tokenizer_revision": args.tokenizer_revision,
        "component_set": args.component_set, "direction": args.direction,
    }
    if observed != expected:
        raise ValueError(
            f"accepted Gate 0B binding mismatch: expected {expected}, got {observed}")
    if sha256_file(args.direction) != PROTOCOL["direction_sha256"]:
        raise ValueError("accepted Gate 0B direction hash mismatch")
    for spec in PROTOCOL["cells"].values():
        if sha256_file(spec["path"]) != spec["sha256"]:
            raise ValueError(f"accepted Gate 0B cell hash mismatch: {spec['path']}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=PROTOCOL["model"])
    ap.add_argument("--revision", default=PROTOCOL["revision"])
    ap.add_argument("--tokenizer-revision", default=PROTOCOL["revision"])
    ap.add_argument("--component-set", default=PROTOCOL["component_set"])
    ap.add_argument("--direction", default=PROTOCOL["direction"])
    ap.add_argument("--run-mode", choices=("accepted", "exploratory"),
                    default="exploratory")
    ap.add_argument("--allowed-dirty", action="append", default=[])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out", default="results/gate0b_task_control_accepted_20260713")
    args = ap.parse_args(argv)

    try:
        _validate_binding(args)
        resolution = resolve_component_sets(
            roles=("positive_writers", "suppressors"), explicit={},
            set_key=args.component_set, model=args.model,
            direction_path=args.direction, run_mode=args.run_mode)
    except (ValueError, ComponentSetError) as exc:
        ap.error(str(exc))

    out = Path(args.out)
    output = out / "task_control.json"
    inputs = (args.direction,
              *(spec["path"] for spec in PROTOCOL["cells"].values()))
    try:
        contract = evaluate_run_contract(
            args.run_mode,
            revisions={
                "model": {
                    "requested": args.revision,
                    "resolution": resolve_hf_commit(
                        args.model, revision=args.revision),
                },
                "tokenizer": {
                    "requested": args.tokenizer_revision,
                    "resolution": resolve_hf_commit(
                        args.model, revision=args.tokenizer_revision),
                },
            },
            output_paths=(output,), source_rules=args.allowed_dirty,
            input_paths=inputs)
    except EvidenceRunError as exc:
        ap.error(str(exc))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.tokenizer_revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16,
        attn_implementation="eager").to(device)
    model.eval()
    direction = torch.tensor(np.load(args.direction), dtype=torch.float32,
                             device=device)
    direction /= torch.linalg.vector_norm(direction)

    pair_sets = {
        name: load_pairs(spec["path"])
        for name, spec in PROTOCOL["cells"].items()
    }
    families = {
        name: [pair["scenario_id"] for pair in pairs]
        for name, pairs in pair_sets.items()
    }

    def score(readout, pairs):
        if readout in ("raw_ab_dual", "chat_ab_dual"):
            values, per_order = choice_scores_dual_order(
                model, tokenizer, pairs, args.batch_size, args.max_tokens,
                device, chat=readout == "chat_ab_dual")
            return {"scores": values, "per_order": per_order}
        return {"scores": continuation_scores(
            model, tokenizer, pairs, args.batch_size, args.max_tokens, device)}

    def score_all():
        return {
            readout: {cell: score(readout, pairs)
                      for cell, pairs in pair_sets.items()}
            for readout in PROTOCOL["readouts"]
        }

    scores = {"baseline": score_all()}
    for condition, role in (("positive_writers_k2", "positive_writers"),
                            ("suppressors_k4", "suppressors")):
        components = [parse_component(value) for value in
                      resolution["sets"][role].split(",")]
        snapshot = snapshot_weights(model, components)
        try:
            for component in components:
                orthogonalize_component(model, component, direction)
            scores[condition] = score_all()
        finally:
            restore_weights(snapshot)
        restoration = assert_restored(snapshot)
        scores[condition]["weight_restoration"] = restoration

    analysis = analyze_gate0b(scores, families)
    cells = {
        name: {**spec, "families": families[name]}
        for name, spec in PROTOCOL["cells"].items()
    }
    artifact = {
        "schema": "empathy-action-probes/gate0b-task-control/1",
        "protocol": {**PROTOCOL, "readouts": list(PROTOCOL["readouts"]),
                     "primary_readouts": list(PROTOCOL["primary_readouts"])},
        "run_contract": contract,
        "component_sets": resolution,
        "direction": {"path": args.direction,
                      "sha256": sha256_file(args.direction)},
        "cells": cells,
        "scores": scores,
        "analysis": analysis,
        "config": {"device": device, "dtype": "bfloat16",
                   "attention": "eager", "batch_size": args.batch_size,
                   "max_tokens": args.max_tokens,
                   "tokenizer_padding_side": tokenizer.padding_side},
        "hf_revisions": resolve_model_and_tokenizer(
            args.model, revision=args.revision,
            tokenizer_revision=args.tokenizer_revision),
        "provenance": collect_run_provenance(),
    }
    try:
        revalidate_run_contract(contract, source_rules=args.allowed_dirty,
                                input_paths=inputs)
    except EvidenceRunError as exc:
        raise SystemExit(str(exc))
    atomic_write_json(output, artifact, require_fresh=True,
                      validate_fn=validate_artifact)
    print(json.dumps(analysis["writer_selectivity"], indent=2))
    print(f"all required gates pass: {analysis['all_required_gates_pass']}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
