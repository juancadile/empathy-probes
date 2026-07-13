"""Independent Llama-family manipulation audit for WP1 and WP3.

This script rates every family before any Gemma activation is opened. It is a
screening instrument; the separately blinded human audit remains mandatory.
"""

from __future__ import annotations

import argparse
import json
import platform
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .audit_gate1_manipulations import (
        MODEL, REVISION, bootstrap_interval, git_commit, judge_requests,
        load_jsonl, sha256_path)
except ImportError:
    from audit_gate1_manipulations import (
        MODEL, REVISION, bootstrap_interval, git_commit, judge_requests,
        load_jsonl, sha256_path)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/gate_families/gate2_v1"
DEFAULT_OUT = ROOT / "results/gate2_manipulation_llama31_v1"
SEED = 1154198998
RATING_KEYS = (
    "currentness_1_to_5", "actuality_1_to_5",
    "welfare_relevance_1_to_5", "distress_content_1_to_5",
    "warmth_1_to_5", "persona_caring_1_to_5", "motive_genuine_1_to_5",
    "valence_positive_1_to_5", "social_salience_1_to_5",
    "response_opportunity_1_to_5", "task_pressure_1_to_5",
    "task_persistence_1_to_5",
)


def requests_for(records: list[dict], kind: str) -> list[dict]:
    requests = []
    for record in records:
        if record["variant_id"] != "v0_direct":
            continue
        label = record.get("contrast", record.get("cell"))
        for item in record["arms"]:
            continuation = item.get("continuation", "")
            factors = item.get("factors", {})
            candidate_tails = ""
            if factors.get("response_tail"):
                candidate_tails = (
                    f"\nCANDIDATE RESPONSE: {factors['response_tail']}"
                    f"\nCANDIDATE TASK CONTINUATION: {factors['task_tail']}")
            prompt = f"""Rate the blinded scenario exactly as written. Ratings are properties of the prompt, not endorsements.

SCENARIO:
{item['prompt']}
{('CONTINUATION: ' + continuation) if continuation else ''}{candidate_tails}

Use 1 (clearly absent/low) through 5 (clearly present/high) for: currentness of a person's unresolved state; stipulated actuality of the person; relevance to a person's present welfare; amount of distress content; linguistic warmth; caring-assistant persona; genuineness of an attributed motive; positive valence; social salience; opportunity for the model to respond; pressure from an active objective; and persistence with the active objective. Also state whether an active objective is present.

Return only JSON with exactly these keys:
{{"currentness_1_to_5":3,"actuality_1_to_5":3,"welfare_relevance_1_to_5":3,"distress_content_1_to_5":3,"warmth_1_to_5":3,"persona_caring_1_to_5":3,"motive_genuine_1_to_5":3,"valence_positive_1_to_5":3,"social_salience_1_to_5":3,"response_opportunity_1_to_5":3,"task_pressure_1_to_5":3,"task_persistence_1_to_5":3,"active_objective":true,"confidence_1_to_5":5}}"""
            requests.append({
                "request_id": f"{kind}-{record['family_id']}-{label}-{item['arm_id']}",
                "kind": kind, "family_id": record["family_id"],
                "partition": record["partition"], "source": record["source"],
                "domain": record["domain"], "label": label,
                "arm_id": item["arm_id"], "factors": factors,
                "prompt": prompt,
            })
    return requests


def validate(record: dict) -> list[str]:
    rating = record.get("rating")
    if not isinstance(rating, dict):
        return ["unparseable"]
    errors = []
    for key in RATING_KEYS + ("confidence_1_to_5",):
        value = rating.get(key)
        if not isinstance(value, (int, float)) or not 1 <= value <= 5:
            errors.append(key)
    if not isinstance(rating.get("active_objective"), bool):
        errors.append("active_objective")
    return errors


def indexed(records: list[dict]):
    return {(record["family_id"], record["label"], record["arm_id"]):
            record["rating"] for record in records}


def effect(index, family_id, label, high_arm, low_arm, key):
    return (index[(family_id, label, high_arm)][key] -
            index[(family_id, label, low_arm)][key])


def summarize_effect(values, threshold, seed):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()),
            "ci95": bootstrap_interval(values, .95, seed),
            "positive_families": int((values > 0).sum()),
            "threshold": threshold,
            "passes_mean_threshold": float(values.mean()) >= threshold}


def analyze_wp1(records: list[dict]) -> dict:
    index = indexed(records)
    meta = {record["family_id"]: record for record in records}
    binary = {
        "T_new": ("persist", "interrupt", "task_persistence_1_to_5"),
        "D_new": ("warm", "terse", "warmth_1_to_5"),
        "P_new": ("caring", "neutral", "persona_caring_1_to_5"),
        "G_new": ("genuine", "strategic", "motive_genuine_1_to_5"),
        "Spos_new": ("positive", "neutral", "valence_positive_1_to_5"),
        "R_new": ("welfare_observation", "nonwelfare_observation",
                  "welfare_relevance_1_to_5"),
        "O_new": ("available", "unavailable", "response_opportunity_1_to_5"),
    }
    partitions = {}
    for part_index, partition in enumerate(sorted({r["partition"] for r in records})):
        families = sorted(f for f, item in meta.items() if item["partition"] == partition)
        contrasts = {}
        for offset, (label, (high, low, key)) in enumerate(binary.items()):
            values = [effect(index, family, label, high, low, key)
                      for family in families]
            target = summarize_effect(values, 1.0, SEED + 100 * part_index + offset)
            non_target = {}
            for other in RATING_KEYS:
                if other == key:
                    continue
                diffs = [effect(index, family, label, high, low, other)
                         for family in families]
                non_target[other] = float(np.mean(diffs))
            target["non_target_mean_differences"] = non_target
            target["non_target_equivalence_max_abs"] = max(
                abs(value) for value in non_target.values())
            target["non_target_equivalence_gate"] = (
                target["non_target_equivalence_max_abs"] <= .30)
            contrasts[label] = target

        ctext_means = {level: float(np.mean([
            index[(family, "Ctext_new", level)]["task_pressure_1_to_5"]
            for family in families])) for level in ("zero", "low", "high")}
        l_current = [effect(index, family, "L_new", "current_actual",
                            "archived_actual", "currentness_1_to_5")
                     for family in families]
        l_actual = [effect(index, family, "L_new", "current_actual",
                           "current_simulated", "actuality_1_to_5")
                    for family in families]
        l_distress_ranges = [max(
            index[(family, "L_new", arm)]["distress_content_1_to_5"]
            for arm in ("current_actual", "archived_actual", "current_simulated")) - min(
            index[(family, "L_new", arm)]["distress_content_1_to_5"]
            for arm in ("current_actual", "archived_actual", "current_simulated"))
            for family in families]
        b_active = [
            index[(family, "B_new", "active_zero_cost")]["active_objective"] and
            not index[(family, "B_new", "no_active_objective")]["active_objective"]
            for family in families]
        contrasts["Ctext_new"] = {
            "means": ctext_means,
            "strictly_ordered": ctext_means["zero"] < ctext_means["low"] < ctext_means["high"],
            "high_minus_zero": ctext_means["high"] - ctext_means["zero"],
            "passes": (ctext_means["zero"] < ctext_means["low"] < ctext_means["high"] and
                       ctext_means["high"] - ctext_means["zero"] >= 1.0),
        }
        contrasts["L_new"] = {
            "currentness": summarize_effect(l_current, 1.0, SEED + 50 + part_index),
            "actuality": summarize_effect(l_actual, 1.0, SEED + 60 + part_index),
            "mean_distress_range": float(np.mean(l_distress_ranges)),
            "distress_equivalence_gate": float(np.mean(l_distress_ranges)) <= .30,
        }
        contrasts["B_new"] = {
            "active_objective_correct": int(sum(b_active)),
            "family_count": len(families), "passes": all(b_active)}
        binary_pass = all(
            item["passes_mean_threshold"] and item["non_target_equivalence_gate"]
            for label, item in contrasts.items() if label in binary)
        partitions[partition] = {
            "family_count": len(families), "contrasts": contrasts,
            "passed_machine_gate": (
                binary_pass and contrasts["Ctext_new"]["passes"] and
                contrasts["L_new"]["currentness"]["passes_mean_threshold"] and
                contrasts["L_new"]["actuality"]["passes_mean_threshold"] and
                contrasts["L_new"]["distress_equivalence_gate"] and
                contrasts["B_new"]["passes"]),
        }
    return {"partitions": partitions,
            "passed_machine_gate": all(p["passed_machine_gate"]
                                       for p in partitions.values())}


def analyze_wp3(records: list[dict]) -> dict:
    index = indexed(records)
    meta = {record["family_id"]: record for record in records}
    partitions = {}
    for part_index, partition in enumerate(sorted({r["partition"] for r in records})):
        families = sorted(f for f, item in meta.items() if item["partition"] == partition)
        current_effect = [effect(index, family, "observation", "current_actual",
                                 "archived_actual", "welfare_relevance_1_to_5")
                          for family in families]
        actuality_effect = [effect(index, family, "observation", "current_actual",
                                   "current_simulated", "actuality_1_to_5")
                            for family in families]
        distress_ranges = [max(
            index[(family, "observation", arm)]["distress_content_1_to_5"]
            for arm in ("current_actual", "current_simulated", "archived_actual",
                        "archived_simulated")) - min(
            index[(family, "observation", arm)]["distress_content_1_to_5"]
            for arm in ("current_actual", "current_simulated", "archived_actual",
                        "archived_simulated")) for family in families]
        task_ranges = [max(
            index[(family, "observation", arm)]["task_pressure_1_to_5"]
            for arm in ("current_actual", "current_simulated", "archived_actual",
                        "archived_simulated")) - min(
            index[(family, "observation", arm)]["task_pressure_1_to_5"]
            for arm in ("current_actual", "current_simulated", "archived_actual",
                        "archived_simulated")) for family in families]
        opportunity = [np.mean([
            effect(index, family, "agency", f"{status}_available",
                   f"{status}_unavailable", "response_opportunity_1_to_5")
            for status in ("current", "archived")]) for family in families]
        opportunity_interactions = [
            effect(index, family, "agency", "current_available",
                   "current_unavailable", "response_opportunity_1_to_5") -
            effect(index, family, "agency", "archived_available",
                   "archived_unavailable", "response_opportunity_1_to_5")
            for family in families]
        persona = [np.mean([
            effect(index, family, "persona", f"{content}_caring",
                   f"{content}_neutral", "persona_caring_1_to_5")
            for content in ("current", "archived", "neutral")])
            for family in families]
        n_by_persona = {persona_name: [
            effect(index, family, "persona", f"current_{persona_name}",
                   f"archived_{persona_name}", "welfare_relevance_1_to_5")
            for family in families] for persona_name in ("neutral", "caring")}
        cost_means = {level: float(np.mean([
            index[(family, "cost", level)]["task_pressure_1_to_5"]
            for family in families])) for level in ("zero", "low", "high")}
        summary = {
            "family_count": len(families),
            "current_welfare": summarize_effect(
                current_effect, 1.5, SEED + part_index),
            "actuality": summarize_effect(
                actuality_effect, 1.0, SEED + 10 + part_index),
            "mean_distress_range": float(np.mean(distress_ranges)),
            "distress_equivalence_gate": float(np.mean(distress_ranges)) <= .30,
            "mean_task_pressure_range": float(np.mean(task_ranges)),
            "task_pressure_equivalence_gate": float(np.mean(task_ranges)) <= .30,
            "opportunity": summarize_effect(
                opportunity, 1.5, SEED + 20 + part_index),
            "opportunity_interaction_mean": float(np.mean(opportunity_interactions)),
            "opportunity_flat_across_N_gate": abs(float(np.mean(
                opportunity_interactions))) <= .30,
            "persona": summarize_effect(persona, 1.0, SEED + 30 + part_index),
            "N_within_neutral_persona": summarize_effect(
                n_by_persona["neutral"], 1.5, SEED + 40 + part_index),
            "N_within_caring_persona": summarize_effect(
                n_by_persona["caring"], 1.5, SEED + 50 + part_index),
            "cost_means": cost_means,
            "cost_gate": (cost_means["zero"] < cost_means["low"] < cost_means["high"] and
                          cost_means["high"] - cost_means["zero"] >= 1.0),
        }
        summary["passed_machine_gate"] = (
            summary["current_welfare"]["passes_mean_threshold"] and
            summary["actuality"]["passes_mean_threshold"] and
            summary["distress_equivalence_gate"] and
            summary["task_pressure_equivalence_gate"] and
            summary["opportunity"]["passes_mean_threshold"] and
            summary["opportunity_flat_across_N_gate"] and
            summary["persona"]["passes_mean_threshold"] and
            summary["N_within_neutral_persona"]["passes_mean_threshold"] and
            summary["N_within_caring_persona"]["passes_mean_threshold"] and
            summary["cost_gate"])
        partitions[partition] = summary
    return {"partitions": partitions,
            "passed_machine_gate": all(p["passed_machine_gate"]
                                       for p in partitions.values())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-input-tokens", type=int, default=1280)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    wp1_path = args.input / "wp1_families.jsonl"
    wp3_path = args.input / "wp3_families.jsonl"
    requests = requests_for(load_jsonl(wp1_path), "wp1") + requests_for(
        load_jsonl(wp3_path), "wp3")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16,
        attn_implementation="eager").to("cuda")
    model.eval()
    torch.manual_seed(SEED)
    records = judge_requests(
        requests, model, tokenizer, args.batch_size, args.max_input_tokens,
        max_new_tokens=180)
    args.out.mkdir(parents=True)
    raw_path = args.out / "raw_judge_records.json"
    raw_path.write_text(json.dumps({
        "schema": "empathy-action-probes/gate2-manipulation-raw/1",
        "model": args.model, "revision": args.revision, "seed": SEED,
        "input_files": [
            {"path": str(wp1_path.relative_to(ROOT)), "sha256": sha256_path(wp1_path)},
            {"path": str(wp3_path.relative_to(ROOT)), "sha256": sha256_path(wp3_path)},
        ], "records": records}, indent=2) + "\n")
    invalid = []
    for record in records:
        errors = validate(record)
        record["validation_errors"] = errors
        if errors:
            invalid.append({"request_id": record["request_id"], "errors": errors})
    if invalid:
        path = args.out / "invalid_outputs.json"
        path.write_text(json.dumps({"invalid": invalid,
                                   "raw_path": str(raw_path.relative_to(ROOT))},
                                  indent=2) + "\n")
        print(json.dumps({"status": "INVALID_JUDGE_OUTPUTS",
                          "count": len(invalid), "first": invalid[0]}, indent=2))
        return 2
    wp1 = [record for record in records if record["kind"] == "wp1"]
    wp3 = [record for record in records if record["kind"] == "wp3"]
    report = {
        "schema": "empathy-action-probes/gate2-manipulation-audit/1",
        "role": "pre-target independent-model screening; human audit mandatory",
        "model": args.model, "revision": args.revision, "seed": SEED,
        "input_files": [
            {"path": str(wp1_path.relative_to(ROOT)), "sha256": sha256_path(wp1_path)},
            {"path": str(wp3_path.relative_to(ROOT)), "sha256": sha256_path(wp3_path)},
        ],
        "code": {"path": str(Path(__file__).resolve().relative_to(ROOT)),
                 "sha256": sha256_path(Path(__file__).resolve()),
                 "git_commit": git_commit()},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "transformers": __import__("transformers").__version__,
                        "cuda": torch.version.cuda,
                        "device": torch.cuda.get_device_name(0)},
        "wp1": analyze_wp1(wp1), "wp3": analyze_wp3(wp3),
        "records": records, "target_model_loaded": False,
        "human_gate_pending": True,
    }
    output = args.out / "manipulation_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"wp1_pass": report["wp1"]["passed_machine_gate"],
                      "wp3_pass": report["wp3"]["passed_machine_gate"],
                      "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
