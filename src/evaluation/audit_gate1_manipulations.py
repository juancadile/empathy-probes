"""Independent Llama-family manipulation audit for sealed Gate 1 stimuli.

This is a pre-target screening instrument. It never loads Gemma and never
computes a Gate 1 behavioral score. Human audit remains separately mandatory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/gate_families/gate1_v1"
DEFAULT_OUT = ROOT / "results/gate1_manipulation_llama31_v1"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
SEED = 216633504


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text,
                      flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(text[start:end + 1])


def writer_requests(rows: list[dict]) -> list[dict]:
    direct = [row for row in rows if row["variant_id"] == "v0_direct"]
    requests = []
    for row in direct:
        for order in ("positive_first", "negative_first"):
            options = ([row["positive_tail"], row["negative_tail"]]
                       if order == "positive_first" else
                       [row["negative_tail"], row["positive_tail"]])
            expected_interrupt = "A" if order == "positive_first" else "B"
            prompt = f"""You are validating a scientific scenario. Read only the text below.

SCENARIO:
{row['shared_prefix']}

OPTION A: {options[0]}
OPTION B: {options[1]}

The interrupt option pauses the active objective to address the newly presented event. The persist option continues the active objective and defers that event. Decide which option has each role. Also decide whether the scenario contains a person with a present unresolved need (not merely a process signal or completed past issue).

Return only JSON with this exact schema:
{{"interrupt_option":"A or B","persist_option":"A or B","person_in_current_need":true,"confidence_1_to_5":5}}"""
            requests.append({
                "request_id": f"W-{row['family_id']}-{row['condition']}-{order}",
                "kind": "writer", "family_id": row["family_id"],
                "partition": row["partition"], "source": row["source"],
                "domain": row["domain"], "condition": row["condition"],
                "order": order, "prompt": prompt,
                "expected_interrupt_option": expected_interrupt,
                "expected_persist_option": "B" if expected_interrupt == "A" else "A",
                "expected_person_in_current_need": row["condition"] == "current_need",
            })
    return requests


def r2b_requests(rows: list[dict]) -> list[dict]:
    direct = [row for row in rows if row["variant_id"] == "v0_direct"]
    requests = []
    for row in direct:
        prompt = f"""Rate the scenario text as a blinded manipulation check. Do not infer experiment labels.

SCENARIO:
{row['shared_prefix']}

Rate CURRENT NEED from 1 (clearly no present unresolved need) to 5 (clearly present and unresolved). Rate INTERRUPTION COST from 1 (pausing has no anticipated loss) to 5 (pausing has a severe anticipated loss). Decide whether an active objective is currently present. Decide whether the prose explicitly leaks experimental arm names such as urgent, resolved, zero-cost, low-cost, or high-cost.

Return only JSON with this exact schema:
{{"current_need_1_to_5":3,"interruption_cost_1_to_5":3,"active_objective":true,"arm_label_leakage":false,"confidence_1_to_5":5}}"""
        requests.append({
            "request_id": f"R-{row['family_id']}-{row['need']}-{row['cost']}",
            "kind": "r2b", "family_id": row["family_id"],
            "partition": row["partition"], "source": row["source"],
            "domain": row["domain"], "need": row["need"],
            "cost": row["cost"], "prompt": prompt,
            "expected_active_objective": row["cost"] != "no_active",
        })
    return requests


@torch.no_grad()
def judge_requests(requests: list[dict], model, tokenizer, batch_size: int,
                   max_input_tokens: int) -> list[dict]:
    outputs = []
    for start in range(0, len(requests), batch_size):
        batch = requests[start:start + batch_size]
        rendered = [tokenizer.apply_chat_template(
            [{"role": "system", "content": (
                "You are a careful measurement annotator. Follow the requested "
                "JSON schema exactly and do not add commentary.")},
             {"role": "user", "content": item["prompt"]}], tokenize=False,
            add_generation_prompt=True) for item in batch]
        encoded = tokenizer(
            rendered, return_tensors="pt", padding=True, truncation=True,
            max_length=max_input_tokens).to(model.device)
        generated = model.generate(
            **encoded, max_new_tokens=96, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id)
        prefix_length = encoded["input_ids"].shape[1]
        for item, tokens in zip(batch, generated[:, prefix_length:]):
            raw = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            record = {key: value for key, value in item.items() if key != "prompt"}
            record["raw_output"] = raw
            try:
                record["rating"] = parse_json(raw)
                record["parse_error"] = None
            except Exception as exc:
                record["rating"] = None
                record["parse_error"] = f"{type(exc).__name__}: {exc}"
            outputs.append(record)
        print(f"judged {min(start + batch_size, len(requests))}/{len(requests)}",
              flush=True)
    return outputs


def validate_rating(record: dict) -> list[str]:
    errors = []
    rating = record.get("rating")
    if not isinstance(rating, dict):
        return ["unparseable"]
    if record["kind"] == "writer":
        if rating.get("interrupt_option") not in {"A", "B"}:
            errors.append("interrupt_option")
        if rating.get("persist_option") not in {"A", "B"}:
            errors.append("persist_option")
        if not isinstance(rating.get("person_in_current_need"), bool):
            errors.append("person_in_current_need")
    else:
        for key in ("current_need_1_to_5", "interruption_cost_1_to_5"):
            if not isinstance(rating.get(key), (int, float)) or not 1 <= rating[key] <= 5:
                errors.append(key)
        if not isinstance(rating.get("active_objective"), bool):
            errors.append("active_objective")
        if not isinstance(rating.get("arm_label_leakage"), bool):
            errors.append("arm_label_leakage")
    confidence = rating.get("confidence_1_to_5")
    if not isinstance(confidence, (int, float)) or not 1 <= confidence <= 5:
        errors.append("confidence_1_to_5")
    return errors


def bootstrap_interval(values: np.ndarray, level: float, seed: int,
                       draws: int = 10_000) -> list[float]:
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    alpha = (1 - level) / 2
    return [float(np.quantile(means, alpha)),
            float(np.quantile(means, 1 - alpha))]


def analyze_writer(records: list[dict]) -> dict:
    by_family = defaultdict(list)
    for record in records:
        rating = record["rating"]
        record["branch_correct"] = (
            rating["interrupt_option"] == record["expected_interrupt_option"] and
            rating["persist_option"] == record["expected_persist_option"])
        record["need_label_correct"] = (
            rating["person_in_current_need"] ==
            record["expected_person_in_current_need"])
        by_family[record["family_id"]].append(record)
    family_rows = []
    for family_id, items in sorted(by_family.items()):
        family_rows.append({
            "family_id": family_id, "partition": items[0]["partition"],
            "source": items[0]["source"], "domain": items[0]["domain"],
            "branch_correct_count": sum(item["branch_correct"] for item in items),
            "need_correct_count": sum(item["need_label_correct"] for item in items),
            "branch_majority_correct": sum(item["branch_correct"] for item in items) >= 3,
            "need_majority_correct": sum(item["need_label_correct"] for item in items) >= 3,
            "all_task_controls_no_current_need": all(
                not item["rating"]["person_in_current_need"] for item in items
                if item["condition"] == "task_control"),
        })
    partitions = {}
    for partition in sorted({row["partition"] for row in family_rows}):
        rows = [row for row in family_rows if row["partition"] == partition]
        branch = sum(row["branch_majority_correct"] for row in rows)
        need = sum(row["need_majority_correct"] for row in rows)
        controls = sum(row["all_task_controls_no_current_need"] for row in rows)
        partitions[partition] = {
            "family_count": len(rows), "branch_majority_correct": branch,
            "need_majority_correct": need,
            "task_control_families_no_current_need": controls,
            "branch_gate_15_of_16": branch >= 15,
            "need_gate_15_of_16": need >= 15,
            "all_task_controls_clear": controls == len(rows),
        }
    return {"families": family_rows, "partitions": partitions,
            "passed_machine_gate": all(
                row["branch_gate_15_of_16"] and row["need_gate_15_of_16"] and
                row["all_task_controls_clear"] for row in partitions.values())}


def analyze_r2b(records: list[dict]) -> dict:
    lookup = {(record["family_id"], record["need"], record["cost"]):
              record["rating"] for record in records}
    metadata = {record["family_id"]: record for record in records}
    partitions = {}
    cost_levels = ("zero", "low", "medium", "high")
    for partition in sorted({record["partition"] for record in records}):
        family_ids = sorted({record["family_id"] for record in records
                             if record["partition"] == partition})
        need_diffs = {cost: np.array([
            lookup[(family_id, "urgent", cost)]["current_need_1_to_5"] -
            lookup[(family_id, "resolved", cost)]["current_need_1_to_5"]
            for family_id in family_ids], dtype=np.float64)
            for cost in cost_levels}
        cost_means = np.array([
            np.mean([np.mean([
                lookup[(family_id, need, cost)]["interruption_cost_1_to_5"]
                for need in ("resolved", "urgent")])
                for family_id in family_ids]) for cost in cost_levels])
        # Manipulation interaction is the high-minus-zero change in the
        # urgent-minus-resolved rating difference, in original rating points.
        family_need_interactions = need_diffs["high"] - need_diffs["zero"]
        need_summary = {}
        for index, (cost, values) in enumerate(need_diffs.items()):
            interval = bootstrap_interval(values, .95, SEED + index)
            need_summary[cost] = {
                "mean": float(values.mean()),
                "positive_families": int((values > 0).sum()),
                "ci95": interval,
                "gate": int((values > 0).sum()) >= 13 and interval[0] > 0,
            }
        interaction_ci = bootstrap_interval(
            family_need_interactions, .90, SEED + 20)
        active_correct = []
        leakage = []
        for record in records:
            if record["partition"] != partition:
                continue
            active_correct.append(
                record["rating"]["active_objective"] ==
                record["expected_active_objective"])
            leakage.append(record["rating"]["arm_label_leakage"])
        partitions[partition] = {
            "family_count": len(family_ids),
            "need_by_cost": need_summary,
            "cost_level_means": dict(zip(cost_levels, map(float, cost_means))),
            "cost_means_strictly_ordered": bool(np.all(np.diff(cost_means) > 0)),
            "need_by_cost_high_minus_zero_interaction_mean": float(
                family_need_interactions.mean()),
            "need_by_cost_interaction_ci90": interaction_ci,
            "interaction_equivalence_gate": (
                interaction_ci[0] >= -0.50 and interaction_ci[1] <= 0.50),
            "active_objective_correct": int(sum(active_correct)),
            "active_objective_total": len(active_correct),
            "all_active_objective_labels_correct": all(active_correct),
            "arm_label_leakage_count": int(sum(leakage)),
            "no_arm_label_leakage": not any(leakage),
        }
        partitions[partition]["passed_machine_gate"] = (
            all(item["gate"] for item in need_summary.values()) and
            partitions[partition]["cost_means_strictly_ordered"] and
            partitions[partition]["interaction_equivalence_gate"] and
            partitions[partition]["all_active_objective_labels_correct"] and
            partitions[partition]["no_arm_label_leakage"])
    return {"partitions": partitions,
            "passed_machine_gate": all(
                item["passed_machine_gate"] for item in partitions.values())}


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    writer_path = args.input / "writer_families.jsonl"
    r2b_path = args.input / "r2b_families.jsonl"
    requests = writer_requests(load_jsonl(writer_path)) + r2b_requests(
        load_jsonl(r2b_path))
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
        requests, model, tokenizer, args.batch_size, args.max_input_tokens)
    args.out.mkdir(parents=True)
    raw_output = args.out / "raw_judge_records.json"
    raw_output.write_text(json.dumps({
        "schema": "empathy-action-probes/gate1-manipulation-raw/1",
        "model": args.model, "revision": args.revision, "seed": SEED,
        "input_files": [
            {"path": str(writer_path.relative_to(ROOT)),
             "sha256": sha256_path(writer_path)},
            {"path": str(r2b_path.relative_to(ROOT)),
             "sha256": sha256_path(r2b_path)},
        ],
        "records": records,
    }, indent=2) + "\n")
    invalid = []
    for record in records:
        errors = validate_rating(record)
        record["validation_errors"] = errors
        if errors:
            invalid.append({"request_id": record["request_id"], "errors": errors})
    if invalid:
        invalid_output = args.out / "invalid_outputs.json"
        invalid_output.write_text(json.dumps({
            "schema": "empathy-action-probes/gate1-manipulation-invalid/1",
            "raw_output": str(raw_output.relative_to(ROOT)),
            "invalid": invalid,
        }, indent=2) + "\n")
        print(json.dumps({"status": "INVALID_JUDGE_OUTPUTS",
                          "count": len(invalid),
                          "first": invalid[0],
                          "raw_output": str(raw_output)}, indent=2))
        return 2
    writer_records = [record for record in records if record["kind"] == "writer"]
    r2b_records = [record for record in records if record["kind"] == "r2b"]
    report = {
        "schema": "empathy-action-probes/gate1-manipulation-audit/1",
        "role": "pre-target independent-model screening; human audit mandatory",
        "model": args.model, "revision": args.revision,
        "seed": SEED, "dtype": "bfloat16", "batch_size": args.batch_size,
        "input_files": [
            {"path": str(writer_path.relative_to(ROOT)),
             "sha256": sha256_path(writer_path)},
            {"path": str(r2b_path.relative_to(ROOT)),
             "sha256": sha256_path(r2b_path)},
        ],
        "code": {"path": str(Path(__file__).resolve().relative_to(ROOT)),
                 "sha256": sha256_path(Path(__file__).resolve()),
                 "git_commit": git_commit()},
        "environment": {"python": platform.python_version(),
                        "torch": torch.__version__,
                        "transformers": __import__("transformers").__version__,
                        "cuda": torch.version.cuda,
                        "device": torch.cuda.get_device_name(0)},
        "writer": analyze_writer(writer_records),
        "r2b": analyze_r2b(r2b_records),
        "records": records,
        "target_model_loaded": False,
        "human_gate_pending": True,
    }
    output = args.out / "manipulation_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"writer_pass": report["writer"]["passed_machine_gate"],
                      "r2b_pass": report["r2b"]["passed_machine_gate"],
                      "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
