"""Repair only invalid Gate 2 judge outputs and preserve first-pass attempts."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .audit_gate1_manipulations import (
        MODEL, REVISION, git_commit, judge_requests, load_jsonl, sha256_path)
    from .audit_gate2_manipulations import (
        DEFAULT_INPUT, analyze_wp1, analyze_wp3, requests_for, validate)
except ImportError:
    from audit_gate1_manipulations import (
        MODEL, REVISION, git_commit, judge_requests, load_jsonl, sha256_path)
    from audit_gate2_manipulations import (
        DEFAULT_INPUT, analyze_wp1, analyze_wp3, requests_for, validate)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIRST_PASS = (
    ROOT / "results/gate2_manipulation_llama31_v1/raw_judge_records.json")
DEFAULT_OUT = ROOT / "results/gate2_manipulation_llama31_v1_repair"
SEED = 1154198998


def merge_repairs(first_records: list[dict], repair_records: list[dict]) -> list[dict]:
    repair_by_id = {record["request_id"]: record for record in repair_records}
    if len(repair_by_id) != len(repair_records):
        raise ValueError("duplicate repair request ID")
    merged = []
    for old in first_records:
        replacement = repair_by_id.pop(old["request_id"], None)
        if replacement is None:
            merged.append(old)
            continue
        replacement = dict(replacement)
        replacement["superseded_attempt"] = {
            "raw_output": old.get("raw_output"),
            "rating": old.get("rating"),
            "parse_error": old.get("parse_error"),
            "validation_errors": validate(old),
        }
        merged.append(replacement)
    if repair_by_id:
        raise ValueError(f"unknown repair IDs: {sorted(repair_by_id)[:3]}")
    return merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--first-pass", type=Path, default=DEFAULT_FIRST_PASS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-input-tokens", type=int, default=1280)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--scale-reminder", action="store_true",
                        help="append the original 1-5 boundary rule to repairs")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")

    first = json.loads(args.first_pass.read_text())
    first_records = first["records"]
    invalid_ids = {record["request_id"] for record in first_records
                   if validate(record)}
    if not invalid_ids:
        raise ValueError("first pass has no invalid records")
    wp1_path = args.input / "wp1_families.jsonl"
    wp3_path = args.input / "wp3_families.jsonl"
    all_requests = requests_for(load_jsonl(wp1_path), "wp1") + requests_for(
        load_jsonl(wp3_path), "wp3")
    request_map = {request["request_id"]: request for request in all_requests}
    missing = invalid_ids - request_map.keys()
    if missing:
        raise ValueError(f"missing requests: {sorted(missing)[:3]}")
    repair_requests = [request_map[request_id] for request_id in sorted(invalid_ids)]
    if args.scale_reminder:
        for request in repair_requests:
            request["prompt"] += (
                "\n\nSCALE REMINDER: Every numeric rating must be between 1 and "
                "5 inclusive. Use 1, never 0, when a property is absent.")

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
    repaired = judge_requests(
        repair_requests, model, tokenizer, args.batch_size,
        args.max_input_tokens, max_new_tokens=args.max_new_tokens)
    merged = merge_repairs(first_records, repaired)
    remaining = [{"request_id": record["request_id"], "errors": validate(record)}
                 for record in merged if validate(record)]

    args.out.mkdir(parents=True)
    raw_path = args.out / "raw_judge_records.json"
    raw_path.write_text(json.dumps({
        "schema": "empathy-action-probes/gate2-manipulation-repaired-raw/1",
        "model": args.model, "revision": args.revision, "seed": SEED,
        "first_pass": {"path": str(args.first_pass.relative_to(ROOT)),
                       "sha256": sha256_path(args.first_pass),
                       "invalid_count": len(invalid_ids)},
        "repair": {"request_count": len(repaired),
                   "max_new_tokens": args.max_new_tokens,
                   "scale_reminder": args.scale_reminder},
        "records": merged,
    }, indent=2) + "\n")
    if remaining:
        (args.out / "invalid_outputs.json").write_text(json.dumps({
            "invalid": remaining, "raw_path": str(raw_path.relative_to(ROOT))
        }, indent=2) + "\n")
        print(json.dumps({"status": "INVALID_AFTER_REPAIR",
                          "count": len(remaining), "first": remaining[0]}, indent=2))
        return 2

    wp1 = [record for record in merged if record["kind"] == "wp1"]
    wp3 = [record for record in merged if record["kind"] == "wp3"]
    report = {
        "schema": "empathy-action-probes/gate2-manipulation-audit/1",
        "role": "pre-target independent-model screening; human audit mandatory",
        "model": args.model, "revision": args.revision, "seed": SEED,
        "input_files": [
            {"path": str(wp1_path.relative_to(ROOT)), "sha256": sha256_path(wp1_path)},
            {"path": str(wp3_path.relative_to(ROOT)), "sha256": sha256_path(wp3_path)},
        ],
        "first_pass": {"path": str(args.first_pass.relative_to(ROOT)),
                       "sha256": sha256_path(args.first_pass),
                       "invalid_count": len(invalid_ids)},
        "repair": {"request_count": len(repaired),
                   "max_new_tokens": args.max_new_tokens,
                   "scale_reminder": args.scale_reminder},
        "code": {"path": str(Path(__file__).resolve().relative_to(ROOT)),
                 "sha256": sha256_path(Path(__file__).resolve()),
                 "git_commit": git_commit()},
        "environment": {"python": platform.python_version(),
                        "torch": torch.__version__,
                        "transformers": __import__("transformers").__version__,
                        "cuda": torch.version.cuda,
                        "device": torch.cuda.get_device_name(0)},
        "wp1": analyze_wp1(wp1), "wp3": analyze_wp3(wp3),
        "records": merged, "target_model_loaded": False,
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
