"""Primary Qwen-family manipulation audit for the repaired R2b-v2 batch."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .audit_gate1_manipulations import (
        analyze_r2b, git_commit, judge_requests, load_jsonl, r2b_requests,
        sha256_path, validate_rating)
except ImportError:
    from audit_gate1_manipulations import (
        analyze_r2b, git_commit, judge_requests, load_jsonl, r2b_requests,
        sha256_path, validate_rating)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/gate_families/gate1_r2b_v2/r2b_families.jsonl"
DEFAULT_OUT = ROOT / "results/gate1_r2b_v2_manipulation_qwen3_14b"
MODEL = "Qwen/Qwen3-14B"
REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"
SEED = 3010918835


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")

    rows = load_jsonl(args.input)
    requests = r2b_requests(rows)
    order = np.random.Generator(np.random.PCG64(SEED)).permutation(len(requests))
    requests = [requests[int(index)] for index in order]

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
        max_new_tokens=128, chat_template_kwargs={"enable_thinking": False})

    args.out.mkdir(parents=True)
    raw_path = args.out / "raw_judge_records.json"
    raw_path.write_text(json.dumps({
        "schema": "empathy-action-probes/gate1-r2b-v2-manipulation-raw/1",
        "model": args.model, "revision": args.revision, "seed": SEED,
        "input": {"path": str(args.input.relative_to(ROOT)),
                  "sha256": sha256_path(args.input)},
        "records": records,
    }, indent=2) + "\n")
    invalid = []
    for record in records:
        errors = validate_rating(record)
        record["validation_errors"] = errors
        if errors:
            invalid.append({"request_id": record["request_id"],
                            "errors": errors})
    if invalid:
        (args.out / "invalid_outputs.json").write_text(json.dumps({
            "invalid": invalid, "raw_path": str(raw_path.relative_to(ROOT))
        }, indent=2) + "\n")
        print(json.dumps({"status": "INVALID_JUDGE_OUTPUTS",
                          "count": len(invalid), "first": invalid[0]}, indent=2))
        return 2

    report = {
        "schema": "empathy-action-probes/gate1-r2b-v2-manipulation-audit/1",
        "role": "primary pre-target Qwen-family screen; human audit mandatory",
        "model": args.model, "revision": args.revision, "seed": SEED,
        "input": {"path": str(args.input.relative_to(ROOT)),
                  "sha256": sha256_path(args.input)},
        "code": {"path": str(Path(__file__).resolve().relative_to(ROOT)),
                 "sha256": sha256_path(Path(__file__).resolve()),
                 "git_commit": git_commit()},
        "environment": {"python": platform.python_version(),
                        "torch": torch.__version__,
                        "transformers": __import__("transformers").__version__,
                        "cuda": torch.version.cuda,
                        "device": torch.cuda.get_device_name(0)},
        "r2b": analyze_r2b(records, seed=SEED),
        "records": records,
        "target_model_loaded": False,
        "human_gate_pending": True,
        "llama_sensitivity_pending": True,
    }
    output = args.out / "manipulation_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["r2b"]["passed_machine_gate"],
                      "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
