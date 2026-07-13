"""Bind Gate 2 v2 structural controls to the pinned Gemma tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
MODEL = "google/gemma-2-9b-it"
REVISION = "11c9b309abf73637e4b6f9a3fa1e92e615547819"
DEFAULT_INPUT = ROOT / "data/gate_families/gate2_v2"


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def prompt_ids(tokenizer, arm: dict) -> list[int]:
    return tokenizer.encode(arm["prompt"], add_special_tokens=False)


def audit_file(path: Path, tokenizer) -> dict:
    rows = load_jsonl(path)
    failures = []
    ranges: dict[str, list[int]] = defaultdict(list)
    response_tails: set[tuple[int, ...]] = set()
    task_tails: set[tuple[int, ...]] = set()
    for row in rows:
        label = row.get("contrast", row.get("cell"))
        lengths = [len(prompt_ids(tokenizer, arm)) for arm in row["arms"]]
        ranges[label].append(max(lengths) - min(lengths))
        for arm, length in zip(row["arms"], lengths, strict=True):
            if length > 512:
                failures.append({"family_id": row["family_id"], "label": label,
                                 "arm_id": arm["arm_id"], "reason": "over_512"})
            factors = arm.get("factors", {})
            if factors.get("response_tail"):
                response_tails.add(tuple(tokenizer.encode(
                    factors["response_tail"], add_special_tokens=False)))
                task_tails.add(tuple(tokenizer.encode(
                    factors["task_tail"], add_special_tokens=False)))

        exact_all = ((row["experiment"] == "gate2_wp1_v2" and
                      label in {"D_new", "G_new", "P_new", "L_new", "O_new"})
                     or (row["experiment"] == "gate2_wp3_v2" and
                         label in {"observation", "agency"}))
        if exact_all and len(set(lengths)) != 1:
            failures.append({"family_id": row["family_id"], "label": label,
                             "variant_id": row["variant_id"],
                             "lengths": lengths, "reason": "position_mismatch"})
        if label == "T_new" and max(lengths) - min(lengths) > 2:
            failures.append({"family_id": row["family_id"], "label": label,
                             "variant_id": row["variant_id"],
                             "lengths": lengths, "reason": "position_mismatch"})
        if label == "persona":
            by_content: dict[str, list[int]] = defaultdict(list)
            for arm, length in zip(row["arms"], lengths, strict=True):
                by_content[arm["factors"]["content"]].append(length)
            if any(len(set(values)) != 1 for values in by_content.values()):
                failures.append({"family_id": row["family_id"], "label": label,
                                 "variant_id": row["variant_id"],
                                 "reason": "persona_position_mismatch"})
        if label in {"L_new", "observation"}:
            quote = row["distress_quote"]
            serialized = json.dumps(quote, ensure_ascii=False)
            span_sequences = []
            for arm in row["arms"]:
                prompt = arm["prompt"]
                start = prompt.index(serialized)
                end = start + len(serialized)
                encoded = tokenizer(
                    prompt, add_special_tokens=False,
                    return_offsets_mapping=True)
                span_ids = tuple(token_id for token_id, (left, right) in zip(
                    encoded["input_ids"], encoded["offset_mapping"], strict=True)
                    if right > start and left < end)
                span_sequences.append(span_ids)
                if not span_ids:
                    failures.append({"family_id": row["family_id"],
                                     "label": label, "arm_id": arm["arm_id"],
                                     "reason": "quote_token_sequence_missing"})
            if len(set(span_sequences)) != 1:
                failures.append({"family_id": row["family_id"],
                                 "label": label,
                                 "variant_id": row["variant_id"],
                                 "reason": "quote_token_sequence_mismatch"})
    if len(response_tails) > 1 or len(task_tails) > 1:
        failures.append({"reason": "candidate_tail_token_mismatch"})
    return {
        "path": str(path.relative_to(ROOT)), "sha256": sha256_path(path),
        "record_count": len(rows),
        "family_count": len({row["family_id"] for row in rows}),
        "prompt_token_range_by_label": {
            key: {"max": max(values), "mean": sum(values) / len(values)}
            for key, values in sorted(ranges.items())},
        "response_tail_token_ids": list(next(iter(response_tails), ())),
        "task_tail_token_ids": list(next(iter(task_tails), ())),
        "failures": failures, "passed": not failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args(argv)
    args.input = args.input.resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=False)
    reports = [audit_file(args.input / name, tokenizer)
               for name in ("wp1_families.jsonl", "wp3_families.jsonl")]
    report = {
        "schema": "empathy-action-probes/gate2-v2-tokenizer-audit/1",
        "model": args.model, "revision": args.revision,
        "tokenizer_class": type(tokenizer).__name__, "files": reports,
        "passed": all(item["passed"] for item in reports),
    }
    output = args.input / "tokenizer_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"],
                      "failures": [len(item["failures"]) for item in reports]},
                     indent=2))
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
