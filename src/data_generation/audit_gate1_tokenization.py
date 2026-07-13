"""Exact Gemma tokenizer audit for frozen Gate 1 decision tails.

This is a pre-model structural gate.  It loads only the tokenizer, verifies the
frozen model revision, and records every tail's exact token IDs and length.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
MODEL = "google/gemma-2-9b-it"
REVISION = "11c9b309abf73637e4b6f9a3fa1e92e615547819"


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def audit_file(path: Path, tokenizer) -> dict:
    rows = load_jsonl(path)
    pairs = {}
    failures = []
    for row in rows:
        positive = row["positive_tail"]
        negative = row["negative_tail"]
        key = (positive, negative)
        if key not in pairs:
            pos_ids = tokenizer.encode(positive, add_special_tokens=False)
            neg_ids = tokenizer.encode(negative, add_special_tokens=False)
            pos_bytes = len(positive.encode("utf-8"))
            neg_bytes = len(negative.encode("utf-8"))
            pairs[key] = {
                "positive_tail": positive,
                "negative_tail": negative,
                "positive_token_ids": pos_ids,
                "negative_token_ids": neg_ids,
                "positive_tokens": tokenizer.convert_ids_to_tokens(pos_ids),
                "negative_tokens": tokenizer.convert_ids_to_tokens(neg_ids),
                "token_length_difference": abs(len(pos_ids) - len(neg_ids)),
                "byte_relative_difference": abs(pos_bytes - neg_bytes) /
                    max(pos_bytes, neg_bytes),
            }
        metrics = pairs[key]
        if metrics["token_length_difference"] > 2:
            failures.append({"family_id": row["family_id"],
                             "variant_id": row["variant_id"],
                             "reason": "token_length_difference"})
        if metrics["byte_relative_difference"] > 0.10:
            failures.append({"family_id": row["family_id"],
                             "variant_id": row["variant_id"],
                             "reason": "byte_relative_difference"})
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_path(path),
        "record_count": len(rows),
        "family_count": len({row["family_id"] for row in rows}),
        "partition_counts": dict(Counter(row["partition"] for row in rows)),
        "unique_tail_pairs": list(pairs.values()),
        "failures": failures,
        "passed": not failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=ROOT / "data/gate_families/gate1_v1")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args(argv)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=False)
    reports = [audit_file(args.input / name, tokenizer)
               for name in ("writer_families.jsonl", "r2b_families.jsonl")]
    report = {
        "schema": "empathy-action-probes/gate1-tokenizer-audit/1",
        "model": args.model,
        "revision": args.revision,
        "tokenizer_class": type(tokenizer).__name__,
        "files": reports,
        "passed": all(item["passed"] for item in reports),
    }
    output = args.input / "tokenizer_audit.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"passed": report["passed"],
                      "files": [{"path": item["path"],
                                 "failures": len(item["failures"])}
                                for item in reports]}, indent=2))
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
