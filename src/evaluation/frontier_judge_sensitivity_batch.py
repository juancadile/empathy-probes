"""Run the precommitted Opus 4.8 sensitivity checks through Message Batches.

This runner cannot alter Gate 1 or Gate 2. R2b must be collected before the
Gate 2 batch can be submitted. Exact frozen local-judge prompts are reused.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_gate1_manipulations import (  # noqa: E402
    load_jsonl,
    parse_json,
    validate_rating,
)
from audit_gate1_r2b_v2 import qwen_r2b_requests  # noqa: E402
from audit_gate2_v2_pairwise import (  # noqa: E402
    analyze as analyze_gate2,
    degeneracy_reason as gate2_degeneracy_reason,
    requests_for as gate2_requests_for,
    validation_errors as gate2_validation_errors,
)


MODEL = "claude-opus-4-8"
SYSTEM = (
    "You are a careful measurement annotator. Follow the requested JSON "
    "schema exactly and do not add commentary."
)
BOOTSTRAP_SEED = 1666403293
LOCK = ROOT / "notes/FRONTIER_JUDGE_SENSITIVITY_LOCK_2026-07-13.md"
OUT = ROOT / "results/frontier_judge_sensitivity_opus48_20260713"
R2B_INPUT = ROOT / "data/gate_families/gate1_r2b_v2/r2b_families.jsonl"
R2B_PACKET = ROOT / "data/gate_families/gate1_r2b_v2/human_audit/r2b_v2_packet.csv"
R2B_KEY = ROOT / "data/gate_families/gate1_r2b_v2/human_audit/r2b_v2_key.csv"
GATE2_INPUT = ROOT / "data/gate_families/gate2_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def require_committed_protocol() -> None:
    paths = [LOCK, Path(__file__).resolve()]
    relative = [str(path.relative_to(ROOT)) for path in paths]
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *relative], cwd=ROOT, text=True
    ).strip()
    if dirty:
        raise ValueError("sensitivity lock and runner must be committed before submission")


def load_api_key() -> None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    env = ROOT / ".env"
    for line in env.read_text().splitlines():
        if line.startswith("ANTHROPIC_API_KEY="):
            os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
            return
    raise ValueError("ANTHROPIC_API_KEY not found")


def r2b_requests() -> list[dict[str, object]]:
    generated = {
        (request["family_id"], request["need"], request["cost"]): request
        for request in qwen_r2b_requests(load_jsonl(R2B_INPUT))
    }
    packet = {row["audit_id"]: row for row in read_csv(R2B_PACKET)}
    requests = []
    for index, key in enumerate(read_csv(R2B_KEY)):
        request = dict(generated[(key["family_id"], key["need"], key["cost"])])
        scenario = request["prompt"].split("SCENARIO:\n", 1)[1].split(
            "\n\nRate CURRENT NEED", 1
        )[0]
        if scenario != packet[key["audit_id"]]["prompt"]:
            raise ValueError(f"R2b packet prompt mismatch: {key['audit_id']}")
        requests.append({
            **request,
            "custom_id": f"r2b-{index:04d}",
            "audit_id": key["audit_id"],
        })
    if len(requests) != 160 or len({item["custom_id"] for item in requests}) != 160:
        raise ValueError("R2b sensitivity set must contain 160 unique requests")
    return requests


def gate2_requests() -> list[dict[str, object]]:
    requests = gate2_requests_for(
        load_jsonl(GATE2_INPUT / "wp1_families.jsonl"),
        load_jsonl(GATE2_INPUT / "wp3_families.jsonl"),
    )
    output = [{**request, "custom_id": f"gate2-{index:04d}"}
              for index, request in enumerate(requests)]
    if len(output) != 848 or len({item["request_id"] for item in output}) != 848:
        raise ValueError("Gate 2 sensitivity set must contain 848 unique requests")
    return output


def request_manifest(kind: str, requests: list[dict[str, object]]) -> dict[str, object]:
    inputs = [R2B_INPUT, R2B_PACKET, R2B_KEY] if kind == "r2b" else [
        GATE2_INPUT / "wp1_families.jsonl",
        GATE2_INPUT / "wp3_families.jsonl",
    ]
    return {
        "schema": "empathy-action-probes/frontier-judge-request-manifest/1",
        "role": "sensitivity only; cannot alter any gate",
        "kind": kind,
        "model": MODEL,
        "system": SYSTEM,
        "temperature": 0,
        "lock": {"path": str(LOCK.relative_to(ROOT)), "sha256": sha256(LOCK)},
        "code": {"path": str(Path(__file__).resolve().relative_to(ROOT)),
                 "sha256": sha256(Path(__file__).resolve())},
        "git_commit": git_commit(),
        "inputs": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
                   for path in inputs],
        "request_count": len(requests),
        "requests": requests,
    }


def batch_requests(kind: str, requests: list[dict[str, object]]):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    max_tokens = 128 if kind == "r2b" else 64
    return [
        Request(
            custom_id=str(request["custom_id"]),
            params=MessageCreateParamsNonStreaming(
                model=MODEL,
                max_tokens=max_tokens,
                temperature=0,
                system=SYSTEM,
                messages=[{"role": "user", "content": str(request["prompt"])}],
            ),
        )
        for request in requests
    ]


def output_dir(kind: str) -> Path:
    return OUT / kind


def submit(client, kind: str) -> None:
    require_committed_protocol()
    if kind == "gate2" and not (output_dir("r2b") / "sensitivity.json").exists():
        raise ValueError("collect and report R2b before submitting Gate 2")
    directory = output_dir(kind)
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite {directory}")
    requests = r2b_requests() if kind == "r2b" else gate2_requests()
    manifest = request_manifest(kind, requests)
    batch = client.messages.batches.create(requests=batch_requests(kind, requests))
    directory.mkdir(parents=True)
    (directory / "request_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    receipt = {
        "schema": "empathy-action-probes/frontier-judge-batch-receipt/1",
        "batch_id": batch.id,
        "processing_status": batch.processing_status,
        "request_counts": batch.request_counts.model_dump(mode="json"),
        "model": MODEL,
        "request_manifest_sha256": sha256(directory / "request_manifest.json"),
    }
    (directory / "submission.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def retrieve_batch(client, kind: str):
    receipt = json.loads((output_dir(kind) / "submission.json").read_text())
    return client.messages.batches.retrieve(receipt["batch_id"])


def status(client, kind: str) -> None:
    batch = retrieve_batch(client, kind)
    print(json.dumps({
        "batch_id": batch.id,
        "processing_status": batch.processing_status,
        "request_counts": batch.request_counts.model_dump(mode="json"),
    }, indent=2))


def collect_raw(client, kind: str) -> list[dict[str, object]]:
    directory = output_dir(kind)
    raw_path = directory / "raw_results.json"
    if raw_path.exists():
        return json.loads(raw_path.read_text())["records"]
    batch = retrieve_batch(client, kind)
    if batch.processing_status != "ended":
        raise ValueError(f"batch is not ended: {batch.processing_status}")
    manifest = json.loads((directory / "request_manifest.json").read_text())
    by_id = {item["custom_id"]: item for item in manifest["requests"]}
    records, errors = [], []
    for result in client.messages.batches.results(batch.id):
        if result.result.type != "succeeded":
            errors.append({"custom_id": result.custom_id, "type": result.result.type})
            continue
        message = result.result.message
        raw = "".join(block.text for block in message.content if block.type == "text").strip()
        item = {key: value for key, value in by_id[result.custom_id].items() if key != "prompt"}
        try:
            rating, parse_error = parse_json(raw), None
        except Exception as exc:
            rating, parse_error = None, f"{type(exc).__name__}: {exc}"
        records.append({
            **item,
            "raw_output": raw,
            "rating": rating,
            "parse_error": parse_error,
            "returned_model": message.model,
            "usage": message.usage.model_dump(mode="json"),
        })
    missing = sorted(set(by_id) - {record["custom_id"] for record in records} -
                     {error["custom_id"] for error in errors})
    payload = {
        "schema": "empathy-action-probes/frontier-judge-raw/1",
        "role": "sensitivity only; cannot alter any gate",
        "kind": kind,
        "batch_id": batch.id,
        "requested_model": MODEL,
        "records": sorted(records, key=lambda row: row["custom_id"]),
        "errors": errors,
        "missing": missing,
    }
    raw_path.write_text(json.dumps(payload, indent=2) + "\n")
    if errors or missing or len(records) != len(by_id):
        raise ValueError("batch has missing or errored results")
    returned = {record["returned_model"] for record in records}
    if returned != {MODEL}:
        raise ValueError(f"returned-model mismatch: {returned}")
    return payload["records"]


def bootstrap_ci(values: np.ndarray, level: float = 0.90) -> list[float]:
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    sampled = values[rng.integers(0, len(values), size=(10_000, len(values)))]
    alpha = (1 - level) / 2
    return [float(np.quantile(sampled.mean(axis=1), alpha)),
            float(np.quantile(sampled.mean(axis=1), 1 - alpha))]


def classify_r2b_part(mean: float, ci90: list[float]) -> str:
    if ci90[0] >= -0.50 and ci90[1] <= 0.50:
        return "equivalently_flat"
    if mean > 0.50 and ci90[0] > 0:
        return "positively_rising"
    return "inconclusive"


def analyze_r2b(records: list[dict[str, object]]) -> dict[str, object]:
    invalid = []
    for record in records:
        errors = validate_rating(record)
        if errors:
            invalid.append({"custom_id": record["custom_id"], "errors": errors})
    if invalid:
        raise ValueError(f"invalid R2b outputs: {invalid[:3]}")
    if len({record["raw_output"] for record in records}) == 1:
        raise ValueError("degenerate byte-identical R2b outputs")
    lookup = {
        (record["partition"], record["family_id"], record["need"], record["cost"]):
        record["rating"] for record in records
    }
    partitions = {}
    for partition in sorted({str(record["partition"]) for record in records}):
        families = sorted({str(record["family_id"]) for record in records
                           if record["partition"] == partition})
        resolved = {
            cost: float(np.mean([
                lookup[(partition, family, "resolved", cost)]["current_need_1_to_5"]
                for family in families
            ])) for cost in ("zero", "low", "medium", "high")
        }
        urgent = {
            cost: float(np.mean([
                lookup[(partition, family, "urgent", cost)]["current_need_1_to_5"]
                for family in families
            ])) for cost in ("zero", "low", "medium", "high")
        }
        deltas = np.asarray([
            lookup[(partition, family, "resolved", "high")]["current_need_1_to_5"] -
            lookup[(partition, family, "resolved", "zero")]["current_need_1_to_5"]
            for family in families
        ], dtype=float)
        mean, ci90 = float(deltas.mean()), bootstrap_ci(deltas)
        partitions[partition] = {
            "family_count": len(families),
            "resolved_need_means": resolved,
            "urgent_need_means": urgent,
            "resolved_high_minus_zero_family_values": deltas.tolist(),
            "resolved_high_minus_zero_mean": mean,
            "resolved_high_minus_zero_ci90": ci90,
            "precommitted_classification": classify_r2b_part(mean, ci90),
        }
    classes = {item["precommitted_classification"] for item in partitions.values()}
    if classes == {"equivalently_flat"}:
        overall = "judge_entanglement_corroborated"
    elif classes == {"positively_rising"}:
        overall = "stimulus_confound_corroborated"
    else:
        overall = "mixed_or_inconclusive"
    return {"partitions": partitions, "precommitted_overall": overall}


def check_key(item: dict[str, object]) -> tuple[object, ...]:
    return (item["kind"], item["partition"], item["label"], item["check"])


def compare_gate2(report: dict[str, object]) -> dict[str, object]:
    paths = {
        "qwen_primary": ROOT / "results/gate2_v2_pairwise_qwen3_14b/pairwise_audit.json",
        "llama_sensitivity": ROOT / "results/gate2_v2_pairwise_llama31_8b/pairwise_audit.json",
    }
    current = {check_key(item): item for item in report["checks"]}
    comparisons = {}
    for name, path in paths.items():
        prior = json.loads(path.read_text())["analysis"]["checks"]
        previous = {check_key(item): item for item in prior}
        if set(previous) != set(current):
            raise ValueError(f"Gate 2 check set differs from {name}")
        comparisons[name] = {
            "artifact": str(path.relative_to(ROOT)),
            "artifact_sha256": sha256(path),
            "pass_status_agreement": sum(
                current[key]["passed"] == previous[key]["passed"] for key in current
            ),
            "check_count": len(current),
            "disagreements": [
                {"key": list(key), "frontier_passed": current[key]["passed"],
                 "prior_passed": previous[key]["passed"]}
                for key in sorted(current)
                if current[key]["passed"] != previous[key]["passed"]
            ],
        }
    return comparisons


def collect(client, kind: str) -> None:
    directory = output_dir(kind)
    report_path = directory / "sensitivity.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")
    records = collect_raw(client, kind)
    if kind == "r2b":
        analysis = analyze_r2b(records)
        comparisons = None
    else:
        invalid = [{"custom_id": record["custom_id"],
                    "errors": gate2_validation_errors(record)}
                   for record in records if gate2_validation_errors(record)]
        if invalid:
            raise ValueError(f"invalid Gate 2 outputs: {invalid[:3]}")
        degeneracy = gate2_degeneracy_reason(records)
        if degeneracy:
            raise ValueError(degeneracy)
        analysis = analyze_gate2(records)
        comparisons = compare_gate2(analysis)
    report = {
        "schema": "empathy-action-probes/frontier-judge-sensitivity/1",
        "role": "sensitivity only; cannot alter any gate",
        "kind": kind,
        "model": MODEL,
        "lock": {"path": str(LOCK.relative_to(ROOT)), "sha256": sha256(LOCK)},
        "request_manifest_sha256": sha256(directory / "request_manifest.json"),
        "raw_results_sha256": sha256(directory / "raw_results.json"),
        "analysis": analysis,
        "prior_model_comparisons": comparisons,
        "human_gate_pending": True,
        "target_model_loaded": False,
        "gate_effect": "none",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def client():
    load_api_key()
    import anthropic
    return anthropic.Anthropic()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("submit", "status", "collect"))
    parser.add_argument("kind", choices=("r2b", "gate2"))
    args = parser.parse_args()
    api = client()
    if args.command == "submit":
        submit(api, args.kind)
    elif args.command == "status":
        status(api, args.kind)
    else:
        collect(api, args.kind)


if __name__ == "__main__":
    main()
