"""Extract sealed Gate 2 residual-stream activations after the human gate.

`--dry-run` validates and records the extraction plan without importing or
loading the target model. A real development extraction requires either the
frozen human Gate 2 decision or a frozen discovery-only development lock.
Confirmation always requires the human decision and a frozen WP2 selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODEL = "google/gemma-2-9b-it"
REVISION = "11c9b309abf73637e4b6f9a3fa1e92e615547819"
ANCHORS = (0.33, 0.48, 0.62)
HISTORICAL_BLOCK = 20
INPUTS = {
    "wp1": ROOT / "data/gate_families/gate2_v2/wp1_families.jsonl",
    "wp3": ROOT / "data/gate_families/gate2_v2/wp3_families.jsonl",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_blocks(num_layers: int) -> list[int]:
    blocks = []
    for anchor in ANCHORS:
        block = min(
            range(num_layers),
            key=lambda index: (abs((index + 1) / num_layers - anchor), index),
        )
        blocks.append(block)
    if HISTORICAL_BLOCK < num_layers:
        blocks.append(HISTORICAL_BLOCK)
    return list(dict.fromkeys(blocks))


def load_records(phase: str) -> list[dict[str, object]]:
    partitions = {
        "dev": {"WP1-v2-dev", "WP3-v2-dev"},
        "confirm": {"WP1-v2-confirm", "WP3-v2-confirm"},
    }[phase]
    records = []
    for dataset, path in INPUTS.items():
        for line in path.read_text().splitlines():
            family = json.loads(line)
            if family["partition"] not in partitions:
                continue
            for arm in family["arms"]:
                base = {
                    "dataset": dataset,
                    "partition": family["partition"],
                    "family_id": family["family_id"],
                    "blueprint_family_id": family["blueprint_family_id"],
                    "variant_id": family["variant_id"],
                    "source": family["source"],
                    "domain": family["domain"],
                    "cell": family.get("cell"),
                    "contrast": family.get("contrast"),
                    "arm_id": arm["arm_id"],
                    "factors": arm.get("factors", {}),
                    "prompt": arm["prompt"],
                    "distress_quote": family.get("distress_quote", ""),
                }
                records.append({**base, "readout_role": "prompt_final"})
                if "quote_boundary" in family["readout_role"]:
                    records.append({**base, "readout_role": "quote_boundary"})
    return records


def quote_token_index(offsets, quote: str, prompt: str) -> int:
    if not quote:
        raise ValueError("quote-boundary row has no exact quote")
    needle = f'"{quote}"'
    start = prompt.find(needle)
    if start < 0 or prompt.find(needle, start + 1) >= 0:
        raise ValueError("exact quote must occur once in quote-boundary prompt")
    quote_end = start + 1 + len(quote)
    matches = [
        index for index, (left, right) in enumerate(offsets)
        if left < quote_end <= right and right > left
    ]
    if len(matches) != 1:
        raise ValueError(f"quote boundary maps to {len(matches)} tokens")
    return matches[0]


def validate_human_gate(path: Path) -> dict[str, object]:
    gate = json.loads(path.read_text())
    if gate.get("schema") != "empathy-action-probes/gate2-v2-human-gate/1":
        raise ValueError("unexpected Gate 2 human-gate schema")
    if gate.get("gate2_v2", {}).get("passed_human_gate") is not True:
        raise ValueError("Gate 2 human manipulation gate is not open")
    return gate


def validate_exploratory_dev_lock(path: Path) -> dict[str, object]:
    lock = json.loads(path.read_text())
    if lock.get("schema") != "empathy-action-probes/wp2-dev-exploratory-lock/1":
        raise ValueError("unexpected WP2 development exploratory-lock schema")
    expected = {
        "status": "frozen-before-target-extraction",
        "phase": "dev",
        "model": MODEL,
        "revision": REVISION,
        "human_gate_required_for_this_run": False,
        "confirmation_authorized": False,
        "claim_authorized": False,
    }
    for field, value in expected.items():
        if lock.get(field) != value:
            raise ValueError(f"exploratory lock field mismatch: {field}")
    if lock.get("candidate_blocks") != candidate_blocks(42):
        raise ValueError("exploratory lock candidate blocks mismatch")
    expected_inputs = {
        "wp1_sha256": sha256(INPUTS["wp1"]),
        "wp3_sha256": sha256(INPUTS["wp3"]),
    }
    if lock.get("inputs") != expected_inputs:
        raise ValueError("exploratory lock input hashes mismatch")
    bindings = (
        ("candidate_classes_and_selection_rule", "candidate_classes_and_selection_rule_sha256"),
        ("selector", "selector_sha256"),
    )
    for path_field, hash_field in bindings:
        bound_path = ROOT / str(lock.get(path_field, ""))
        if not bound_path.is_file() or lock.get(hash_field) != sha256(bound_path):
            raise ValueError(f"exploratory lock binding mismatch: {path_field}")
    return lock


def validate_selection_lock(path: Path) -> dict[str, object]:
    lock = json.loads(path.read_text())
    if lock.get("schema") != "empathy-action-probes/wp2-selection/1":
        raise ValueError("unexpected WP2 selection schema")
    if lock.get("selection", {}).get("outcome") != "representation":
        raise ValueError("confirmation cannot open after no-representation outcome")
    if lock.get("model") != MODEL or lock.get("revision") != REVISION:
        raise ValueError("selection lock model/revision mismatch")
    return lock


def plan(phase: str, num_layers: int, records: list[dict[str, object]]) -> dict[str, object]:
    blocks = candidate_blocks(num_layers)
    return {
        "schema": "empathy-action-probes/gate2-activation-plan/1",
        "phase": phase,
        "model": MODEL,
        "revision": REVISION,
        "num_layers": num_layers,
        "relative_depth_anchors": list(ANCHORS),
        "blocks": blocks,
        "historical_block": HISTORICAL_BLOCK,
        "row_count": len(records),
        "prompt_final_rows": sum(row["readout_role"] == "prompt_final" for row in records),
        "quote_boundary_rows": sum(row["readout_role"] == "quote_boundary" for row in records),
        "inputs": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in INPUTS.values()
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("dev", "confirm"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--human-gate", type=Path)
    parser.add_argument("--exploratory-dev-lock", type=Path)
    parser.add_argument("--selection-lock", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-layers", type=int, default=42,
                        help="dry-run architecture check; actual run verifies model config")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=768)
    args = parser.parse_args()

    records = load_records(args.phase)
    extraction_plan = plan(args.phase, args.num_layers, records)
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError(f"refusing to overwrite extraction output: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    plan_path = args.out / "extraction_plan.json"
    if args.dry_run:
        plan_path.write_text(json.dumps(extraction_plan, indent=2) + "\n")
        print(json.dumps(extraction_plan, indent=2))
        return

    exploratory_lock = None
    if args.exploratory_dev_lock is not None:
        if args.phase != "dev":
            raise ValueError("exploratory lock authorizes development extraction only")
        if args.human_gate is not None:
            raise ValueError("choose either human-gated or exploratory development mode")
        exploratory_lock = validate_exploratory_dev_lock(args.exploratory_dev_lock)
    else:
        if args.human_gate is None:
            raise ValueError("real extraction requires --human-gate or --exploratory-dev-lock")
        validate_human_gate(args.human_gate)
    if args.phase == "confirm":
        if args.selection_lock is None:
            raise ValueError("confirmation extraction requires --selection-lock")
        validate_selection_lock(args.selection_lock)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=REVISION,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda")
    model.eval()
    num_layers = model.config.num_hidden_layers
    if num_layers != args.num_layers:
        raise ValueError(f"model has {num_layers} layers; plan expected {args.num_layers}")
    blocks = candidate_blocks(num_layers)

    activations = np.empty((len(records), len(blocks), model.config.hidden_size), dtype=np.float16)
    with torch.no_grad():
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            encoded = tokenizer(
                [row["prompt"] for row in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_tokens,
                return_offsets_mapping=True,
            )
            offsets = encoded.pop("offset_mapping")
            inputs = {key: value.to("cuda") for key, value in encoded.items()}
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            for local_index, row in enumerate(batch):
                if row["readout_role"] == "prompt_final":
                    token_index = int(inputs["attention_mask"][local_index].sum().item()) - 1
                else:
                    token_index = quote_token_index(
                        offsets[local_index].tolist(), row["distress_quote"], row["prompt"]
                    )
                for site_index, block in enumerate(blocks):
                    activations[start + local_index, site_index] = (
                        outputs.hidden_states[block + 1][local_index, token_index]
                        .float().cpu().numpy().astype(np.float16)
                    )
            print(f"extracted {min(start + len(batch), len(records))}/{len(records)}", flush=True)

    np.savez_compressed(
        args.out / "activations.npz",
        activations=activations,
        blocks=np.asarray(blocks, dtype=np.int16),
    )
    with (args.out / "rows.jsonl").open("w") as handle:
        for index, row in enumerate(records):
            handle.write(json.dumps({"row_index": index, **row}) + "\n")
    extraction_plan.update({
        "hidden_size": model.config.hidden_size,
        "code": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "authorization_mode": "exploratory_dev" if exploratory_lock else "human_gate",
        "claim_authorized": False if exploratory_lock else True,
        "human_gate": (
            {"path": str(args.human_gate), "sha256": sha256(args.human_gate)}
            if args.human_gate else None
        ),
        "exploratory_dev_lock": (
            {
                "path": str(args.exploratory_dev_lock),
                "sha256": sha256(args.exploratory_dev_lock),
                "experiment_id": exploratory_lock["experiment_id"],
            }
            if exploratory_lock else None
        ),
        "selection_lock": (
            {"path": str(args.selection_lock), "sha256": sha256(args.selection_lock)}
            if args.selection_lock else None
        ),
        "artifacts": {
            "activations": "activations.npz",
            "activations_sha256": sha256(args.out / "activations.npz"),
            "rows": "rows.jsonl",
            "rows_sha256": sha256(args.out / "rows.jsonl"),
        },
    })
    plan_path.write_text(json.dumps(extraction_plan, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
