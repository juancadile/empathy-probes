"""Build a deterministic synthetic artifact for WP2 end-to-end exercises.

The row plan is real, but activations are low-dimensional synthetic vectors.
One target feature separates the WP3 observation arms while independently
generated nuisance features carry WP1 contrasts. No model is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from extract_gate2_activations import candidate_blocks, load_records
from select_wp2_representation import DIAGNOSTICS, QUIET_CONTROLS


SEED = 904721
HIDDEN_SIZE = 16
TARGET_STRENGTH = {13: 0.7, 19: 1.4, 25: 0.8, 20: 1.1}


def stable_seed(*parts: object) -> int:
    encoded = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def arm_sign(arm: str, positive: str, negative: str) -> float:
    if arm == positive:
        return 1.0
    if arm == negative:
        return -1.0
    return 0.0


def build(output: Path) -> dict[str, object]:
    if output.exists():
        raise ValueError(f"refusing to overwrite synthetic fixture: {output}")
    output.mkdir(parents=True)
    rows = load_records("dev")
    blocks = candidate_blocks(42)
    activations = np.zeros((len(rows), len(blocks), HIDDEN_SIZE), dtype=np.float16)
    quiet_dims = {name: index + 2 for index, name in enumerate(QUIET_CONTROLS)}

    for row_index, row in enumerate(rows):
        family_rng = np.random.default_rng(stable_seed(SEED, row["family_id"], row["variant_id"]))
        family_base = family_rng.normal(0, 0.20, HIDDEN_SIZE)
        row_rng = np.random.default_rng(stable_seed(SEED, row_index, row["readout_role"]))
        for block_index, block in enumerate(blocks):
            vector = family_base + row_rng.normal(0, 0.035, HIDDEN_SIZE)
            if row["dataset"] == "wp3" and row["cell"] == "observation":
                vector[0] += TARGET_STRENGTH[block] * arm_sign(
                    str(row["arm_id"]), "current_actual", "archived_actual"
                )
            if row["dataset"] == "wp1" and row["contrast"] in QUIET_CONTROLS:
                positive, negative = QUIET_CONTROLS[str(row["contrast"])]
                vector[quiet_dims[str(row["contrast"])]] += 0.9 * arm_sign(
                    str(row["arm_id"]), positive, negative
                )
            if row["dataset"] == "wp1" and row["contrast"] in DIAGNOSTICS:
                positive, negative = DIAGNOSTICS[str(row["contrast"])]
                vector[0] += 0.8 * arm_sign(str(row["arm_id"]), positive, negative)
            activations[row_index, block_index] = vector.astype(np.float16)

    np.savez_compressed(
        output / "activations.npz",
        activations=activations,
        blocks=np.asarray(blocks, dtype=np.int16),
    )
    with (output / "rows.jsonl").open("w") as handle:
        for index, row in enumerate(rows):
            handle.write(json.dumps({"row_index": index, **row}) + "\n")
    plan = {
        "schema": "empathy-action-probes/gate2-activation-plan/1",
        "phase": "dev",
        "mode": "synthetic_fixture",
        "model": "synthetic-wp2-fixture",
        "target_model_loaded": False,
        "seed": SEED,
        "blocks": blocks,
        "hidden_size": HIDDEN_SIZE,
        "row_count": len(rows),
        "expected_outcome": "representation",
    }
    (output / "extraction_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.out), indent=2))


if __name__ == "__main__":
    main()
