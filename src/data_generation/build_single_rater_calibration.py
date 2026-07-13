"""Materialize the frozen single-rater human-calibration packet.

The annotator receives two blinded forms. Design labels, stratum membership,
duplicate links, and gold expectations remain in a private key. Sampling is
deterministic and family-preserving where a contrast requires paired arms.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/gate_families/human_calibration_single_rater_20260713"
PROTOCOL = ROOT / "notes/HUMAN_CALIBRATION_PROTOCOL_2026-07-13.md"
SEED = 1973658643
MIN_RETEST_LAG = 10

GATE2_PACKET = ROOT / "data/gate_families/gate2_v2/human_audit/gate2_v2_packet.csv"
GATE2_KEY = ROOT / "data/gate_families/gate2_v2/human_audit/gate2_v2_key.csv"
R2B_PACKET = ROOT / "data/gate_families/gate1_r2b_v2/human_audit/r2b_v2_packet.csv"
R2B_KEY = ROOT / "data/gate_families/gate1_r2b_v2/human_audit/r2b_v2_key.csv"
WP1 = ROOT / "data/gate_families/gate2_v2/wp1_families.jsonl"
WP3 = ROOT / "data/gate_families/gate2_v2/wp3_families.jsonl"

S2_ARMS = {
    "observation": {
        "current_actual", "archived_actual", "current_fictional", "archived_fictional"
    },
    "persona": {"current_neutral", "current_caring", "neutral_neutral", "neutral_caring"},
    "cost": {"high", "zero"},
    "resolved_neutral_controls": {"resolved_current", "neutral_current"},
}
S3_LABELS = {"P_new", "Spos_new", "L_new"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def join_packet(packet_path: Path, key_path: Path) -> list[dict[str, object]]:
    packets = {row["audit_id"]: row for row in read_csv(packet_path)}
    keys = {row["audit_id"]: row for row in read_csv(key_path)}
    if packets.keys() != keys.keys():
        raise ValueError(f"packet/key mismatch: {packet_path}")
    return [
        {"source_audit_id": audit_id, "packet": packets[audit_id], "key": keys[audit_id]}
        for audit_id in packets
    ]


def family_metadata(path: Path) -> dict[str, dict[str, str]]:
    output = {}
    for row in read_jsonl(path):
        output[str(row["family_id"])] = {
            "source": str(row["source"]),
            "partition": str(row["partition"]),
        }
    return output


def choose_one_family_per_source(
    family_ids: set[str], metadata: dict[str, dict[str, str]], rng: np.random.Generator
) -> list[str]:
    by_source: dict[str, list[str]] = defaultdict(list)
    for family in sorted(family_ids):
        by_source[metadata[family]["source"]].append(family)
    chosen = []
    for source in sorted(by_source):
        options = by_source[source]
        chosen.append(options[int(rng.integers(0, len(options)))])
    if len(chosen) != 4:
        raise ValueError(f"expected four source strata, got {len(chosen)}")
    return chosen


def select_gate2(rows: list[dict[str, object]], rng: np.random.Generator):
    wp1_meta, wp3_meta = family_metadata(WP1), family_metadata(WP3)

    s1 = [
        row for row in rows
        if row["key"]["label"] == "Ctext_new" and row["key"]["arm_id"] in {"zero", "low"}
    ]

    wp3_rows = [row for row in rows if row["key"]["family_id"] in wp3_meta]
    s2_families = choose_one_family_per_source(
        {str(row["key"]["family_id"]) for row in wp3_rows}, wp3_meta, rng
    )
    s2 = [
        row for row in wp3_rows
        if row["key"]["family_id"] in s2_families
        and row["key"]["label"] in S2_ARMS
        and row["key"]["arm_id"] in S2_ARMS[row["key"]["label"]]
    ]

    wp1_rows = [row for row in rows if row["key"]["family_id"] in wp1_meta]
    s3_families = choose_one_family_per_source(
        {str(row["key"]["family_id"]) for row in wp1_rows}, wp1_meta, rng
    )
    s3 = [
        row for row in wp1_rows
        if row["key"]["family_id"] in s3_families and row["key"]["label"] in S3_LABELS
    ]

    selected = []
    for name, items, expected in (("S1", s1, 16), ("S2", s2, 48), ("S3", s3, 32)):
        if len(items) != expected:
            raise ValueError(f"{name} expected {expected} rows, got {len(items)}")
        for row in items:
            selected.append({**row, "stratum": name})
    if len({row["source_audit_id"] for row in selected}) != 96:
        raise ValueError("Gate 2 strata overlap")
    return selected, {"S2": s2_families, "S3": s3_families}


def choose_form_b_duplicates(rows: list[dict[str, object]], rng: np.random.Generator):
    by_family: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["key"]["need"] == "resolved" and row["key"]["cost"] in {"zero", "high"}:
            by_family[row["key"]["family_id"]].append(row)
    forced = []
    for index, family in enumerate(sorted(by_family)):
        target = "zero" if index % 2 == 0 else "high"
        forced.append(next(row for row in by_family[family] if row["key"]["cost"] == target))
    remaining = [row for row in rows if row not in forced]
    extra_indices = rng.choice(len(remaining), size=8, replace=False)
    chosen = forced + [remaining[int(index)] for index in extra_indices]
    if len(chosen) != 24 or len({row["source_audit_id"] for row in chosen}) != 24:
        raise ValueError("Form B duplicate selection failed")
    return chosen


def choose_form_a_duplicates(rows: list[dict[str, object]], rng: np.random.Generator):
    counts = {"S1": 4, "S2": 6, "S3": 5}
    chosen = []
    for stratum, count in counts.items():
        options = [row for row in rows if row["stratum"] == stratum]
        indices = rng.choice(len(options), size=count, replace=False)
        chosen.extend(options[int(index)] for index in indices)
    return chosen


def ordered_presentations(
    originals: list[dict[str, object]], duplicates: list[dict[str, object]], rng: np.random.Generator
) -> list[dict[str, object]]:
    duplicate_ids = {row["source_audit_id"] for row in duplicates}
    for _ in range(10000):
        order = list(rng.permutation(len(originals)))
        arranged = [originals[int(index)] for index in order]
        positions = {row["source_audit_id"]: index for index, row in enumerate(arranged)}
        if all(positions[audit_id] <= len(arranged) - MIN_RETEST_LAG - 1 for audit_id in duplicate_ids):
            break
    else:
        raise ValueError("could not place duplicate originals early enough")

    output = [{**row, "presentation": "original"} for row in arranged]
    for duplicate in rng.permutation(duplicates):
        source_id = duplicate["source_audit_id"]
        parent = next(i for i, row in enumerate(output) if row["source_audit_id"] == source_id)
        insertion = int(rng.integers(parent + MIN_RETEST_LAG, len(output) + 1))
        output.insert(insertion, {**duplicate, "presentation": "retest"})
    return output


def gold_metadata(row: dict[str, object]) -> tuple[str, str, str]:
    if row["key"].get("label") != "P_new":
        return "", "", ""
    arm = row["key"]["arm_id"]
    if arm == "caring":
        return "persona_instruction", "persona_caring_1_to_5", "4:5"
    if arm == "neutral":
        return "persona_instruction", "persona_caring_1_to_5", "1:2"
    return "", "", ""


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def materialize_form(
    form: str,
    originals: list[dict[str, object]],
    duplicates: list[dict[str, object]],
    output: Path,
    rng: np.random.Generator,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    presentations = ordered_presentations(originals, duplicates, rng)
    id_by_presentation = {
        index: f"HC-{form}-{index:04d}" for index in range(len(presentations))
    }
    original_output_id = {
        row["source_audit_id"]: id_by_presentation[index]
        for index, row in enumerate(presentations) if row["presentation"] == "original"
    }
    packet_rows, key_rows = [], []
    for index, row in enumerate(presentations):
        output_id = id_by_presentation[index]
        packet_rows.append({**row["packet"], "audit_id": output_id})
        gold_kind, gold_field, gold_range = gold_metadata(row)
        key_rows.append({
            "audit_id": output_id,
            "form": form,
            "source_audit_id": row["source_audit_id"],
            "presentation": row["presentation"],
            "duplicate_of_audit_id": (
                original_output_id[row["source_audit_id"]]
                if row["presentation"] == "retest" else ""
            ),
            "stratum": row.get("stratum", "R2b-census"),
            "gold_kind": gold_kind,
            "gold_rating_field": gold_field,
            "gold_expected_range": gold_range,
            **{key: value for key, value in row["key"].items() if key != "audit_id"},
        })
    packet_path = output / "annotator" / f"form_{form}.csv"
    write_csv(packet_path, packet_rows, list(packet_rows[0]))
    return packet_rows, key_rows


def build(output: Path = OUTPUT) -> dict[str, object]:
    if output.exists():
        raise ValueError(f"refusing to overwrite calibration packet: {output}")
    (output / "annotator").mkdir(parents=True)
    (output / "private_do_not_send").mkdir()
    rng = np.random.default_rng(SEED)

    gate2 = join_packet(GATE2_PACKET, GATE2_KEY)
    r2b = join_packet(R2B_PACKET, R2B_KEY)
    form_a, selected_families = select_gate2(gate2, rng)
    form_b = [{**row, "stratum": "R2b-census"} for row in r2b]
    dup_a = choose_form_a_duplicates(form_a, rng)
    dup_b = choose_form_b_duplicates(form_b, rng)

    packet_a, key_a = materialize_form("A", form_a, dup_a, output, rng)
    packet_b, key_b = materialize_form("B", form_b, dup_b, output, rng)
    private_rows = key_a + key_b
    private_fields = list(private_rows[0])
    for row in private_rows:
        for field in row:
            if field not in private_fields:
                private_fields.append(field)
    private_path = output / "private_do_not_send" / "combined_key.csv"
    write_csv(private_path, private_rows, private_fields)

    readme = output / "annotator" / "README.md"
    readme.write_text(
        "# Blinded Human Calibration\n\n"
        "Complete Form B first. Complete Form A afterward if time permits. Do not inspect "
        "files outside this `annotator` directory. Some items may recur; rate every row "
        "independently without comparing it to earlier answers.\n\n"
        "Use integer ratings from 1 (clearly absent/low) to 5 (clearly present/high). "
        "Mark `active_objective_yes_no` as `yes` or `no`. Use `notes` only when the text "
        "is ambiguous or a requested rating cannot be made.\n\n"
        "Form B separates the quoted speaker's current need from the interruption cost "
        "to the narrator's active objective. Do not infer one rating from the other.\n"
    )

    manifest = {
        "schema": "empathy-action-probes/single-rater-calibration-packet/1",
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": sha256(PROTOCOL),
        "seed": SEED,
        "rng": "numpy.random.PCG64",
        "minimum_retest_lag_rows": MIN_RETEST_LAG,
        "forms": {
            "A": {
                "canonical_rows": 96,
                "retest_rows": 15,
                "total_rows": len(packet_a),
                "strata": {"S1": 16, "S2": 48, "S3": 32},
                "selected_families": selected_families,
                "selected_source_audit_ids": {
                    stratum: [row["source_audit_id"] for row in form_a if row["stratum"] == stratum]
                    for stratum in ("S1", "S2", "S3")
                },
                "retest_source_audit_ids": [row["source_audit_id"] for row in dup_a],
                "gold_unique_items": sum(bool(gold_metadata(row)[0]) for row in form_a),
            },
            "B": {
                "canonical_rows": 160,
                "retest_rows": 24,
                "total_rows": len(packet_b),
                "retest_source_audit_ids": [row["source_audit_id"] for row in dup_b],
            },
        },
        "sources": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (GATE2_PACKET, GATE2_KEY, R2B_PACKET, R2B_KEY, WP1, WP3)
        },
        "artifacts": {
            "annotator/form_A.csv": sha256(output / "annotator/form_A.csv"),
            "annotator/form_B.csv": sha256(output / "annotator/form_B.csv"),
            "annotator/README.md": sha256(readme),
            "private_do_not_send/combined_key.csv": sha256(private_path),
        },
        "blinding": {
            "annotator_files_contain_design_labels": False,
            "duplicate_links_private": True,
            "gold_expectations_private": True,
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "manifest.sha256").write_text(f"{sha256(manifest_path)}  manifest.json\n")
    return manifest


def main() -> None:
    print(json.dumps(build(), indent=2))


if __name__ == "__main__":
    main()
