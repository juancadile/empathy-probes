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

R2B_PACKET = ROOT / "data/gate_families/gate1_r2b_v2/human_audit/r2b_v2_packet.csv"
R2B_KEY = ROOT / "data/gate_families/gate1_r2b_v2/human_audit/r2b_v2_key.csv"
WP1 = ROOT / "data/gate_families/gate2_v2/wp1_families.jsonl"
WP3 = ROOT / "data/gate_families/gate2_v2/wp3_families.jsonl"

FAILED_CONTROL_ARMS = {
    "B_new": {"active_zero_cost", "no_active_objective"},
    "Spos_new": {"positive", "neutral"},
    "O_new": {"available", "unavailable"},
    "Ctext_new": {"zero", "low", "medium", "high"},
}
PACKET_FIELDS = [
    "audit_id", "kind", "prompt", "continuation", "currentness_1_to_5",
    "actuality_1_to_5", "welfare_relevance_1_to_5", "distress_content_1_to_5",
    "warmth_1_to_5", "persona_caring_1_to_5", "motive_genuine_1_to_5",
    "valence_positive_1_to_5", "social_salience_1_to_5",
    "response_opportunity_1_to_5", "task_pressure_1_to_5",
    "task_persistence_1_to_5", "active_objective_yes_no", "notes",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def blank_packet(kind: str, prompt: str, continuation: str = "") -> dict[str, str]:
    row = {field: "" for field in PACKET_FIELDS}
    row.update({"kind": kind, "prompt": prompt, "continuation": continuation})
    return row


def materialized_audit_rows(path: Path, kind: str) -> list[dict[str, object]]:
    output = []
    for record in read_jsonl(path):
        if record["variant_id"] != "v0_direct":
            continue
        label = str(record.get("contrast", record.get("cell")))
        for arm in record["arms"]:
            output.append({
                "source_audit_id": (
                    f"{record['family_id']}::{label}::{arm['arm_id']}::v0_direct"
                ),
                "packet": blank_packet(
                    kind, str(arm["prompt"]), str(arm.get("continuation", ""))
                ),
                "key": {
                    "family_id": str(record["family_id"]),
                    "label": label,
                    "arm_id": str(arm["arm_id"]),
                    "partition": str(record["partition"]),
                    "source": str(record["source"]),
                },
            })
    return output


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


def choose_families_per_source(
    family_ids: set[str], metadata: dict[str, dict[str, str]], count: int,
    rng: np.random.Generator,
) -> list[str]:
    by_source: dict[str, list[str]] = defaultdict(list)
    for family in sorted(family_ids):
        by_source[metadata[family]["source"]].append(family)
    chosen = []
    for source in sorted(by_source):
        options = by_source[source]
        if len(options) < count:
            raise ValueError(f"source {source} has only {len(options)} families")
        indices = rng.choice(len(options), size=count, replace=False)
        chosen.extend(options[int(index)] for index in sorted(indices))
    if len(chosen) != 4 * count:
        raise ValueError(f"expected {4 * count} source-balanced families, got {len(chosen)}")
    return chosen


def select_gate2(rng: np.random.Generator):
    wp1_meta, wp3_meta = family_metadata(WP1), family_metadata(WP3)
    wp1_rows = materialized_audit_rows(WP1, "wp1")
    wp3_rows = materialized_audit_rows(WP3, "wp3")
    dev_wp1 = [row for row in wp1_rows if row["key"]["partition"] == "WP1-v2-dev"]

    selected = []
    selected_families: dict[str, list[str]] = {}
    for label in FAILED_CONTROL_ARMS:
        label_rows = [row for row in dev_wp1 if row["key"]["label"] == label]
        all_families = {str(row["key"]["family_id"]) for row in label_rows}
        families = (
            sorted(all_families) if label == "B_new"
            else choose_families_per_source(all_families, wp1_meta, 2, rng)
        )
        expected_families = 16 if label == "B_new" else 8
        if len(families) != expected_families:
            raise ValueError(f"{label} family selection failed")
        rows = [
            row for row in label_rows
            if row["key"]["family_id"] in families
            and row["key"]["arm_id"] in FAILED_CONTROL_ARMS[label]
        ]
        expected_rows = expected_families * len(FAILED_CONTROL_ARMS[label])
        if len(rows) != expected_rows:
            raise ValueError(f"{label} expected {expected_rows} rows, got {len(rows)}")
        selected.extend({**row, "stratum": f"FAILED-{label}"} for row in rows)
        selected_families[label] = families

    # The target itself must be humanly valid. Audit every development family
    # on the exact current-actual versus archived-actual contrast used by WP2.
    observation = [
        row for row in wp3_rows
        if row["key"]["partition"] == "WP3-v2-dev"
        and row["key"]["label"] == "observation"
        and row["key"]["arm_id"] in {"current_actual", "archived_actual"}
    ]
    if len(observation) != 32 or len({row["key"]["family_id"] for row in observation}) != 16:
        raise ValueError("WP3 development observation census is incomplete")
    selected.extend({**row, "stratum": "TARGET-WP3-observation"} for row in observation)
    selected_families["WP3_observation"] = sorted({
        str(row["key"]["family_id"]) for row in observation
    })

    # Form A still needs its own competence checks. Select one complete P_new
    # development family per source; these rows are gold checks, not evidence.
    persona_rows = [
        row for row in dev_wp1 if row["key"]["label"] == "P_new"
    ]
    gold_families = choose_one_family_per_source(
        {str(row["key"]["family_id"]) for row in persona_rows}, wp1_meta, rng
    )
    gold = [row for row in persona_rows if row["key"]["family_id"] in gold_families]
    if len(gold) != 8:
        raise ValueError(f"P_new gold expected 8 rows, got {len(gold)}")

    selected.extend({**row, "stratum": "GOLD-P_new"} for row in gold)
    selected_families["gold_P_new"] = gold_families
    if len(selected) != 136 or len({row["source_audit_id"] for row in selected}) != 136:
        raise ValueError("Gate 2 failed-control census overlaps or is incomplete")
    return selected, selected_families


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
    chosen = []
    # Every arm of every failed WP2 quietness gate receives a hidden retest.
    for label, arms in FAILED_CONTROL_ARMS.items():
        for arm in sorted(arms):
            options = [
                row for row in rows
                if row["key"]["label"] == label and row["key"]["arm_id"] == arm
            ]
            chosen.append(options[int(rng.integers(0, len(options)))])

    # B_new's inverse loading is the most consequential signed diagnostic, so
    # give each B arm one additional independently selected retest.
    for arm in sorted(FAILED_CONTROL_ARMS["B_new"]):
        options = [
            row for row in rows
            if row["key"]["label"] == "B_new" and row["key"]["arm_id"] == arm
            and row not in chosen
        ]
        chosen.append(options[int(rng.integers(0, len(options)))])

    # The target contrast receives one retest per arm.
    for arm in ("current_actual", "archived_actual"):
        options = [
            row for row in rows
            if row["key"]["label"] == "observation" and row["key"]["arm_id"] == arm
        ]
        chosen.append(options[int(rng.integers(0, len(options)))])

    # Retest both gold-arm types as a stability check on the competence items.
    for arm in ("caring", "neutral"):
        options = [
            row for row in rows
            if row["key"]["label"] == "P_new" and row["key"]["arm_id"] == arm
        ]
        chosen.append(options[int(rng.integers(0, len(options)))])

    # Bring Form A to approximately 15% retests with one additional item from
    # every failed cell, without changing the contrast coverage above.
    for label in FAILED_CONTROL_ARMS:
        options = [
            row for row in rows
            if row["key"]["label"] == label and row not in chosen
        ]
        chosen.append(options[int(rng.integers(0, len(options)))])
    if len(chosen) != 20 or len({row["source_audit_id"] for row in chosen}) != 20:
        raise ValueError("Form A duplicate coverage selection failed")
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
    if row.get("gold_override"):
        return tuple(row["gold_override"])
    if row["key"].get("label") != "P_new":
        return "", "", ""
    arm = row["key"]["arm_id"]
    if arm == "caring":
        return "persona_instruction", "persona_caring_1_to_5", "4:5"
    if arm == "neutral":
        return "persona_instruction", "persona_caring_1_to_5", "1:2"
    return "", "", ""


def mark_form_b_gold(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    chosen: set[str] = set()
    sources = sorted({str(row["key"]["source"]) for row in rows})
    for source in sources:
        for cost in ("no_active", "zero"):
            options = sorted(
                (
                    row for row in rows
                    if row["key"]["source"] == source and row["key"]["cost"] == cost
                ),
                key=lambda row: row["source_audit_id"],
            )
            chosen.add(options[0]["source_audit_id"])
    output = []
    for row in rows:
        if row["source_audit_id"] not in chosen:
            output.append(row)
            continue
        expected = "no" if row["key"]["cost"] == "no_active" else "yes"
        output.append({
            **row,
            "gold_override": ("active_objective_explicit", "active_objective_yes_no", expected),
        })
    if len(chosen) != 8:
        raise ValueError(f"expected eight Form B gold items, got {len(chosen)}")
    return output


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

    r2b = join_packet(R2B_PACKET, R2B_KEY)
    form_a, selected_families = select_gate2(rng)
    form_b = mark_form_b_gold([{**row, "stratum": "R2b-census"} for row in r2b])
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
        "Complete Form A first, then complete Form B. Do not inspect "
        "files outside this `annotator` directory. Some items may recur; rate every row "
        "independently without comparing it to earlier answers.\n\n"
        "Use integer ratings from 1 (clearly absent/low) to 5 (clearly present/high). "
        "Mark `active_objective_yes_no` as `yes` or `no`. Use `notes` only when the text "
        "is ambiguous or a requested rating cannot be made.\n\n"
        "Form A checks four independent prompt manipulations. Rate each requested "
        "dimension directly from the text; do not infer one dimension from another.\n\n"
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
                "canonical_rows": 136,
                "retest_rows": 20,
                "total_rows": len(packet_a),
                "strata": {
                    "FAILED-B_new": 32,
                    "FAILED-Spos_new": 16,
                    "FAILED-O_new": 16,
                    "FAILED-Ctext_new": 32,
                    "TARGET-WP3-observation": 32,
                    "GOLD-P_new": 8,
                },
                "selected_families": selected_families,
                "selected_source_audit_ids": {
                    stratum: [
                        row["source_audit_id"] for row in form_a
                        if row["stratum"] == stratum
                    ]
                    for stratum in (
                        "FAILED-B_new", "FAILED-Spos_new", "FAILED-O_new",
                        "FAILED-Ctext_new", "TARGET-WP3-observation", "GOLD-P_new",
                    )
                },
                "retest_source_audit_ids": [row["source_audit_id"] for row in dup_a],
                "gold_unique_items": sum(bool(gold_metadata(row)[0]) for row in form_a),
            },
            "B": {
                "canonical_rows": 160,
                "retest_rows": 24,
                "total_rows": len(packet_b),
                "retest_source_audit_ids": [row["source_audit_id"] for row in dup_b],
                "gold_unique_items": sum(bool(gold_metadata(row)[0]) for row in form_b),
            },
        },
        "sources": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (R2B_PACKET, R2B_KEY, WP1, WP3)
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
