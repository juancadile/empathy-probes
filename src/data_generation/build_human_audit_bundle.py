"""Build the blinded 604-row human-audit handoff.

The Gate 2 and Gate 1 R2b packets use different rating instruments. They stay
as separate forms inside one annotator bundle; their hidden design labels are
merged into a private key that must not be sent to annotators.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/gate_families/human_audit_604_20260713"
SOURCES = (
    {
        "form": "A",
        "study": "gate2_v2",
        "packet": ROOT / "data/gate_families/gate2_v2/human_audit/gate2_v2_packet.csv",
        "key": ROOT / "data/gate_families/gate2_v2/human_audit/gate2_v2_key.csv",
        "expected_rows": 444,
    },
    {
        "form": "B",
        "study": "gate1_r2b_v2",
        "packet": ROOT / "data/gate_families/gate1_r2b_v2/human_audit/r2b_v2_packet.csv",
        "key": ROOT / "data/gate_families/gate1_r2b_v2/human_audit/r2b_v2_key.csv",
        "expected_rows": 160,
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_private_key(path: Path, rows: list[dict[str, str]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    annotator = OUTPUT / "annotator"
    private = OUTPUT / "private_do_not_send"
    annotator.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    private_rows: list[dict[str, str]] = []
    manifest_sources = []
    seen_ids: set[str] = set()

    for source in SOURCES:
        packet_rows = read_rows(source["packet"])
        key_rows = read_rows(source["key"])
        if len(packet_rows) != source["expected_rows"]:
            raise ValueError(f"unexpected row count for {source['packet']}: {len(packet_rows)}")
        if len(key_rows) != len(packet_rows):
            raise ValueError(f"packet/key length mismatch for {source['study']}")

        packet_ids = {row["audit_id"] for row in packet_rows}
        key_ids = {row["audit_id"] for row in key_rows}
        if packet_ids != key_ids:
            raise ValueError(f"packet/key ID mismatch for {source['study']}")
        if seen_ids & packet_ids:
            raise ValueError("audit IDs collide across forms")
        seen_ids |= packet_ids

        output_packet = annotator / f"form_{source['form']}.csv"
        shutil.copyfile(source["packet"], output_packet)
        for row in key_rows:
            private_rows.append({
                "audit_id": row["audit_id"],
                "form": source["form"],
                "source_study": source["study"],
                **{k: v for k, v in row.items() if k != "audit_id"},
            })
        manifest_sources.append({
            "form": source["form"],
            "rows": len(packet_rows),
            "source_packet": str(source["packet"].relative_to(ROOT)),
            "source_packet_sha256": sha256(source["packet"]),
            "source_key": str(source["key"].relative_to(ROOT)),
            "source_key_sha256": sha256(source["key"]),
            "annotator_packet": str(output_packet.relative_to(ROOT)),
            "annotator_packet_sha256": sha256(output_packet),
        })

    key_path = private / "combined_key.csv"
    write_private_key(key_path, private_rows)
    manifest = {
        "schema": "empathy-action-probes/human-audit-bundle/1",
        "total_rows": len(private_rows),
        "blinding": {
            "annotator_directory_contains_design_labels": False,
            "private_key_must_not_be_sent": True,
            "forms_remain_separate_because_rating_instruments_differ": True,
        },
        "sources": manifest_sources,
        "private_key": str(key_path.relative_to(ROOT)),
        "private_key_sha256": sha256(key_path),
    }
    if manifest["total_rows"] != 604:
        raise ValueError(f"expected 604 total rows, got {manifest['total_rows']}")
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    (annotator / "README.md").write_text(
        "# Blinded Human Audit\n\n"
        "Complete both forms: 444 rows in Form A and 160 rows in Form B "
        "(604 total). Do not inspect files outside this `annotator` directory.\n\n"
        "Use integer ratings from 1 (clearly absent/low) to 5 (clearly "
        "present/high). Mark `active_objective_yes_no` as `yes` or `no`. "
        "Use `notes` only for ambiguity or a rating you could not make.\n\n"
        "Form A asks about the dimensions named in its columns. Rate the "
        "present text only; do not infer hidden experimental conditions.\n\n"
        "Form B asks for (1) the speaker's current need for support and (2) "
        "the interruption cost imposed by the stated short pause. Rate these "
        "as separate dimensions. In particular, do not increase current need "
        "merely because interruption cost is larger.\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
