"""Builder-integrity helpers + repaired T builders (Integrity Repair A).

Proves: (1) the FORMER lockstep %2/%4 construction fails the new uniqueness
assert; (2) the repaired Cartesian builders emit 40 rows = 40 unique pairs
with the full variant grid and disjoint development/confirmation families;
(3) a builder cannot silently overwrite an artifact — differing content is
preserved first, and preservation records are append-only.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "data_generation"))

from builder_integrity import (  # noqa: E402
    BuilderIntegrityError, SidecarValidationError, assert_disjoint_families,
    assert_unique_pairs, assert_variant_grid, preserve_existing_artifact,
    repair_sidecar, unique_pair_count, write_jsonl_guarded,
)
import build_cell_t  # noqa: E402
import build_cell_t_confirm  # noqa: E402


def lockstep_rows(scenarios, openers, closings, n_variants=8):
    """The pre-repair construction: openers/closings cycled in lockstep."""
    rows = []
    for scenario in scenarios:
        for index in range(n_variants):
            shared = scenario["prefix"] + openers[index % len(openers)]
            closing = closings[index % len(closings)]
            rows.append({
                "scenario_id": scenario["id"],
                "pair_index": index,
                "opener_variant": index % len(openers),
                "closing_variant": index % len(closings),
                "pos_text": f"{shared}I will {scenario['task']}. {closing}",
                "neg_text": f"{shared}I will {scenario['distract']}. {closing}",
            })
    return rows


def test_former_lockstep_construction_fails_uniqueness():
    rows = lockstep_rows(build_cell_t.SCENARIOS, build_cell_t.OPENERS,
                         build_cell_t.CLOSINGS)
    assert len(rows) == 40
    assert unique_pair_count(rows) == 20  # the audited defect
    with pytest.raises(BuilderIntegrityError, match="duplicate"):
        assert_unique_pairs(rows)


def test_former_lockstep_construction_fails_variant_grid():
    rows = lockstep_rows(build_cell_t.SCENARIOS, build_cell_t.OPENERS,
                         build_cell_t.CLOSINGS)
    with pytest.raises(BuilderIntegrityError, match="variant grid"):
        assert_variant_grid(rows, [s["id"] for s in build_cell_t.SCENARIOS],
                            ("opener_variant", "closing_variant"), (2, 4))


@pytest.mark.parametrize("builder", [build_cell_t, build_cell_t_confirm])
def test_repaired_builder_rows(builder):
    rows = builder.build_rows()
    assert len(rows) == 40
    assert unique_pair_count(rows) == 40
    assert_unique_pairs(rows)
    assert_variant_grid(rows, [s["id"] for s in builder.SCENARIOS],
                        ("opener_variant", "closing_variant"), (2, 4))
    for row in rows:
        assert row["pair_index"] == (row["opener_variant"] * 4
                                     + row["closing_variant"])
        assert row["pos_text"].startswith(row["shared_prefix"])
        assert row["neg_text"].startswith(row["shared_prefix"])


def test_development_confirmation_families_disjoint():
    dev, confirm = build_cell_t.build_rows(), build_cell_t_confirm.build_rows()
    assert_disjoint_families(dev, confirm)
    with pytest.raises(BuilderIntegrityError, match="shared"):
        assert_disjoint_families(dev, dev)


def test_committed_artifacts_match_builders():
    """The on-disk artifacts are exactly what the repaired builders emit."""
    for builder, name in [(build_cell_t, "T_templated.jsonl"),
                          (build_cell_t_confirm, "T_confirm_templated.jsonl")]:
        path = ROOT / "data" / "contrastive_pairs" / "v2_1" / name
        on_disk = [json.loads(line) for line in
                   path.read_text().splitlines() if line.strip()]
        assert on_disk == builder.build_rows()


def make_rows(tag, n_openers=2, n_closings=2):
    rows = []
    for fam in ("fam_a", "fam_b"):
        for oi in range(n_openers):
            for ci in range(n_closings):
                rows.append({
                    "scenario_id": fam,
                    "opener_variant": oi, "closing_variant": ci,
                    "pair_index": oi * n_closings + ci,
                    "pos_text": f"{tag} {fam} pos {oi}/{ci}",
                    "neg_text": f"{tag} {fam} neg {oi}/{ci}",
                })
    return rows


def guarded(path, rows, **kwargs):
    return write_jsonl_guarded(
        path, rows, reason="test",
        expected_families=("fam_a", "fam_b"),
        variant_fields=("opener_variant", "closing_variant"),
        expected_variant_counts=(2, 2), **kwargs)


def test_guarded_write_create_unchanged_and_preserve(tmp_path):
    out = tmp_path / "cell.jsonl"
    status = guarded(out, make_rows("v1"))
    assert status["action"] == "created"
    sidecar = json.loads((tmp_path / "cell.jsonl.provenance.json").read_text())
    assert sidecar["n_rows"] == 8 and sidecar["n_unique_pairs"] == 8

    status = guarded(out, make_rows("v1"))
    assert status["action"] == "unchanged"
    assert not (tmp_path / "historical").exists()

    old_bytes = out.read_bytes()
    status = guarded(out, make_rows("v2"))
    assert status["action"] == "replaced_with_preservation"
    preserved = status["preserved"]
    copies = list((tmp_path / "historical").rglob("cell.jsonl"))
    assert len(copies) == 1 and copies[0].read_bytes() == old_bytes
    meta = json.loads(
        (copies[0].parent / "preservation.json").read_text())
    assert meta["sha256"] == preserved["sha256"]
    assert meta["n_rows"] == 8

    # an unchanged rerun must not erase the sidecar's replaced_artifact
    # pointer back to the preserved history
    sidecar_path = tmp_path / "cell.jsonl.provenance.json"
    assert json.loads(sidecar_path.read_text())["replaced_artifact"]
    status = guarded(out, make_rows("v2"))
    assert status["action"] == "unchanged"
    assert json.loads(sidecar_path.read_text())["replaced_artifact"][
        "sha256"] == preserved["sha256"]


def test_preservation_is_append_only(tmp_path):
    out = tmp_path / "cell.jsonl"
    out.write_text("original content\n")
    meta1 = preserve_existing_artifact(out, reason="first")
    # same content again -> verified, same record, no duplicate dir
    meta2 = preserve_existing_artifact(out, reason="again")
    assert meta2["sha256"] == meta1["sha256"]
    assert len(list((tmp_path / "historical").iterdir())) == 1
    # tampering with the preserved copy is detected
    preserved_copy = next((tmp_path / "historical").rglob("cell.jsonl"))
    preserved_copy.write_text("tampered\n")
    with pytest.raises(BuilderIntegrityError, match="refusing to overwrite"):
        preserve_existing_artifact(out, reason="third")


def test_guarded_write_rejects_bad_rows(tmp_path):
    rows = make_rows("dup")
    rows[1] = dict(rows[0])  # duplicate pair + broken grid
    with pytest.raises(BuilderIntegrityError):
        guarded(tmp_path / "cell.jsonl", rows)
    assert not (tmp_path / "cell.jsonl").exists()


def test_guarded_write_cross_disjoint(tmp_path):
    other = tmp_path / "other.jsonl"
    other.write_text(json.dumps({"scenario_id": "fam_a", "pos_text": "x",
                                 "neg_text": "y"}) + "\n")
    with pytest.raises(BuilderIntegrityError, match="shared"):
        guarded(tmp_path / "cell.jsonl", make_rows("v1"),
                cross_disjoint_with=other)


def test_unchanged_artifact_requires_sidecar(tmp_path):
    out = tmp_path / "cell.jsonl"
    guarded(out, make_rows("v1"))
    out.with_suffix(".jsonl.provenance.json").unlink()
    with pytest.raises(SidecarValidationError, match="sidecar missing"):
        guarded(out, make_rows("v1"))


@pytest.mark.parametrize("field,value,match", [
    ("artifact", "other.jsonl", "artifact name"),
    ("sha256", "0" * 64, "artifact hash"),
    ("n_rows", 999, "n_rows"),
    ("n_unique_pairs", 999, "n_unique_pairs"),
    ("families", ["wrong"], "families"),
    ("reason", "", "reason"),
    ("written_at", "", "written_at"),
])
def test_sidecar_tamper_matrix_fails_closed(tmp_path, field, value, match):
    out = tmp_path / "cell.jsonl"
    guarded(out, make_rows("v1"))
    sidecar = out.with_suffix(".jsonl.provenance.json")
    record = json.loads(sidecar.read_text())
    record[field] = value
    sidecar.write_text(json.dumps(record))
    with pytest.raises(SidecarValidationError, match=match):
        guarded(out, make_rows("v1"))


def test_sidecar_detects_preserved_copy_tampering(tmp_path):
    out = tmp_path / "cell.jsonl"
    guarded(out, make_rows("v1"))
    guarded(out, make_rows("v2"))
    preserved = next((tmp_path / "historical").rglob("cell.jsonl"))
    preserved.write_text("tampered\n")
    with pytest.raises(SidecarValidationError, match="historical record"):
        guarded(out, make_rows("v2"))


def test_repair_refuses_artifact_without_preserved_copy_and_builder_grid(tmp_path):
    artifact = tmp_path / "cell.jsonl"
    artifact.write_text(json.dumps({"scenario_id": "x", "variant": 0,
                                    "pos_text": "p", "neg_text": "n"}) + "\n")
    with pytest.raises(BuilderIntegrityError, match="builder-specific"):
        repair_sidecar(artifact, "test")
    with pytest.raises(BuilderIntegrityError, match="preserved-copy"):
        repair_sidecar(
            artifact, "test", expected_families=("x",),
            variant_fields=("variant",), expected_variant_counts=(1,))
