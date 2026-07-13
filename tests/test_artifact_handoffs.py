import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "utils"))

from build_artifact_manifest import build  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_human_audit_bundle_is_blinded_and_complete():
    root = ROOT / "data/gate_families/human_audit_604_20260713"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["total_rows"] == 604
    assert [source["rows"] for source in manifest["sources"]] == [444, 160]
    assert not list((root / "annotator").glob("*key*"))

    packet_ids = set()
    for source in manifest["sources"]:
        packet = ROOT / source["annotator_packet"]
        rows = read_csv(packet)
        assert len(rows) == source["rows"]
        assert sha256(packet) == source["annotator_packet_sha256"]
        assert "source_study" not in rows[0]
        ids = {row["audit_id"] for row in rows}
        assert not packet_ids & ids
        packet_ids |= ids

    key = read_csv(ROOT / manifest["private_key"])
    assert len(key) == 604
    assert {row["audit_id"] for row in key} == packet_ids
    assert sha256(ROOT / manifest["private_key"]) == manifest["private_key_sha256"]


def test_e27_tree_matches_manifest_and_accepted_gate0b_binding():
    root = ROOT / "results/e27_game_variants"
    stored = json.loads((root / "PROVENANCE.json").read_text())
    rebuilt = build([root / "baseline", root / "suppressors"])
    assert rebuilt == stored
    assert stored["file_count"] == 54

    audit = json.loads((root / "GATE0B_BINDING_AUDIT.json").read_text())
    assert audit["exact_match"] is True
    assert audit["accepted_declared_file_count"] == 54
    assert audit["observed_file_count"] == 54
    assert audit["missing"] == []
    assert audit["extra"] == []
    assert audit["hash_mismatches"] == []
    assert sha256(ROOT / audit["accepted_artifact"]) == audit["accepted_artifact_sha256"]
    assert sha256(ROOT / audit["tree_manifest"]) == audit["tree_manifest_sha256"]


def test_activation_cache_manifest_is_self_consistent():
    manifest = json.loads((
        ROOT / "results/controlled_directions_gemma2_9b_it/ACTIVATION_CACHE_MANIFEST.json"
    ).read_text())
    assert manifest["schema"] == "empathy-action-probes/artifact-tree-manifest/1"
    assert manifest["file_count"] == 9
    assert manifest["total_bytes"] == sum(item["bytes"] for item in manifest["files"])
    assert len({item["path"] for item in manifest["files"]}) == manifest["file_count"]
