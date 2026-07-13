import csv
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data_generation"))

import build_single_rater_calibration as calibration  # noqa: E402


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_materialized_packet_matches_frozen_counts_and_is_blinded(tmp_path):
    out = tmp_path / "packet"
    manifest = calibration.build(out)
    form_a = rows(out / "annotator/form_A.csv")
    form_b = rows(out / "annotator/form_B.csv")
    assert len(form_a) == 111
    assert len(form_b) == 184
    assert manifest["forms"]["A"]["strata"] == {"S1": 16, "S2": 48, "S3": 32}
    forbidden = {"family_id", "partition", "source", "label", "arm_id", "need", "cost",
                 "duplicate_of_audit_id", "gold_kind"}
    assert not forbidden.intersection(form_a[0])
    assert not forbidden.intersection(form_b[0])


def test_duplicates_are_exact_hidden_retests_with_minimum_lag(tmp_path):
    out = tmp_path / "packet"
    calibration.build(out)
    key = rows(out / "private_do_not_send/combined_key.csv")
    for form in ("A", "B"):
        packet = rows(out / f"annotator/form_{form}.csv")
        by_id = {row["audit_id"]: row for row in packet}
        positions = {row["audit_id"]: index for index, row in enumerate(packet)}
        retests = [row for row in key if row["form"] == form and row["presentation"] == "retest"]
        assert len(retests) == (15 if form == "A" else 24)
        for retest in retests:
            parent = retest["duplicate_of_audit_id"]
            assert positions[retest["audit_id"]] - positions[parent] >= calibration.MIN_RETEST_LAG
            left = {k: v for k, v in by_id[retest["audit_id"]].items() if k != "audit_id"}
            right = {k: v for k, v in by_id[parent].items() if k != "audit_id"}
            assert left == right


def test_complete_strata_and_gold_items_are_preserved(tmp_path):
    out = tmp_path / "packet"
    calibration.build(out)
    key = rows(out / "private_do_not_send/combined_key.csv")
    originals = [row for row in key if row["form"] == "A" and row["presentation"] == "original"]
    assert sum(row["stratum"] == "S1" for row in originals) == 16
    assert sum(row["stratum"] == "S2" for row in originals) == 48
    assert sum(row["stratum"] == "S3" for row in originals) == 32
    assert sum(bool(row["gold_kind"]) for row in originals) == 8

    s2 = [row for row in originals if row["stratum"] == "S2"]
    families = {row["family_id"] for row in s2}
    assert len(families) == 4
    assert all(sum(row["family_id"] == family for row in s2) == 12 for family in families)


def test_manifest_hash_is_external_and_exact(tmp_path):
    out = tmp_path / "packet"
    calibration.build(out)
    expected = (out / "manifest.sha256").read_text().split()[0]
    actual = hashlib.sha256((out / "manifest.json").read_bytes()).hexdigest()
    assert expected == actual
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["blinding"]["duplicate_links_private"] is True


def test_builder_refuses_to_overwrite_frozen_packet(tmp_path):
    out = tmp_path / "packet"
    calibration.build(out)
    try:
        calibration.build(out)
    except ValueError as exc:
        assert "refusing to overwrite" in str(exc)
    else:
        raise AssertionError("builder overwrote a frozen packet")
