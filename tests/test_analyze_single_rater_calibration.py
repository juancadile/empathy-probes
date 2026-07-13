import csv
import json
import shutil
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "analysis"))

import analyze_single_rater_calibration as analysis  # noqa: E402


def make_responses(tmp_path):
    out = tmp_path / "responses"
    out.mkdir()
    keys = {
        row["audit_id"]: row
        for row in csv.DictReader(open(analysis.BUNDLE / "private_do_not_send/combined_key.csv"))
    }
    for form in ("A", "B"):
        source = analysis.BUNDLE / f"annotator/form_{form}.csv"
        rows = list(csv.DictReader(open(source)))
        for row in rows:
            key = keys[row["audit_id"]]
            for field in analysis.NUMERIC_FIELDS[form]:
                row[field] = "3"
            row["active_objective_yes_no"] = "no" if key.get("cost") == "no_active" else "yes"
            if form == "B":
                row["current_need_rating_1_to_5"] = "1" if key["need"] == "resolved" else "5"
                row["interruption_cost_rating_1_to_5"] = {
                    "no_active": "1", "zero": "1", "low": "2", "medium": "3", "high": "5"
                }[key["cost"]]
            else:
                label, arm = key["label"], key["arm_id"]
                if label == "Ctext_new":
                    row["task_pressure_1_to_5"] = {"zero": "1", "low": "2"}[arm]
                if label == "observation":
                    row["welfare_relevance_1_to_5"] = "5" if arm == "current_actual" else "1"
                if label == "persona":
                    row["welfare_relevance_1_to_5"] = "5" if arm.startswith("current_") else "1"
                    row["persona_caring_1_to_5"] = "5" if arm.endswith("caring") else "1"
                if label == "cost":
                    row["task_pressure_1_to_5"] = "5" if arm == "high" else "1"
                if label == "P_new":
                    row["persona_caring_1_to_5"] = "5" if arm == "caring" else "1"
        with (out / f"form_{form}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    return out


def test_complete_consistent_rater_passes_competence_and_reports_claims(tmp_path):
    report = analysis.analyze(make_responses(tmp_path))
    assert report["rater_competence_passed"] is True
    assert report["gold"]["failure_count"] == 0
    assert report["claims"]["r2b_resolved_need_by_cost"]["R2b-v2-dev"][
        "classification"] == "equivalently_flat"
    assert report["claims"]["wp3_observation"]["human_verdict"] == "positive"
    assert report["claim_authorization"].startswith("none")


def test_changed_prompt_fails_closed(tmp_path):
    responses = make_responses(tmp_path)
    rows = list(csv.DictReader(open(responses / "form_B.csv")))
    rows[0]["prompt"] += " changed"
    with (responses / "form_B.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="immutable field changed"):
        analysis.analyze(responses)


def test_manifest_hash_mismatch_fails_closed(tmp_path):
    bundle = tmp_path / "bundle"
    shutil.copytree(analysis.BUNDLE, bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["seed"] += 1
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        analysis.validate_bundle(bundle)
