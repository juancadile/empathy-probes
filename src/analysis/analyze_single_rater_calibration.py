"""Analyze the frozen single-rater calibration packet after rating.

The analyzer fails closed if the frozen manifest, source packet, immutable
prompt columns, row IDs, or private key differ from the committed packet. It
reports within-rater stability and claim-local manipulation verdicts; it never
turns single-rater corroboration into human certification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score


ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "data/gate_families/human_calibration_single_rater_20260713"
NUMERIC_FIELDS = {
    "A": [
        "currentness_1_to_5", "actuality_1_to_5", "welfare_relevance_1_to_5",
        "distress_content_1_to_5", "warmth_1_to_5", "persona_caring_1_to_5",
        "motive_genuine_1_to_5", "valence_positive_1_to_5",
        "social_salience_1_to_5", "response_opportunity_1_to_5",
        "task_pressure_1_to_5", "task_persistence_1_to_5",
    ],
    "B": ["current_need_rating_1_to_5", "interruption_cost_rating_1_to_5"],
}
REQUIRED_STABILITY_FIELDS = {
    "A": [
        "actuality_1_to_5", "welfare_relevance_1_to_5", "distress_content_1_to_5",
        "persona_caring_1_to_5", "valence_positive_1_to_5",
        "response_opportunity_1_to_5", "task_pressure_1_to_5",
    ],
    "B": ["current_need_rating_1_to_5", "interruption_cost_rating_1_to_5"],
}
IMMUTABLE = {
    "A": ["audit_id", "kind", "prompt", "continuation"],
    "B": ["audit_id", "prompt"],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def validate_bundle(bundle: Path) -> dict[str, object]:
    manifest_path = bundle / "manifest.json"
    expected_line = (bundle / "manifest.sha256").read_text().strip().split()
    if len(expected_line) != 2 or expected_line[1] != "manifest.json":
        raise ValueError("malformed frozen manifest hash file")
    if sha256(manifest_path) != expected_line[0]:
        raise ValueError("frozen calibration manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest["artifacts"].items():
        path = bundle / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"frozen packet artifact hash mismatch: {relative}")
    for relative, expected in manifest["sources"].items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"source artifact hash mismatch: {relative}")
    return manifest


def load_response(form: str, responses: Path, bundle: Path) -> list[dict[str, str]]:
    frozen = read_csv(bundle / f"annotator/form_{form}.csv")
    response_path = responses / f"form_{form}.csv"
    if not response_path.is_file():
        raise ValueError(f"missing response form {form}")
    rated = read_csv(response_path)
    if len(rated) != len(frozen):
        raise ValueError(f"response Form {form} row count mismatch")
    for expected, observed in zip(frozen, rated, strict=True):
        for field in IMMUTABLE[form]:
            if observed.get(field) != expected.get(field):
                raise ValueError(f"Form {form} immutable field changed: {field}")
        for field in NUMERIC_FIELDS[form]:
            try:
                value = int(observed.get(field, ""))
            except ValueError as exc:
                raise ValueError(f"Form {form} invalid rating: {field}") from exc
            if value not in range(1, 6):
                raise ValueError(f"Form {form} rating out of range: {field}")
        if observed.get("active_objective_yes_no", "").strip().lower() not in {"yes", "no"}:
            raise ValueError(f"Form {form} invalid active-objective response")
    return rated


def private_key(bundle: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(bundle / "private_do_not_send/combined_key.csv")
    return {row["audit_id"]: row for row in rows}


def quadratic_kappas(
    form: str, responses: dict[str, dict[str, str]], keys: dict[str, dict[str, str]]
) -> dict[str, float | None]:
    retests = [row for row in keys.values() if row["form"] == form and row["presentation"] == "retest"]
    output = {}
    for field in NUMERIC_FIELDS[form]:
        first = [int(responses[row["duplicate_of_audit_id"]][field]) for row in retests]
        second = [int(responses[row["audit_id"]][field]) for row in retests]
        if len(set(first) | set(second)) < 2:
            output[field] = None
            continue
        value = float(cohen_kappa_score(first, second, weights="quadratic"))
        output[field] = value if np.isfinite(value) else None
    return output


def active_objective_kappa(
    form: str, responses: dict[str, dict[str, str]], keys: dict[str, dict[str, str]]
) -> float | None:
    retests = [row for row in keys.values() if row["form"] == form and row["presentation"] == "retest"]
    first = [responses[row["duplicate_of_audit_id"]]["active_objective_yes_no"] for row in retests]
    second = [responses[row["audit_id"]]["active_objective_yes_no"] for row in retests]
    if len(set(first) | set(second)) < 2:
        return None
    value = float(cohen_kappa_score(first, second))
    return value if np.isfinite(value) else None


def gold_result(
    available_forms: set[str], responses: dict[str, dict[str, str]], keys: dict[str, dict[str, str]]
) -> dict[str, object]:
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    metadata = {}
    for audit_id, key in keys.items():
        if key["form"] not in available_forms or not key["gold_kind"]:
            continue
        by_source[key["source_audit_id"]].append(responses[audit_id])
        metadata[key["source_audit_id"]] = key
    failures = []
    for source_id, presentations in sorted(by_source.items()):
        key = metadata[source_id]
        field, expected = key["gold_rating_field"], key["gold_expected_range"]
        values = [row[field].strip().lower() for row in presentations]
        if field.endswith("_1_to_5"):
            low, high = (int(value) for value in expected.split(":"))
            passed = low <= float(np.mean([int(value) for value in values])) <= high
        else:
            passed = all(value == expected for value in values)
        if not passed:
            failures.append({"source_audit_id": source_id, "field": field,
                             "expected": expected, "observed": values})
    return {
        "unique_gold_items": len(by_source),
        "failures": failures,
        "failure_count": len(failures),
        "passed": len(failures) < 2,
    }


def canonical_rows(
    form: str, responses: dict[str, dict[str, str]], keys: dict[str, dict[str, str]]
) -> list[dict[str, object]]:
    grouped: dict[str, list[tuple[dict[str, str], dict[str, str]]]] = defaultdict(list)
    for audit_id, key in keys.items():
        if key["form"] == form:
            grouped[key["source_audit_id"]].append((responses[audit_id], key))
    output = []
    for source_id, presentations in grouped.items():
        first_key = presentations[0][1]
        row: dict[str, object] = {**first_key, "source_audit_id": source_id}
        for field in NUMERIC_FIELDS[form]:
            row[field] = float(np.mean([int(response[field]) for response, _ in presentations]))
        yes_no = [response["active_objective_yes_no"].strip().lower() for response, _ in presentations]
        row["active_objective_yes_no"] = max(set(yes_no), key=yes_no.count)
        output.append(row)
    return output


def paired_deltas(
    rows: list[dict[str, object]], label: str, positive: str, negative: str, field: str
) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row.get("label") == label and row.get("arm_id") in {positive, negative}:
            grouped[str(row["family_id"])][str(row["arm_id"])] = float(row[field])
    output = []
    for family, arms in sorted(grouped.items()):
        if set(arms) != {positive, negative}:
            raise ValueError(f"incomplete human contrast {label} in {family}")
        output.append({"family_id": family, "delta": arms[positive] - arms[negative]})
    if not output:
        raise ValueError(f"no human rows for contrast {label}")
    return output


def directional_result(deltas: list[dict[str, object]], machine_direction: str = "positive"):
    values = np.asarray([row["delta"] for row in deltas])
    mean = float(values.mean())
    verdict = "positive" if mean > 0 else "negative" if mean < 0 else "flat"
    return {
        "family_count": len(values),
        "family_deltas": deltas,
        "mean_delta": mean,
        "human_verdict": verdict,
        "machine_verdict": machine_direction,
        "machine_human_match": verdict == machine_direction,
    }


def mean_delta(deltas: list[dict[str, object]]) -> float:
    return float(np.mean([row["delta"] for row in deltas]))


def positive_family_count(deltas: list[dict[str, object]]) -> int:
    return sum(float(row["delta"]) > 0 for row in deltas)


def rows_for_label(rows: list[dict[str, object]], label: str) -> list[dict[str, object]]:
    selected = [row for row in rows if row.get("label") == label]
    if not selected:
        raise ValueError(f"no human rows for {label}")
    return selected


def b_new_result(rows: list[dict[str, object]]) -> dict[str, object]:
    selected = rows_for_label(rows, "B_new")
    grouped: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in selected:
        grouped[str(row["family_id"])][str(row["arm_id"])] = row
    correct = []
    for family, arms in sorted(grouped.items()):
        expected = {"active_zero_cost", "no_active_objective"}
        if set(arms) != expected:
            raise ValueError(f"incomplete B_new family: {family}")
        correct.append({
            "family_id": family,
            "correct": (
                arms["active_zero_cost"]["active_objective_yes_no"] == "yes"
                and arms["no_active_objective"]["active_objective_yes_no"] == "no"
            ),
        })
    task = paired_deltas(
        selected, "B_new", "active_zero_cost", "no_active_objective",
        "task_pressure_1_to_5",
    )
    target_passed = sum(row["correct"] for row in correct) >= 7
    invariant_passed = abs(mean_delta(task)) <= 0.30
    return {
        "family_count": len(grouped),
        "objective_presence_correct_families": int(sum(row["correct"] for row in correct)),
        "family_objective_checks": correct,
        "task_pressure_active_minus_no_active": task,
        "mean_task_pressure_delta": mean_delta(task),
        "target_passed": target_passed,
        "invariants_passed": invariant_passed,
        "valid_for_wp2_negative": target_passed and invariant_passed,
        "observed_wp2_auroc": 0.359375,
        "observed_wp2_inverse": True,
    }


def spos_new_result(rows: list[dict[str, object]]) -> dict[str, object]:
    selected = rows_for_label(rows, "Spos_new")
    valence = paired_deltas(
        selected, "Spos_new", "positive", "neutral", "valence_positive_1_to_5"
    )
    distress = paired_deltas(
        selected, "Spos_new", "positive", "neutral", "distress_content_1_to_5"
    )
    arm_means = {
        arm: float(np.mean([
            row["distress_content_1_to_5"] for row in selected if row["arm_id"] == arm
        ]))
        for arm in ("positive", "neutral")
    }
    target_passed = mean_delta(valence) >= 1.0 and positive_family_count(valence) >= 7
    invariant_passed = max(arm_means.values()) <= 2.0 and abs(mean_delta(distress)) <= 0.30
    return {
        "family_count": len(valence),
        "valence_positive_minus_neutral": valence,
        "mean_valence_delta": mean_delta(valence),
        "positive_valence_families": positive_family_count(valence),
        "distress_positive_minus_neutral": distress,
        "mean_distress_delta": mean_delta(distress),
        "distress_arm_means": arm_means,
        "target_passed": target_passed,
        "invariants_passed": invariant_passed,
        "valid_for_wp2_negative": target_passed and invariant_passed,
        "observed_wp2_auroc": 0.625,
    }


def o_new_result(rows: list[dict[str, object]]) -> dict[str, object]:
    selected = rows_for_label(rows, "O_new")
    opportunity = paired_deltas(
        selected, "O_new", "available", "unavailable", "response_opportunity_1_to_5"
    )
    welfare = paired_deltas(
        selected, "O_new", "available", "unavailable", "welfare_relevance_1_to_5"
    )
    task = paired_deltas(
        selected, "O_new", "available", "unavailable", "task_pressure_1_to_5"
    )
    target_passed = mean_delta(opportunity) >= 1.0 and positive_family_count(opportunity) >= 7
    invariant_passed = abs(mean_delta(welfare)) <= 0.30 and abs(mean_delta(task)) <= 0.30
    return {
        "family_count": len(opportunity),
        "opportunity_available_minus_unavailable": opportunity,
        "mean_opportunity_delta": mean_delta(opportunity),
        "positive_opportunity_families": positive_family_count(opportunity),
        "welfare_available_minus_unavailable": welfare,
        "mean_welfare_delta": mean_delta(welfare),
        "task_available_minus_unavailable": task,
        "mean_task_delta": mean_delta(task),
        "target_passed": target_passed,
        "invariants_passed": invariant_passed,
        "valid_for_wp2_negative": target_passed and invariant_passed,
        "observed_wp2_auroc": 0.609375,
    }


def ctext_new_result(rows: list[dict[str, object]]) -> dict[str, object]:
    selected = rows_for_label(rows, "Ctext_new")
    task_high_zero = paired_deltas(
        selected, "Ctext_new", "high", "zero", "task_pressure_1_to_5"
    )
    welfare_high_zero = paired_deltas(
        selected, "Ctext_new", "high", "zero", "welfare_relevance_1_to_5"
    )
    adjacent = {}
    for positive, negative in (("low", "zero"), ("medium", "low"), ("high", "medium")):
        deltas = paired_deltas(
            selected, "Ctext_new", positive, negative, "task_pressure_1_to_5"
        )
        adjacent[f"{positive}_minus_{negative}"] = {
            "family_deltas": deltas,
            "mean_delta": mean_delta(deltas),
            "positive_families": positive_family_count(deltas),
        }
    target_passed = (
        mean_delta(task_high_zero) >= 1.0 and positive_family_count(task_high_zero) >= 7
    )
    invariant_passed = abs(mean_delta(welfare_high_zero)) <= 0.30
    return {
        "family_count": len(task_high_zero),
        "task_high_minus_zero": task_high_zero,
        "mean_task_high_minus_zero": mean_delta(task_high_zero),
        "positive_task_families": positive_family_count(task_high_zero),
        "welfare_high_minus_zero": welfare_high_zero,
        "mean_welfare_high_minus_zero": mean_delta(welfare_high_zero),
        "adjacent_task_rungs_diagnostic": adjacent,
        "target_passed": target_passed,
        "invariants_passed": invariant_passed,
        "valid_for_wp2_negative": target_passed and invariant_passed,
        "observed_wp2_auroc": 0.640625,
    }


def observation_target_result(rows: list[dict[str, object]]) -> dict[str, object]:
    selected = rows_for_label(rows, "observation")
    welfare = paired_deltas(
        selected, "observation", "current_actual", "archived_actual",
        "welfare_relevance_1_to_5",
    )
    actuality = paired_deltas(
        selected, "observation", "current_actual", "archived_actual",
        "actuality_1_to_5",
    )
    task = paired_deltas(
        selected, "observation", "current_actual", "archived_actual",
        "task_pressure_1_to_5",
    )
    target_passed = mean_delta(welfare) >= 1.0 and positive_family_count(welfare) >= 14
    invariant_passed = abs(mean_delta(actuality)) <= 0.30 and abs(mean_delta(task)) <= 0.30
    return {
        "family_count": len(welfare),
        "welfare_current_minus_archived": welfare,
        "mean_welfare_delta": mean_delta(welfare),
        "positive_welfare_families": positive_family_count(welfare),
        "actuality_current_minus_archived": actuality,
        "mean_actuality_delta": mean_delta(actuality),
        "task_current_minus_archived": task,
        "mean_task_delta": mean_delta(task),
        "target_passed": target_passed,
        "invariants_passed": invariant_passed,
        "valid_for_wp2_interpretation": target_passed and invariant_passed,
        "observed_wp2_target_auroc": 1.0,
        "claim_ceiling": "prompt-stipulated current welfare status",
    }


def bootstrap_interval(values: np.ndarray, level: float = 0.90) -> list[float]:
    rng = np.random.default_rng(761931)
    draws = np.mean(rng.choice(values, size=(20000, len(values)), replace=True), axis=1)
    tail = (1 - level) / 2
    return [float(np.quantile(draws, tail)), float(np.quantile(draws, 1 - tail))]


def r2b_result(rows: list[dict[str, object]]) -> dict[str, object]:
    output = {}
    for partition in sorted({str(row["partition"]) for row in rows}):
        subset = [row for row in rows if row["partition"] == partition and row["need"] == "resolved"]
        grouped: dict[str, dict[str, float]] = defaultdict(dict)
        for row in subset:
            grouped[str(row["family_id"])][str(row["cost"])] = float(
                row["current_need_rating_1_to_5"]
            )
        deltas = np.asarray([arms["high"] - arms["zero"] for arms in grouped.values()])
        ci = bootstrap_interval(deltas)
        classification = (
            "equivalently_flat" if ci[0] >= -0.5 and ci[1] <= 0.5
            else "rising" if ci[0] > 0
            else "falling" if ci[1] < 0
            else "inconclusive"
        )
        output[partition] = {
            "family_count": len(grouped),
            "family_high_minus_zero": dict(zip(sorted(grouped), deltas.tolist(), strict=True)),
            "mean_high_minus_zero": float(deltas.mean()),
            "bootstrap_90_ci": ci,
            "classification": classification,
            "opus_classification": "equivalently_flat",
            "machine_human_match": classification == "equivalently_flat",
        }
    return output


def contrast_membership(key: dict[str, str]) -> list[tuple[str, str]]:
    if key["form"] == "B" and key.get("need") == "resolved" and key.get("cost") in {"zero", "high"}:
        return [("r2b_resolved_need_by_cost", "current_need_rating_1_to_5")]
    label, arm = key.get("label"), key.get("arm_id")
    output = []
    if label == "B_new":
        output.extend((
            ("wp2_B_new", "active_objective_yes_no"),
            ("wp2_B_new", "task_pressure_1_to_5"),
        ))
    if label == "Spos_new":
        output.extend((
            ("wp2_Spos_new", "valence_positive_1_to_5"),
            ("wp2_Spos_new", "distress_content_1_to_5"),
        ))
    if label == "O_new":
        output.extend((
            ("wp2_O_new", "response_opportunity_1_to_5"),
            ("wp2_O_new", "welfare_relevance_1_to_5"),
            ("wp2_O_new", "task_pressure_1_to_5"),
        ))
    if label == "Ctext_new":
        output.extend((
            ("wp2_Ctext_new", "task_pressure_1_to_5"),
            ("wp2_Ctext_new", "welfare_relevance_1_to_5"),
        ))
    if label == "observation" and arm in {"current_actual", "archived_actual"}:
        output.extend((
            ("wp2_target_observation", "welfare_relevance_1_to_5"),
            ("wp2_target_observation", "actuality_1_to_5"),
            ("wp2_target_observation", "task_pressure_1_to_5"),
        ))
    return output


def rating_value(response: dict[str, str], field: str) -> int:
    if field == "active_objective_yes_no":
        return int(response[field] == "yes")
    return int(response[field])


def stability_results(responses: dict[str, dict[str, str]], keys: dict[str, dict[str, str]]):
    grouped: dict[str, list[bool]] = defaultdict(list)
    for key in keys.values():
        if key["presentation"] != "retest":
            continue
        for contrast, field in contrast_membership(key):
            delta = abs(rating_value(responses[key["audit_id"]], field) -
                        rating_value(responses[key["duplicate_of_audit_id"]], field))
            threshold = 1 if field == "active_objective_yes_no" else 2
            grouped[contrast].append(delta >= threshold)
    required = {
        "r2b_resolved_need_by_cost", "wp2_B_new", "wp2_Spos_new",
        "wp2_O_new", "wp2_Ctext_new", "wp2_target_observation",
    }
    output = {}
    for contrast in sorted(required):
        flags = grouped.get(contrast, [])
        rate = float(np.mean(flags)) if flags else None
        output[contrast] = {
            "retest_items": len(flags),
            "unstable_items": int(sum(flags)),
            "unstable_rate": rate,
            "passed": bool(flags) and rate <= 0.20,
        }
    return output


def analyze(responses_dir: Path, bundle: Path = BUNDLE) -> dict[str, object]:
    manifest = validate_bundle(bundle)
    available = {form for form in ("A", "B") if (responses_dir / f"form_{form}.csv").is_file()}
    if not available:
        raise ValueError("at least one completed response form is required")
    response_rows = {form: load_response(form, responses_dir, bundle) for form in available}
    responses = {row["audit_id"]: row for rows in response_rows.values() for row in rows}
    keys = private_key(bundle)
    kappas = {form: quadratic_kappas(form, responses, keys) for form in available}
    required_kappas = {
        form: {field: kappas[form][field] for field in REQUIRED_STABILITY_FIELDS[form]}
        for form in available
    }
    objective_kappas = {
        form: active_objective_kappa(form, responses, keys) for form in available
    }
    kappa_passed = all(
        value is not None and value >= 0.60
        for fields in required_kappas.values() for value in fields.values()
    ) and all(value is not None and value >= 0.60 for value in objective_kappas.values())
    gold = gold_result(available, responses, keys)
    competence = kappa_passed and gold["passed"]
    stability = stability_results(responses, keys)

    report: dict[str, object] = {
        "schema": "empathy-action-probes/single-rater-calibration-analysis/1",
        "role": "single-rater corroboration, not human certification",
        "bundle_manifest_sha256": sha256(bundle / "manifest.json"),
        "available_forms": sorted(available),
        "within_rater_quadratic_kappa": kappas,
        "within_rater_active_objective_kappa": objective_kappas,
        "required_kappas": required_kappas,
        "gold": gold,
        "rater_competence_passed": competence,
        "stability_by_claim": stability,
        "claims": {},
    }
    if "B" in available:
        b_rows = canonical_rows("B", responses, keys)
        report["claims"]["r2b_resolved_need_by_cost"] = r2b_result(b_rows)
    if "A" in available:
        a_rows = canonical_rows("A", responses, keys)
        controls = {
            "B_new": b_new_result(a_rows),
            "Spos_new": spos_new_result(a_rows),
            "O_new": o_new_result(a_rows),
            "Ctext_new": ctext_new_result(a_rows),
        }
        target = observation_target_result(a_rows)
        report["claims"]["wp2_failed_controls"] = controls
        report["claims"]["wp2_target_observation"] = target
        report["wp2_negative_interpretation_ready"] = (
            target["valid_for_wp2_interpretation"]
            and all(result["valid_for_wp2_negative"] for result in controls.values())
            and competence
            and all(
                result["passed"] for name, result in stability.items()
                if name.startswith("wp2_")
            )
        )
    report["claim_authorization"] = "none; apply claim-local protocol forks manually"
    report["packet_protocol_sha256"] = manifest["protocol_sha256"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError(f"refusing to overwrite analysis: {args.out}")
    report = analyze(args.responses)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "rater_competence_passed": report["rater_competence_passed"],
        "claims": list(report["claims"]),
    }, indent=2))


if __name__ == "__main__":
    main()
