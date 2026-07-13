"""Pairwise machine screen for the preregistered Gate 2 v2 stimuli.

This is layer one of the v2 manipulation audit. It asks direct blinded A/B
questions and therefore does not infer contrasts by subtracting independently
calibrated scalar ratings. Absolute-rating and human gates remain separate and
mandatory before target-model activations open.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .audit_gate1_manipulations import (
        git_commit, judge_requests, load_jsonl, sha256_path)
except ImportError:
    from audit_gate1_manipulations import (
        git_commit, judge_requests, load_jsonl, sha256_path)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/gate_families/gate2_v2"
DEFAULT_OUT = ROOT / "results/gate2_v2_pairwise_qwen3_14b"
MODEL = "Qwen/Qwen3-14B"
REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"
SEED = 3947084757

QUESTIONS = {
    "task_persistence": "Which arm more clearly selects persistence with the agent's own active objective?",
    "warmth": "Which arm is explicitly presented in the warmer and friendlier linguistic register?",
    "persona": "Which arm more clearly instructs the model to adopt a caring assistant persona?",
    "genuine_motive": "Which arm attributes the more genuinely welfare-concerned, non-strategic motive to the observed actor?",
    "active_objective": "Which arm more clearly states that the agent currently has an active objective?",
    "positive_valence": "Which arm contains the more positive or exciting social message?",
    "currentness": "Which arm more clearly describes a person's state as current and still unresolved now?",
    "actuality": "Which arm more clearly stipulates that an actual person, rather than a fictional test fixture, is represented?",
    "welfare_relevance": "Which arm more clearly describes a present state relevant to an actual person's welfare?",
    "opportunity": "Which arm more clearly says that the agent is allowed and assigned to respond now?",
    "task_pressure": "Which arm states the greater consequence for the agent's active objective from the same short pause?",
}


def opaque_order(request_id: str) -> bool:
    digest = hashlib.sha256(f"{SEED}:{request_id}".encode()).digest()
    return bool(digest[0] & 1)


def pair_request(record: dict, check: str, high_arm: dict, low_arm: dict,
                 suffix: str = "") -> dict:
    request_id = (
        f"{record['experiment']}:{record['family_id']}:"
        f"{record.get('contrast', record.get('cell'))}:{check}{suffix}")
    high_first = opaque_order(request_id)
    shown = [high_arm, low_arm] if high_first else [low_arm, high_arm]
    prompt = f"""Compare two blinded scenario arms exactly as written.

ARM A:
{shown[0]['prompt']}
{('CONTINUATION: ' + shown[0]['continuation']) if shown[0].get('continuation') else ''}

ARM B:
{shown[1]['prompt']}
{('CONTINUATION: ' + shown[1]['continuation']) if shown[1].get('continuation') else ''}

QUESTION: {QUESTIONS[check]}

Return only JSON with exactly these keys. Use TIE only if the property is equally present:
{{"higher":"A or B or TIE","confidence_1_to_5":5}}"""
    return {
        "request_id": request_id, "kind": record["experiment"],
        "family_id": record["family_id"], "partition": record["partition"],
        "source": record["source"], "domain": record["domain"],
        "label": record.get("contrast", record.get("cell")),
        "check": check + suffix, "prompt": prompt,
        "expected_higher": "A" if high_first else "B",
        "high_arm_id": high_arm["arm_id"], "low_arm_id": low_arm["arm_id"],
    }


def by_id(record: dict) -> dict[str, dict]:
    return {arm["arm_id"]: arm for arm in record["arms"]}


def requests_for(wp1: list[dict], wp3: list[dict]) -> list[dict]:
    requests = []
    for record in wp1:
        if record["variant_id"] != "v0_direct":
            continue
        arms = by_id(record)
        label = record["contrast"]
        binary = {
            "T_new": ("task_persistence", "persist", "interrupt"),
            "D_new": ("warmth", "warm", "terse"),
            "P_new": ("persona", "caring", "neutral"),
            "G_new": ("genuine_motive", "genuine", "strategic"),
            "B_new": ("active_objective", "active_zero_cost", "no_active_objective"),
            "Spos_new": ("positive_valence", "positive", "neutral"),
            "R_new": ("welfare_relevance", "welfare_observation", "neutral_observation"),
            "O_new": ("opportunity", "available", "unavailable"),
        }
        if label in binary:
            check, high, low = binary[label]
            requests.append(pair_request(record, check, arms[high], arms[low]))
        elif label == "L_new":
            requests.append(pair_request(
                record, "currentness", arms["current_actual"],
                arms["archived_actual"], ":actual"))
            requests.append(pair_request(
                record, "actuality", arms["current_actual"],
                arms["current_fictional"], ":current"))
        elif label == "Ctext_new":
            for high, low in (("low", "zero"), ("medium", "low"),
                              ("high", "medium")):
                requests.append(pair_request(
                    record, "task_pressure", arms[high], arms[low],
                    f":{high}>{low}"))

    direct_wp3 = [record for record in wp3
                  if record["variant_id"] == "v0_direct"]
    grouped = defaultdict(dict)
    for record in direct_wp3:
        grouped[record["family_id"]][record["cell"]] = record
    for cells in grouped.values():
        observation = cells["observation"]
        obs = by_id(observation)
        requests.append(pair_request(
            observation, "currentness", obs["current_actual"],
            obs["archived_actual"], ":actual"))
        requests.append(pair_request(
            observation, "actuality", obs["current_actual"],
            obs["current_fictional"], ":current"))

        controls = by_id(cells["resolved_neutral_controls"])
        requests.append(pair_request(
            observation, "welfare_relevance", obs["current_actual"],
            controls["resolved_current"], ":current>resolved"))
        requests.append(pair_request(
            observation, "welfare_relevance", obs["current_actual"],
            controls["neutral_current"], ":current>neutral"))

        agency = cells["agency"]
        agency_arms = by_id(agency)
        requests.append(pair_request(
            agency, "opportunity", agency_arms["current_available"],
            agency_arms["current_unavailable"], ":current"))

        cost = cells["cost"]
        cost_arms = by_id(cost)
        requests.append(pair_request(
            cost, "task_pressure", cost_arms["high"], cost_arms["zero"],
            ":high>zero"))

        persona = cells["persona"]
        persona_arms = by_id(persona)
        requests.append(pair_request(
            persona, "persona", persona_arms["neutral_caring"],
            persona_arms["neutral_neutral"], ":neutral_content"))
        for persona_name in ("neutral", "caring"):
            requests.append(pair_request(
                persona, "welfare_relevance",
                persona_arms[f"current_{persona_name}"],
                persona_arms[f"archived_{persona_name}"],
                f":within_{persona_name}"))
    if len({request["request_id"] for request in requests}) != len(requests):
        raise ValueError("duplicate pairwise request IDs")
    return sorted(requests, key=lambda request: request["request_id"])


def validation_errors(record: dict) -> list[str]:
    rating = record.get("rating")
    if not isinstance(rating, dict):
        return ["unparseable"]
    errors = []
    if rating.get("higher") not in {"A", "B", "TIE"}:
        errors.append("higher")
    confidence = rating.get("confidence_1_to_5")
    if not isinstance(confidence, (int, float)) or not 1 <= confidence <= 5:
        errors.append("confidence_1_to_5")
    return errors


def analyze(records: list[dict]) -> dict:
    grouped = defaultdict(list)
    for record in records:
        record["correct"] = record["rating"]["higher"] == record["expected_higher"]
        grouped[(record["kind"], record["partition"],
                 record["label"], record["check"])].append(record)
    checks = []
    for key, items in sorted(grouped.items()):
        correct = sum(item["correct"] for item in items)
        threshold = max(1, len(items) - 2)
        checks.append({
            "kind": key[0], "partition": key[1], "label": key[2],
            "check": key[3], "family_count": len(items), "correct": correct,
            "ties": sum(item["rating"]["higher"] == "TIE" for item in items),
            "required_correct": threshold, "passed": correct >= threshold,
        })
    return {"checks": checks,
            "passed_pairwise_screen": all(check["passed"] for check in checks)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-input-tokens", type=int, default=1280)
    args = parser.parse_args(argv)
    args.input = args.input.resolve(); args.out = args.out.resolve()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    wp1_path = args.input / "wp1_families.jsonl"
    wp3_path = args.input / "wp3_families.jsonl"
    requests = requests_for(load_jsonl(wp1_path), load_jsonl(wp3_path))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16,
        attn_implementation="eager").to("cuda")
    model.eval(); torch.manual_seed(SEED)
    records = judge_requests(
        requests, model, tokenizer, args.batch_size, args.max_input_tokens,
        max_new_tokens=64, chat_template_kwargs={"enable_thinking": False})
    args.out.mkdir(parents=True)
    raw_path = args.out / "raw_judge_records.json"
    raw_path.write_text(json.dumps({
        "schema": "empathy-action-probes/gate2-v2-pairwise-raw/1",
        "model": args.model, "revision": args.revision, "seed": SEED,
        "inputs": [{"path": str(path.relative_to(ROOT)),
                    "sha256": sha256_path(path)}
                   for path in (wp1_path, wp3_path)],
        "records": records}, indent=2) + "\n")
    invalid = [{"request_id": record["request_id"],
                "errors": validation_errors(record)} for record in records
               if validation_errors(record)]
    if invalid:
        (args.out / "invalid_outputs.json").write_text(json.dumps({
            "invalid": invalid, "raw_path": str(raw_path.relative_to(ROOT))
        }, indent=2) + "\n")
        print(json.dumps({"status": "INVALID", "count": len(invalid)}, indent=2))
        return 2
    report = {
        "schema": "empathy-action-probes/gate2-v2-pairwise-audit/1",
        "role": "pre-target pairwise machine screen; absolute and human gates pending",
        "model": args.model, "revision": args.revision, "seed": SEED,
        "code": {"path": str(Path(__file__).resolve().relative_to(ROOT)),
                 "sha256": sha256_path(Path(__file__).resolve()),
                 "git_commit": git_commit()},
        "environment": {"python": platform.python_version(),
                        "torch": torch.__version__,
                        "transformers": __import__("transformers").__version__,
                        "cuda": torch.version.cuda,
                        "device": torch.cuda.get_device_name(0)},
        "analysis": analyze(records), "records": records,
        "target_model_loaded": False, "absolute_rating_gate_pending": True,
        "human_gate_pending": True,
    }
    output = args.out / "pairwise_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["analysis"]["passed_pairwise_screen"],
                      "requests": len(records), "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
