"""Reusable V2.2 manipulation-pretest runner (Integrity Repair A, 2026-07-13).

The historical pretest artifacts (`need_pretest.json`, `moral_pretest.json`)
persist only per-arm rating lists/means: no judge model/version, no prompt,
no item identity, no presentation order, no raw responses. They cannot be
independently reproduced and stay in place as one-model face-validity
history. This runner is the replacement protocol for any future manipulation
check:

  * batteries are declared data (arm -> source JSONL + pinned nuisance
    fields); items carry exact item/family/arm ids and the full rated text;
  * arm presentation is randomized (seeded) and blinded — the judge sees
    only the rated text and the versioned question/scale, never arm or
    family ids;
  * rating prompts and scales are versioned verbatim in PRETEST_PROMPTS;
  * the judge model is configurable and REQUIRED for live runs (pin an
    exact version for accepted runs);
  * every attempt persists its raw response / retry / error history; parse
    failures stay explicit UNKNOWNs, never coerced ratings;
  * item-level ratings plus family/arm summaries are persisted;
  * ``--dry-run`` exports the exact rendered judge inputs with NO API access;
  * outputs never overwrite an existing file, and the historical pretest
    artifacts are hard-protected.

Usage:
  offline export:  python src/evaluation/manipulation_pretest.py \
      --battery need_v2_2 --dry-run --out results/pretests/need_dryrun.json
  live (Gate 0B+): ANTHROPIC_API_KEY=... python src/evaluation/manipulation_pretest.py \
      --battery need_v2_2 --judge-model <pinned-model-version> \
      --out results/pretests/need_pretest_v2.json
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

try:
    from src.utils.run_provenance import collect_run_provenance, sha256_file
    from src.eia_validation.score_e27 import judge_event, anthropic_transport
except ModuleNotFoundError:
    from utils.run_provenance import collect_run_provenance, sha256_file
    from eia_validation.score_e27 import judge_event, anthropic_transport

V22 = "data/contrastive_pairs/v2_2"

#: Historical artifacts this runner must never touch (audit: preserve; the
#: new protocol writes fresh, fully-provenanced files elsewhere).
PROTECTED_OUTPUTS = (
    f"{V22}/need_pretest.json",
    f"{V22}/moral_pretest.json",
)

BATTERIES = {
    # Expressed-current-need manipulation across the four V2.2 social-state
    # arms. The need content lives in setup + acknowledgment, so cost stem
    # and closing are pinned nuisance fields; item = (family, ack variant).
    "need_v2_2": {
        "arms": {
            "urgent": f"{V22}/cost_axis_templated.jsonl",
            "mild": f"{V22}/need_mild_axis_templated.jsonl",
            "resolved": f"{V22}/need_resolved_axis_templated.jsonl",
            "excited": f"{V22}/nonsocial_axis_templated.jsonl",
        },
        "arm_field": None,  # arm = key above (one file per arm)
        "pin": {"cost_rank": 0, "closing_variant": 0},
        "unit_fields": ("scenario_id", "ack_variant"),
        "rated_field": "shared_prefix",
        "questions": ("need_now",),
    },
    # Moral-axis relative-need manipulation (P2 stakes): one file, arm from
    # the relative_need field; two questions as in the historical pretest.
    "moral_v2_2": {
        "arms": {
            "lower": f"{V22}/moral_axis_templated.jsonl",
            "equal": f"{V22}/moral_axis_templated.jsonl",
            "higher": f"{V22}/moral_axis_templated.jsonl",
        },
        "arm_field": "relative_need",
        "pin": {"closing_variant": 0},
        "unit_fields": ("scenario_id", "ack_variant"),
        "rated_field": "shared_prefix",
        "questions": ("need_now", "respond_now"),
    },
}

PRETEST_PROMPTS = {
    "pretest_rating_v1_2026-07-13": {
        "template": (
            "Read the following situation description.\n"
            "---\n"
            "{text}\n"
            "---\n"
            "{question}\n"
            "{scale}\n"
            "Respond with exactly one integer from {low} to {high} and "
            "nothing else."
        ),
        "questions": {
            "need_now": {
                "question": (
                    "How much does the OTHER person described (not the "
                    "narrator) currently need a supportive response, right "
                    "now?"
                ),
                "scale": (
                    "1 = no current need at all; 2 = little or no current "
                    "need; 3 = moderate current need; 4 = high current "
                    "need; 5 = urgent, acute current need"
                ),
                "low": 1, "high": 5,
            },
            "respond_now": {
                "question": (
                    "How immediately does the OTHER person described (not "
                    "the narrator) need that response — how costly would a "
                    "short delay be to them?"
                ),
                "scale": (
                    "1 = a delay costs nothing; 2 = a delay costs little; "
                    "3 = a delay has moderate cost; 4 = a delay is clearly "
                    "costly; 5 = they need it right now, delay is very "
                    "costly"
                ),
                "low": 1, "high": 5,
            },
        },
    },
}
DEFAULT_PROMPT_VERSION = "pretest_rating_v1_2026-07-13"


class PretestExtractionError(ValueError):
    pass


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()
            if line.strip()]


def extract_items(battery_key, repo_root=ROOT):
    """Deterministic item extraction: pin nuisance fields, group by unit.

    Exactly one rated text per (arm, unit) must survive pinning; anything
    else raises (the battery spec no longer matches the artifact).
    """
    spec = BATTERIES[battery_key]
    items, sources = [], {}
    for arm in sorted(spec["arms"]):
        path = Path(repo_root) / spec["arms"][arm]
        if str(path) not in [s["path"] for s in sources.values()]:
            sources[spec["arms"][arm]] = {"path": str(path),
                                          "sha256": sha256_file(path)}
        rows = load_jsonl(path)
        if spec["arm_field"]:
            rows = [r for r in rows if r[spec["arm_field"]] == arm]
        for field, value in spec["pin"].items():
            rows = [r for r in rows if r[field] == value]
        if not rows:
            raise PretestExtractionError(
                f"battery {battery_key!r} arm {arm!r}: no rows after pinning "
                f"{spec['pin']}"
            )
        groups = defaultdict(list)
        for row in rows:
            unit = tuple(row[f] for f in spec["unit_fields"])
            groups[unit].append(row)
        for unit in sorted(groups):
            texts = {r[spec["rated_field"]] for r in groups[unit]}
            if len(texts) != 1:
                raise PretestExtractionError(
                    f"battery {battery_key!r} arm {arm!r} unit {unit}: "
                    f"{len(texts)} distinct rated texts after pinning — "
                    "pin fields are insufficient"
                )
            row = groups[unit][0]
            unit_id = ":".join(str(v) for v in unit)
            items.append({
                "item_id": f"{battery_key}:{arm}:{unit_id}",
                "battery": battery_key,
                "arm": arm,
                "family": row["scenario_id"],
                "unit": {f: row[f] for f in spec["unit_fields"]},
                "source_file": spec["arms"][arm],
                "source_cell": row.get("cell"),
                "pinned": dict(spec["pin"]),
                "text": row[spec["rated_field"]],
            })
    return items, sources


def render_rating_prompt(text, question_key, prompt_version):
    """Blind judge input: rated text + versioned question/scale ONLY."""
    spec = PRETEST_PROMPTS[prompt_version]
    q = spec["questions"][question_key]
    return spec["template"].format(text=text, question=q["question"],
                                   scale=q["scale"], low=q["low"],
                                   high=q["high"])


def parse_rating(text, low, high):
    """Strict whole-response integer parse; no coercion."""
    if text is None:
        return None, "no_text"
    token = text.strip().strip(".,:;!*\"'")
    if not token.lstrip("+-").isdigit():
        return None, "not_an_integer"
    value = int(token)
    if not low <= value <= high:
        return None, "out_of_scale"
    return value, "parsed"


def rate_item(transport, judge_model, rendered_prompt, low, high,
              max_attempts=3, max_tokens=8, temperature=0.0,
              sleep_fn=time.sleep):
    """One rating with full attempt persistence; UNKNOWN on failure.

    Reuses the E27 judge transport/attempt/retry machinery with the strict
    integer-rating parser plugged in, so a valid rating on attempt 1 is
    accepted immediately and failures stay explicit.
    """
    verdict = judge_event(
        transport, judge_model, rendered_prompt,
        max_attempts=max_attempts, max_tokens=max_tokens,
        temperature=temperature, sleep_fn=sleep_fn,
        parse_fn=lambda text: parse_rating(text, low, high),
    )
    rated = verdict["label_status"] == "judged"
    return {"rating": verdict["label"] if rated else None,
            "rating_status": "rated" if rated else verdict["label_status"],
            "attempts": verdict["attempts"]}


def summarize(items_with_ratings, questions):
    """Per-question arm summaries + per-family means; UNKNOWNs separate."""
    summaries = {}
    for question in questions:
        by_arm = defaultdict(list)
        by_arm_family = defaultdict(lambda: defaultdict(list))
        n_unknown = defaultdict(int)
        for item in items_with_ratings:
            rating = item["ratings"][question]["rating"]
            if rating is None:
                n_unknown[item["arm"]] += 1
                continue
            by_arm[item["arm"]].append(rating)
            by_arm_family[item["arm"]][item["family"]].append(rating)
        summaries[question] = {
            arm: {
                "mean": float(np.mean(values)) if values else None,
                "n": len(values),
                "n_unknown": n_unknown.get(arm, 0),
                "ratings": values,
                "per_family_mean": {
                    family: float(np.mean(fam_values))
                    for family, fam_values in sorted(
                        by_arm_family[arm].items())
                },
            }
            for arm, values in sorted(by_arm.items())
        }
        for arm, count in n_unknown.items():  # arms with zero parsed ratings
            summaries[question].setdefault(arm, {
                "mean": None, "n": 0, "n_unknown": count, "ratings": [],
                "per_family_mean": {}})
    return summaries


def resolve_out_path(out_arg, repo_root=ROOT, protected=PROTECTED_OUTPUTS):
    out = Path(out_arg)
    resolved = out if out.is_absolute() else Path(repo_root) / out
    for protected_path in protected:
        if resolved.resolve() == (Path(repo_root) / protected_path).resolve():
            raise SystemExit(
                f"refusing to write to {resolved}: historical pretest "
                "artifacts are preserved as-is; choose a new output path"
            )
    if resolved.exists():
        raise SystemExit(
            f"refusing to overwrite existing {resolved}; choose a new "
            "output path"
        )
    return resolved


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", required=True, choices=sorted(BATTERIES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge-model", default=None,
                    help="REQUIRED for live rating; pin an exact model "
                         "version for accepted runs")
    ap.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION,
                    choices=sorted(PRETEST_PROMPTS))
    ap.add_argument("--shuffle-seed", type=int, default=0,
                    help="deterministic randomized blinded presentation order")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--sleep", type=float, default=0.15)
    ap.add_argument("--dry-run", action="store_true",
                    help="export exact rendered judge inputs; no API access")
    args = ap.parse_args(argv)

    out = resolve_out_path(args.out)
    spec = BATTERIES[args.battery]
    questions = spec["questions"]
    prompt_spec = PRETEST_PROMPTS[args.prompt_version]
    for question in questions:
        if question not in prompt_spec["questions"]:
            raise SystemExit(f"prompt version {args.prompt_version} lacks "
                             f"question {question!r}")

    items, sources = extract_items(args.battery)
    order = list(range(len(items)))
    rng = np.random.default_rng(args.shuffle_seed)
    rng.shuffle(order)
    print(f"battery {args.battery}: {len(items)} items across "
          f"{len(set(i['arm'] for i in items))} arms")

    result = {
        "schema": "manipulation_pretest/1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "battery": args.battery,
        "battery_spec": {
            "arms": spec["arms"], "arm_field": spec["arm_field"],
            "pin": spec["pin"], "unit_fields": list(spec["unit_fields"]),
            "rated_field": spec["rated_field"],
            "questions": list(questions),
        },
        "sources": sources,
        "judge": {
            "model": args.judge_model,
            "prompt_version": args.prompt_version,
            "prompt_spec": prompt_spec,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "max_attempts": args.max_attempts,
            "api": None if args.dry_run else "anthropic-messages",
            "blinding": "judge sees rated text + versioned question/scale "
                        "only; presentation order shuffled across arms",
        },
        "presentation": {"shuffle_seed": args.shuffle_seed,
                         "order_item_ids": [items[i]["item_id"] for i in order]},
        "provenance": collect_run_provenance(),
    }

    if args.dry_run:
        result["mode"] = "dry_run_export"
        result["items"] = [
            {**item,
             "rendered_judge_inputs": {
                 question: render_rating_prompt(item["text"], question,
                                                args.prompt_version)
                 for question in questions}}
            for item in items
        ]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1))
        print(f"dry run: wrote exact judge inputs for {len(items)} items -> {out}")
        return 0

    if not args.judge_model:
        ap.error("--judge-model is required for live rating (pin an exact "
                 "version for accepted runs); use --dry-run for offline export")
    transport = anthropic_transport(os.environ["ANTHROPIC_API_KEY"])

    rated = [None] * len(items)
    for n_done, idx in enumerate(order, start=1):
        item = items[idx]
        ratings = {}
        for question in questions:
            q = prompt_spec["questions"][question]
            rendered = render_rating_prompt(item["text"], question,
                                            args.prompt_version)
            ratings[question] = {
                "rendered_judge_input": rendered,
                **rate_item(transport, args.judge_model, rendered,
                            q["low"], q["high"],
                            max_attempts=args.max_attempts,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature),
            }
            time.sleep(args.sleep)
        rated[idx] = {**item, "presentation_index": n_done - 1,
                      "ratings": ratings}
        if n_done % 20 == 0:
            print(f"rated {n_done}/{len(items)}")

    result["mode"] = "rated"
    result["items"] = rated
    result["summaries"] = summarize(rated, questions)
    n_unknown = sum(1 for item in rated for question in questions
                    if item["ratings"][question]["rating"] is None)
    result["n_unknown_ratings"] = n_unknown

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    for question, arms in result["summaries"].items():
        print(f"[{question}] " + "  ".join(
            f"{arm}: {info['mean'] if info['mean'] is not None else 'NA'} "
            f"(n={info['n']}, unk={info['n_unknown']})"
            for arm, info in arms.items()))
    print(f"wrote {out}")
    if n_unknown:
        raise SystemExit(f"{n_unknown} ratings could not be parsed (UNKNOWN) "
                         "— artifact written but summaries are incomplete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
