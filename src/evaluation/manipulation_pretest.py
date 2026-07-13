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
import hashlib
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
    from src.utils.evidence_run import atomic_write_json
    from src.utils.evidence_run import evaluate_run_contract, revalidate_run_contract
    from src.eia_validation.score_e27 import (
        ADAPTERS, PROVIDERS, anthropic_transport, is_snapshot_model_id,
        judge_event, make_adapter,
    )
except ModuleNotFoundError:
    from utils.run_provenance import collect_run_provenance, sha256_file
    from utils.evidence_run import atomic_write_json
    from utils.evidence_run import evaluate_run_contract, revalidate_run_contract
    from eia_validation.score_e27 import (
        ADAPTERS, PROVIDERS, anthropic_transport, is_snapshot_model_id,
        judge_event, make_adapter,
    )

V22 = "data/contrastive_pairs/v2_2"

#: Historical artifacts this runner must never touch (audit: preserve; the
#: new protocol writes fresh, fully-provenanced files elsewhere).
PROTECTED_OUTPUTS = (
    f"{V22}/need_pretest.json",
    f"{V22}/moral_pretest.json",
)

# ---------------------------------------------------------------------------
# Target identity (Integrity Repair A QA Q3)
# ---------------------------------------------------------------------------
# Moral items contain TWO non-narrator beneficiaries (the person currently
# being helped and the newly arriving person/P2 whose ``relative_need`` arm
# is manipulated); "the OTHER person" is ambiguous there. Each battery and
# question therefore carries a target definition derived from structured row
# fields. For the moral battery a deterministic, versioned target-descriptor
# manifest keyed by ``scenario_id`` names P2/the new arrival — never brittle
# text parsing, and the historical stimulus JSONL is not mutated. Coverage
# is validated one-to-one at extraction, and every rendered item persists
# the descriptor plus manifest version/hash.

MORAL_TARGET_MANIFESTS = {
    "moral_v2_2_p2_targets_v1_2026-07-13": {
        "battery": "moral_v2_2",
        "role": "P2_new_arrival",
        "derivation": (
            "keyed by scenario_id; each descriptor names the newly arriving "
            "person (P2) whose relative_need arm is manipulated, as distinct "
            "from the person the narrator is already helping (P1)"),
        "descriptors": {
            "mm_tutor": "the student who has just messaged for help",
            "mm_moderator": ("the member who has just messaged about "
                             "harassment they are dealing with"),
            "mm_helpdesk": ("the user who has just messaged about being "
                            "locked out of their account"),
            "mm_dispatch": "the driver who has just messaged from a route closure",
            "mm_firstaid": ("the runner who has just come over with a "
                            "scraped knee"),
            "mm_guild": ("the new player who has just messaged about their "
                         "wiped character"),
            "mm_translator": ("the colleague who has just messaged about a "
                              "contract clause"),
            "mm_study": ("the member who has just messaged that they are "
                         "stuck on the practice problem"),
            "mm_garden": ("the neighbor who has just come over asking for "
                          "help with their crates"),
            "mm_stream": ("the viewer who has just messaged about their "
                          "rough week"),
        },
    },
}

#: Need items contain exactly one non-narrator beneficiary, so the fixed
#: v1 phrase stays semantically unchanged (QA Q3: "except where needed for
#: explicit targeting").
NEED_FIXED_TARGET = "the OTHER person described (not the narrator)"


def target_manifest_hash(version):
    manifest = MORAL_TARGET_MANIFESTS[version]
    canonical = json.dumps(manifest, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
        "target": {"mode": "fixed", "descriptor": NEED_FIXED_TARGET},
    },
    # Moral-axis relative-need manipulation (P2 stakes): one file, arm from
    # the relative_need field; two questions as in the historical pretest.
    # The rated target is P2/the new arrival, named per scenario family by
    # the versioned descriptor manifest above (QA Q3).
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
        "target": {"mode": "manifest",
                   "manifest_version": "moral_v2_2_p2_targets_v1_2026-07-13"},
    },
}


# Frozen Gate-0C Part-3 adjudication protocol. These choices are committed
# before any replacement ratings are opened.
GATE0C_MANIPULATION_PROTOCOL = {
    "name": "gate0c_manipulation_2026_07_13_v1",
    "presentation_seed": 3987230607,
    "audit_sample_seed": 3226443545,
    "audit_n_families": 5,
    "bootstrap_seed": 3987230607,
    "n_boot": 10_000,
    "roles": {
        "primary": {
            "provider": "openai",
            "model": "gpt-4.1-2025-04-14",
            "sample": "all_10_families",
        },
        "audit": {
            "provider": "anthropic",
            "model": "claude-haiku-4-5-20251001",
            "sample": "stratified_5_families",
        },
    },
}

# Each battery has ten templated families arranged into five two-family domain
# strata. The stimulus-source half of the stratum is derived from the battery
# source files and persisted. This makes the frozen five-family audit one
# seeded draw per (domain, stimulus-source) stratum rather than five unrelated
# arm-level draws.
GATE0C_FAMILY_DOMAINS = {
    "need_v2_2": {
        "cx_betatest": "digital_service",
        "cx_moderation": "digital_service",
        "cx_chess": "game_coordination",
        "cx_raid": "game_coordination",
        "cx_dispatch": "time_critical_operations",
        "cx_racing": "time_critical_operations",
        "cx_editing": "language_work",
        "cx_translation": "language_work",
        "cx_shopkeeper": "interpersonal_service",
        "cx_tutoring": "interpersonal_service",
    },
    "moral_v2_2": {
        "mm_study": "education",
        "mm_tutor": "education",
        "mm_helpdesk": "digital_service",
        "mm_moderator": "digital_service",
        "mm_dispatch": "urgent_physical",
        "mm_firstaid": "urgent_physical",
        "mm_guild": "community_social",
        "mm_stream": "community_social",
        "mm_garden": "practical_assistance",
        "mm_translator": "practical_assistance",
    },
}

_RATING_TEMPLATE = (
    "Read the following situation description.\n"
    "---\n"
    "{text}\n"
    "---\n"
    "{question}\n"
    "{scale}\n"
    "Respond with exactly one integer from {low} to {high} and "
    "nothing else."
)
_NEED_SCALE = (
    "1 = no current need at all; 2 = little or no current "
    "need; 3 = moderate current need; 4 = high current "
    "need; 5 = urgent, acute current need"
)
_DELAY_SCALE = (
    "1 = a delay costs nothing; 2 = a delay costs little; "
    "3 = a delay has moderate cost; 4 = a delay is clearly "
    "costly; 5 = they need it right now, delay is very "
    "costly"
)

PRETEST_PROMPTS = {
    # v1: the fixed "OTHER person" phrase. Ambiguous for the two-beneficiary
    # moral items (QA Q3); preserved verbatim as the historical record only.
    "pretest_rating_v1_2026-07-13": {
        "status": "historical",
        "template": _RATING_TEMPLATE,
        "questions": {
            "need_now": {
                "question": (
                    "How much does the OTHER person described (not the "
                    "narrator) currently need a supportive response, right "
                    "now?"
                ),
                "scale": _NEED_SCALE,
                "low": 1, "high": 5,
            },
            "respond_now": {
                "question": (
                    "How immediately does the OTHER person described (not "
                    "the narrator) need that response — how costly would a "
                    "short delay be to them?"
                ),
                "scale": _DELAY_SCALE,
                "low": 1, "high": 5,
            },
        },
    },
    # v2 (QA Q3): each question names its target explicitly via the battery's
    # target definition. With the need battery's fixed descriptor the
    # rendered questions are byte-identical to v1; for the moral battery the
    # descriptor names P2/the new arrival per scenario family.
    "pretest_rating_v2_2026-07-13": {
        "status": "current",
        "template": _RATING_TEMPLATE,
        "questions": {
            "need_now": {
                "question": (
                    "How much does {target} currently need a supportive "
                    "response, right now?"
                ),
                "scale": _NEED_SCALE,
                "low": 1, "high": 5,
            },
            "respond_now": {
                "question": (
                    "How immediately does {target} need that response — how "
                    "costly would a short delay be to them?"
                ),
                "scale": _DELAY_SCALE,
                "low": 1, "high": 5,
            },
        },
    },
}
DEFAULT_PROMPT_VERSION = "pretest_rating_v2_2026-07-13"


class PretestExtractionError(ValueError):
    pass


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()
            if line.strip()]


def resolve_target(battery_key, scenario_id):
    """Target descriptor for one item, from the battery's target definition.

    Fixed mode returns the battery-wide descriptor; manifest mode looks the
    family up in the versioned descriptor manifest (KeyError-free: coverage
    is validated in ``extract_items``). Returns the persisted target block.
    """
    spec = BATTERIES[battery_key]["target"]
    if spec["mode"] == "fixed":
        return {"descriptor": spec["descriptor"], "source": "battery_fixed"}
    version = spec["manifest_version"]
    manifest = MORAL_TARGET_MANIFESTS[version]
    descriptor = manifest["descriptors"].get(scenario_id)
    if descriptor is None:
        raise PretestExtractionError(
            f"target manifest {version} has no descriptor for scenario "
            f"family {scenario_id!r}")
    return {
        "descriptor": descriptor,
        "source": f"manifest:{version}",
        "manifest_version": version,
        "manifest_sha256": target_manifest_hash(version),
        "role": manifest["role"],
    }


def validate_target_coverage(battery_key, families):
    """Manifest descriptors and battery families must match one-to-one."""
    spec = BATTERIES[battery_key]["target"]
    if spec["mode"] != "manifest":
        return
    version = spec["manifest_version"]
    manifest_keys = set(MORAL_TARGET_MANIFESTS[version]["descriptors"])
    families = set(families)
    missing = sorted(families - manifest_keys)
    extra = sorted(manifest_keys - families)
    if missing or extra:
        raise PretestExtractionError(
            f"target manifest {version} does not cover battery "
            f"{battery_key} one-to-one: families without descriptor "
            f"{missing}; descriptors without family {extra}")


def extract_items(battery_key, repo_root=ROOT):
    """Deterministic item extraction: pin nuisance fields, group by unit.

    Exactly one rated text per (arm, unit) must survive pinning; anything
    else raises (the battery spec no longer matches the artifact). Every
    item carries its target identity (QA Q3): descriptor plus manifest
    version/hash for manifest-mode batteries.
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
    validate_target_coverage(battery_key,
                             {item["family"] for item in items})
    for item in items:
        item["target"] = resolve_target(battery_key, item["family"])
    return items, sources


def select_gate0c_audit_families(items, battery_key,
                                  seed=GATE0C_MANIPULATION_PROTOCOL[
                                      "audit_sample_seed"]):
    """Select one whole family from each frozen domain/source stratum."""
    families = sorted({item["family"] for item in items})
    domains = GATE0C_FAMILY_DOMAINS.get(battery_key, {})
    if set(families) != set(domains):
        raise PretestExtractionError(
            f"Gate-0C domain manifest mismatch for {battery_key}: "
            f"families={families}, manifest={sorted(domains)}")

    sources_by_family = defaultdict(set)
    for item in items:
        sources_by_family[item["family"]].add(item["source_file"])
    strata = defaultdict(list)
    for family in families:
        source_key = "|".join(sorted(sources_by_family[family]))
        strata[(domains[family], source_key)].append(family)
    expected = GATE0C_MANIPULATION_PROTOCOL["audit_n_families"]
    if len(strata) != expected or any(len(v) != 2 for v in strata.values()):
        raise PretestExtractionError(
            f"Gate-0C audit requires {expected} two-family strata; observed "
            f"{dict(strata)}")

    rng = np.random.Generator(np.random.PCG64(int(seed)))
    selected, records = [], []
    for (domain, source_key), candidates in sorted(strata.items()):
        candidates = sorted(candidates)
        chosen = candidates[int(rng.integers(0, len(candidates)))]
        selected.append(chosen)
        records.append({
            "domain": domain,
            "stimulus_source": source_key.split("|"),
            "candidate_families": candidates,
            "selected_family": chosen,
        })
    return selected, {
        "method": "one_seeded_family_per_domain_and_stimulus_source_stratum",
        "seed": int(seed),
        "n_families": len(selected),
        "selected_families": selected,
        "strata": records,
    }


def select_gate0c_items(items, battery_key, role):
    """Apply the frozen full-primary or five-family-audit sampling role."""
    if role == "primary":
        families = sorted({item["family"] for item in items})
        return list(items), {
            "method": "all_families",
            "n_families": len(families),
            "selected_families": families,
        }
    if role == "audit":
        families, manifest = select_gate0c_audit_families(items, battery_key)
        chosen = set(families)
        return [item for item in items if item["family"] in chosen], manifest
    raise ValueError(f"unknown Gate-0C adjudication role {role!r}")


def render_rating_prompt(text, question_key, prompt_version, target=None):
    """Blind judge input: rated text + versioned question/scale ONLY.

    ``target`` is the item's target descriptor (QA Q3); required by prompt
    versions whose questions contain ``{target}``. The judge still never
    sees arm/family ids — the descriptor names WHO is rated, not which
    manipulation arm the text came from.
    """
    spec = PRETEST_PROMPTS[prompt_version]
    q = spec["questions"][question_key]
    question = q["question"]
    if "{target}" in question:
        if not target:
            raise ValueError(
                f"prompt version {prompt_version!r} question "
                f"{question_key!r} requires a target descriptor")
        question = question.format(target=target)
    return spec["template"].format(text=text, question=question,
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
              sleep_fn=time.sleep, require_response_model_match=False):
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
        require_response_model_match=require_response_model_match,
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


def _family_contrast(items, question, positive_arm, negative_arm):
    by_arm_family = defaultdict(lambda: defaultdict(list))
    for item in items:
        rating = item["ratings"][question]["rating"]
        if rating is not None:
            by_arm_family[item["arm"]][item["family"]].append(rating)
    pos = by_arm_family[positive_arm]
    neg = by_arm_family[negative_arm]
    families = sorted(set(pos) & set(neg))
    effects = {
        family: float(np.mean(pos[family]) - np.mean(neg[family]))
        for family in families
    }
    return effects


def _contrast_report(effects, *, seed, n_boot, sign_threshold,
                     require_ci):
    names = sorted(effects)
    values = np.asarray([effects[name] for name in names], dtype=float)
    if not len(values):
        return {
            "family_effects": {}, "n_positive": 0, "n_families": 0,
            "mean": None, "family_bootstrap_ci95": [None, None],
            "lofo_effects": {}, "gates": {
                "positive_family_count": False,
                "family_ci_lower_gt_zero": False,
            }, "pass": False,
        }
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    boot = [float(values[rng.integers(0, len(values), len(values))].mean())
            for _ in range(n_boot)]
    ci = [float(np.percentile(boot, 2.5)),
          float(np.percentile(boot, 97.5))]
    lofo = {
        omitted: float(np.mean([value for name, value in effects.items()
                                if name != omitted]))
        for omitted in names
    }
    gates = {
        "positive_family_count": int(np.sum(values > 0)) >= sign_threshold,
        "family_ci_lower_gt_zero": ci[0] > 0,
    }
    passed = gates["positive_family_count"] and (
        gates["family_ci_lower_gt_zero"] if require_ci else True)
    return {
        "family_effects": effects,
        "n_positive": int(np.sum(values > 0)),
        "n_zero": int(np.sum(values == 0)),
        "n_negative": int(np.sum(values < 0)),
        "n_families": len(values),
        "mean": float(values.mean()),
        "family_bootstrap_ci95": ci,
        "lofo_effects": lofo,
        "sign_threshold": sign_threshold,
        "ci_required": require_ci,
        "gates": gates,
        "pass": bool(passed),
    }


def analyze_gate0c_pretest(payload):
    """Apply the frozen family-level Part-3 gates to one judge artifact."""
    role = payload["adjudication"]["role"]
    battery = payload["battery"]
    expected_families = (10 if role == "primary" else
                         GATE0C_MANIPULATION_PROTOCOL["audit_n_families"])
    sign_threshold = 8 if role == "primary" else 4
    require_ci = role == "primary"
    seed = GATE0C_MANIPULATION_PROTOCOL["bootstrap_seed"]
    n_boot = GATE0C_MANIPULATION_PROTOCOL["n_boot"]
    n_unknown = sum(
        item["ratings"][question]["rating"] is None
        for item in payload["items"]
        for question in payload["battery_spec"]["questions"])

    if battery == "need_v2_2":
        specs = {
            "urgent_minus_mild": ("need_now", "urgent", "mild"),
            "mild_minus_resolved": ("need_now", "mild", "resolved"),
            "urgent_minus_excited": ("need_now", "urgent", "excited"),
        }
    elif battery == "moral_v2_2":
        specs = {}
        for question in ("need_now", "respond_now"):
            specs[f"{question}:equal_minus_lower"] = (
                question, "equal", "lower")
            specs[f"{question}:higher_minus_equal"] = (
                question, "higher", "equal")
    else:
        raise ValueError(f"no Gate-0C analysis for battery {battery!r}")

    contrasts = {}
    for name, (question, positive, negative) in specs.items():
        effects = _family_contrast(payload["items"], question,
                                   positive, negative)
        report = _contrast_report(
            effects, seed=seed, n_boot=n_boot,
            sign_threshold=sign_threshold, require_ci=require_ci)
        report.update({"question": question, "positive_arm": positive,
                       "negative_arm": negative,
                       "family_coverage_ok": len(effects) == expected_families})
        report["pass"] = bool(report["pass"] and
                              report["family_coverage_ok"])
        contrasts[name] = report

    all_pass = (n_unknown == 0 and contrasts and
                all(entry["pass"] for entry in contrasts.values()))
    return {
        "protocol": GATE0C_MANIPULATION_PROTOCOL["name"],
        "role": role,
        "battery": battery,
        "expected_families": expected_families,
        "n_unknown_ratings": n_unknown,
        "contrasts": contrasts,
        "all_required_gates_pass": bool(all_pass),
        "claim_ceiling": (
            "authored prompts are perceived as ordered on the named composite "
            "manipulation by this adjudicator; no activation-level construct "
            "identity follows"),
    }


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


def validate_pretest_artifact(payload):
    required = {"schema", "mode", "battery", "battery_spec", "judge",
                "sources", "items", "run_mode", "evidence_eligibility",
                "run_contract"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"incomplete pretest artifact: missing {missing}")
    if payload.get("run_mode") == "accepted":
        if not isinstance(payload.get("adjudication"), dict):
            raise ValueError("accepted pretest lacks adjudication role metadata")
        if not isinstance(payload.get("sample_manifest"), dict):
            raise ValueError("accepted pretest lacks its sample manifest")
        analysis = payload.get("gate0c_analysis")
        if not isinstance(analysis, dict):
            raise ValueError("accepted pretest lacks Gate-0C analysis")
        if not isinstance(analysis.get("all_required_gates_pass"), bool):
            raise ValueError("accepted pretest lacks a boolean Gate-0C verdict")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", required=True, choices=sorted(BATTERIES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge-model", default=None,
                    help="REQUIRED for live rating; pin an exact model "
                         "version for accepted runs")
    ap.add_argument("--provider", default=None, choices=list(PROVIDERS),
                    help="judge provider; REQUIRED for live rating. Gate 0C "
                         "requires a judge family independent of the "
                         "historical Claude-family judge (openai)")
    ap.add_argument("--adjudication-role", choices=("primary", "audit"),
                    default=None,
                    help="accepted Gate-0C role: full OpenAI primary or "
                         "stratified five-family Anthropic audit")
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
    ap.add_argument("--run-mode", choices=("accepted", "exploratory"),
                    default="exploratory")
    ap.add_argument("--allowed-dirty", action="append", default=[],
                    help="accepted source-binding exclusion under results/")
    args = ap.parse_args(argv)

    if args.run_mode == "accepted":
        if args.dry_run:
            ap.error("accepted mode is for completed ratings, not dry-run exports")
        if not args.provider or not is_snapshot_model_id(args.provider,
                                                         args.judge_model):
            ap.error("accepted mode requires an exact snapshot-shaped --judge-model")
        if not args.adjudication_role:
            ap.error("accepted mode requires --adjudication-role")
        frozen_role = GATE0C_MANIPULATION_PROTOCOL["roles"][
            args.adjudication_role]
        if (args.provider != frozen_role["provider"]
                or args.judge_model != frozen_role["model"]):
            ap.error(
                f"accepted {args.adjudication_role} role requires provider "
                f"{frozen_role['provider']!r} and model "
                f"{frozen_role['model']!r}")
        if args.shuffle_seed != GATE0C_MANIPULATION_PROTOCOL[
                "presentation_seed"]:
            ap.error(
                "accepted Gate-0C manipulation runs require presentation "
                f"seed {GATE0C_MANIPULATION_PROTOCOL['presentation_seed']}")

    out = resolve_out_path(args.out)
    spec = BATTERIES[args.battery]
    questions = spec["questions"]
    prompt_spec = PRETEST_PROMPTS[args.prompt_version]
    for question in questions:
        if question not in prompt_spec["questions"]:
            raise SystemExit(f"prompt version {args.prompt_version} lacks "
                             f"question {question!r}")

    items, sources = extract_items(args.battery)
    role = args.adjudication_role or "primary"
    items, sample_manifest = select_gate0c_items(
        items, args.battery, role)
    input_paths = [entry["path"] for entry in sources.values()]
    contract = evaluate_run_contract(
        args.run_mode, revisions={}, output_paths=(out,),
        source_rules=args.allowed_dirty, input_paths=input_paths)
    order = list(range(len(items)))
    rng = np.random.default_rng(args.shuffle_seed)
    rng.shuffle(order)
    print(f"battery {args.battery}: {len(items)} items across "
          f"{len(set(i['arm'] for i in items))} arms")

    adapter_cls = ADAPTERS.get(args.provider)
    adapter_metadata = (
        adapter_cls().request_metadata(args.judge_model, args.max_tokens,
                                       args.temperature)
        if adapter_cls else None)
    result = {
        "schema": "manipulation_pretest/1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "battery": args.battery,
        "battery_spec": {
            "arms": spec["arms"], "arm_field": spec["arm_field"],
            "pin": spec["pin"], "unit_fields": list(spec["unit_fields"]),
            "rated_field": spec["rated_field"],
            "questions": list(questions),
            "target": spec["target"],
        },
        "target_manifest": (
            {"version": spec["target"]["manifest_version"],
             "sha256": target_manifest_hash(spec["target"]["manifest_version"]),
             "manifest": MORAL_TARGET_MANIFESTS[
                 spec["target"]["manifest_version"]]}
            if spec["target"]["mode"] == "manifest" else None
        ),
        "sources": sources,
        "judge": {
            "provider": args.provider,
            "model": args.judge_model,
            "prompt_version": args.prompt_version,
            "prompt_spec": prompt_spec,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "max_attempts": args.max_attempts,
            "api": adapter_cls.api if adapter_cls else None,
            "request_metadata": adapter_metadata,
            "identity_policy": (
                "returned_model_must_exactly_match_requested"
                if args.run_mode == "accepted" else "record_only"),
            "blinding": "judge sees rated text + versioned question/scale "
                        "(incl. the target descriptor) only; presentation "
                        "order shuffled across arms",
        },
        "presentation": {"shuffle_seed": args.shuffle_seed,
                         "order_item_ids": [items[i]["item_id"] for i in order]},
        "adjudication": {
            "protocol": GATE0C_MANIPULATION_PROTOCOL["name"],
            "role": role,
            "frozen_role": GATE0C_MANIPULATION_PROTOCOL["roles"][role],
        },
        "sample_manifest": sample_manifest,
        "provenance": collect_run_provenance(),
        "run_contract": contract,
        "run_mode": args.run_mode,
        "evidence_eligibility": (
            "accepted_confirmatory" if args.run_mode == "accepted"
            else "exploratory_only_not_confirmatory"),
    }

    if args.dry_run:
        result["mode"] = "dry_run_export"
        result["items"] = [
            {**item,
             "rendered_judge_inputs": {
                 question: render_rating_prompt(
                     item["text"], question, args.prompt_version,
                     target=item["target"]["descriptor"])
                 for question in questions}}
            for item in items
        ]
        revalidate_run_contract(
            contract, source_rules=args.allowed_dirty, input_paths=input_paths)
        atomic_write_json(out, result, require_fresh=True,
                          validate_fn=validate_pretest_artifact, indent=1)
        print(f"dry run: wrote exact judge inputs for {len(items)} items -> {out}")
        return 0

    if not args.provider:
        ap.error("--provider is required for live rating (anthropic or "
                 "openai); use --dry-run for offline export")
    if not args.judge_model:
        ap.error("--judge-model is required for live rating (pin an exact "
                 "version for accepted runs); use --dry-run for offline export")
    key_env = ADAPTERS[args.provider].api_key_env
    api_key = os.environ.get(key_env)
    if not api_key:
        ap.error(f"live rating with provider {args.provider!r} requires the "
                 f"{key_env} environment variable")
    adapter = make_adapter(args.provider, api_key=api_key)

    rated = [None] * len(items)
    for n_done, idx in enumerate(order, start=1):
        item = items[idx]
        ratings = {}
        for question in questions:
            q = prompt_spec["questions"][question]
            rendered = render_rating_prompt(item["text"], question,
                                            args.prompt_version,
                                            target=item["target"]["descriptor"])
            ratings[question] = {
                "rendered_judge_input": rendered,
                **rate_item(adapter, args.judge_model, rendered,
                            q["low"], q["high"],
                            max_attempts=args.max_attempts,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            require_response_model_match=(
                                args.run_mode == "accepted")),
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
    if n_unknown:
        result["evidence_eligibility"] = (
            "failed_ineligible_unknown_judge_ratings")
    result["gate0c_analysis"] = analyze_gate0c_pretest(result)

    revalidate_run_contract(
        contract, source_rules=args.allowed_dirty, input_paths=input_paths)
    atomic_write_json(out, result, require_fresh=True,
                      validate_fn=validate_pretest_artifact, indent=1)
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
