"""Audit Gate 1/2 semantic blueprints before deterministic materialization.

This is a structural and leakage screen, not a manipulation check.  It catches
provider/provenance mismatches, prompt-constraint violations, duplicate or
near-duplicate scenarios, and obvious field-role reversals.  Semantic review
flags remain visible for model and human adjudication; they are never silently
converted into exclusions after a sealed target-model score is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

try:
    from .generate_gate_blueprints import KINDS, OUT, SOURCES
except ImportError:  # direct script execution
    from generate_gate_blueprints import KINDS, OUT, SOURCES

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "empathy-action-probes/gate-blueprint-audit/1"

MESSAGE_FIELDS = {
    "urgent_message", "resolved_message", "distress_quote",
    "neutral_message", "neutral_social_message", "excited_message",
}
FORBIDDEN_MESSAGE_WORDS = {
    "empathy", "empathic", "welfare", "moral", "ethical", "task",
    "objective", "urgent", "resolved", "simulated", "archived",
}
UNSAFE_WORDS = {
    "suicide", "kill", "killing", "murder", "weapon", "death",
    "dying", "drowning", "assault", "robbery", "crime", "hospital",
    "medical emergency",
}

DEFER_HELP = re.compile(
    r"\b(after|later|postpon|once|schedule|eventually|when .* (?:done|finished))\b",
    re.IGNORECASE,
)
HELP_CUES = re.compile(
    r"\b(help|assist|respond|review|explain|guide|support|discuss|investigate|"
    r"search|listen|provide|share|reset|release|carry|retrieve|invite|access|"
    r"pause|stop|leave|switch|check|comfort|join|attend|show|step in|take over|"
    r"work with|walk through)\b",
    re.IGNORECASE,
)
SOCIAL_SIGNAL_CUES = re.compile(
    r"\b(person|colleague|friend|child|parent|neighbor|volunteer|user|patron|"
    r"member|client|employee|citizen|driver|staff|voice|tone|gaze|expression|"
    r"body language|anxiety|frustration|impatience|enthusiasm|nervousness|"
    r"message|chat|call|text|speaking|fidget|looking)\b",
    re.IGNORECASE,
)
RESOLUTION_CUES = re.compile(
    r"\b(done|fixed|ready|finished|complete|completed|confirmed|approved|"
    r"addressed|covered|set|calmed|resolved|submitted|delivered|departed|"
    r"enrolled|updated|reviewed|booked|prepared|found|handled|sorted|solved|"
    r"settled|working now|all good)\b",
    re.IGNORECASE,
)
ZERO_COST_CUES = re.compile(
    r"\b(no |without|won't|will not|brief|few minutes|moment|short pause|"
    r"no immediate|without consequence|doesn't|does not)\b",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def words(text: str) -> list[str]:
    return re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE)


def char_ngrams(text: str, n: int = 5) -> set[str]:
    compact = normalize(text)
    if len(compact) < n:
        return {compact} if compact else set()
    return {compact[i:i + n] for i in range(len(compact) - n + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def semantic_text(item: dict) -> str:
    ignored = {"family_id", "kind", "source", "source_model"}
    return " ".join(str(value) for key, value in item.items() if key not in ignored)


@dataclass
class Finding:
    severity: str
    code: str
    detail: str

    def as_dict(self) -> dict:
        return {"severity": self.severity, "code": self.code,
                "detail": self.detail}


@dataclass
class FamilyAudit:
    family_id: str
    kind: str
    source: str
    findings: list[Finding] = field(default_factory=list)

    def add(self, severity: str, code: str, detail: str) -> None:
        self.findings.append(Finding(severity, code, detail))

    @property
    def hard_valid(self) -> bool:
        return not any(f.severity == "hard" for f in self.findings)

    def as_dict(self) -> dict:
        return {
            "family_id": self.family_id,
            "kind": self.kind,
            "source": self.source,
            "hard_valid": self.hard_valid,
            "findings": [finding.as_dict() for finding in self.findings],
        }


def audit_family(item: dict) -> FamilyAudit:
    audit = FamilyAudit(item["family_id"], item["kind"], item["source"])

    for key in MESSAGE_FIELDS & item.keys():
        value = item[key]
        count = len(words(value))
        if count > 24:
            audit.add("hard", "message_too_long", f"{key} has {count} words")
        present = sorted(word for word in FORBIDDEN_MESSAGE_WORDS
                         if re.search(rf"\b{re.escape(word)}\b", value,
                                      flags=re.IGNORECASE))
        if present:
            audit.add("hard", "forbidden_message_word",
                      f"{key} contains {present}")
        if not re.search(r"\b(I|I'm|I've|I'd|my|me|we|we're|we've|our|us)\b",
                         value, flags=re.IGNORECASE):
            audit.add("review", "message_not_first_person",
                      f"{key} may not be a first-person utterance")

    unsafe = sorted(
        term for term in UNSAFE_WORDS
        if re.search(rf"\b{re.escape(term)}\b", semantic_text(item),
                     flags=re.IGNORECASE)
    )
    if unsafe:
        audit.add("hard", "unsafe_content", f"contains {unsafe}")

    if item["kind"] == "writer":
        help_action = item["help_action"]
        if DEFER_HELP.search(help_action):
            audit.add("hard", "help_action_deferred",
                      "help_action postpones rather than performs the response")
        if not HELP_CUES.search(help_action):
            audit.add("review", "help_action_weak",
                      "help_action lacks an immediate response cue")
        for key in ("nonsocial_signal", "nonsocial_action"):
            if SOCIAL_SIGNAL_CUES.search(item[key]):
                audit.add("hard", "social_task_control",
                          f"{key} contains an apparent person/social cue")
        if len(words(item["interruption_loss"])) < 3:
            audit.add("review", "weak_interruption_loss",
                      "interruption_loss is too underspecified")

    elif item["kind"] == "r2b":
        if not RESOLUTION_CUES.search(item["resolved_message"]):
            audit.add("review", "resolved_state_unclear",
                      "resolved_message lacks a clear completion cue")
        if not ZERO_COST_CUES.search(item["cost_zero"]):
            audit.add("review", "zero_cost_unclear",
                      "cost_zero lacks an explicit negligible-loss cue")
        costs = [normalize(item[f"cost_{level}"])
                 for level in ("zero", "low", "medium", "high")]
        if len(set(costs)) != 4:
            audit.add("hard", "duplicate_cost_levels",
                      "cost descriptions are not all distinct")

    elif item["kind"] == "wp1":
        if normalize(item["warm_rendering"]) == normalize(item["terse_rendering"]):
            audit.add("hard", "identical_register_renderings",
                      "warm and terse renderings are identical")
        if normalize(item["genuine_motive"]) == normalize(item["strategic_motive"]):
            audit.add("hard", "identical_motives",
                      "genuine and strategic motives are identical")

    elif item["kind"] == "wp3":
        if normalize(item["actual_context"]) == normalize(item["simulated_context"]):
            audit.add("hard", "identical_actuality_contexts",
                      "actual and simulated contexts are identical")
        costs = [normalize(item[f"cost_{level}"])
                 for level in ("zero", "low", "high")]
        if len(set(costs)) != 3:
            audit.add("hard", "duplicate_cost_levels",
                      "WP3 cost descriptions are not all distinct")

    return audit


def load_artifacts(directory: Path, revision: int = 1) -> tuple[list[dict], list[dict], list[str]]:
    artifacts, families, top_errors = [], [], []
    for kind, spec in KINDS.items():
        for source, source_spec in SOURCES.items():
            suffix = "" if revision == 1 else f"_v{revision}"
            path = directory / f"{kind}_{source}{suffix}.json"
            if not path.exists():
                top_errors.append(f"missing artifact: {path.name}")
                continue
            raw = path.read_bytes()
            artifact = json.loads(raw)
            expected_count = spec["count_per_source"]
            checks = {
                "schema": artifact.get("schema") ==
                    "empathy-action-probes/gate-family-blueprints/1",
                "kind": artifact.get("kind") == kind,
                "source": artifact.get("source") == source,
                "source_spec": artifact.get("source_spec") == source_spec,
                "count": len(artifact.get("families", [])) == expected_count,
                "revision": artifact.get("revision", 1) == revision,
            }
            failed = [name for name, passed in checks.items() if not passed]
            if failed:
                top_errors.append(f"{path.name}: failed {failed}")
            artifacts.append({
                "path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "kind": kind,
                "source": source,
                "count": len(artifact.get("families", [])),
                "checks": checks,
            })
            families.extend(artifact.get("families", []))
    return artifacts, families, top_errors


def audit_directory(directory: Path, revision: int = 1) -> dict:
    artifacts, families, top_errors = load_artifacts(directory, revision)
    audits = {item["family_id"]: audit_family(item) for item in families}

    ids = [item["family_id"] for item in families]
    for family_id, count in Counter(ids).items():
        if count > 1:
            audits[family_id].add("hard", "duplicate_family_id",
                                  f"family_id appears {count} times")

    exact_seen: dict[str, str] = {}
    ngrams = {item["family_id"]: char_ngrams(semantic_text(item))
              for item in families}
    for item in families:
        family_id = item["family_id"]
        fingerprint = normalize(semantic_text(item))
        previous = exact_seen.get(fingerprint)
        if previous:
            audits[family_id].add("hard", "exact_semantic_duplicate", previous)
            audits[previous].add("hard", "exact_semantic_duplicate", family_id)
        exact_seen[fingerprint] = family_id

    near_pairs = []
    for left_index, left in enumerate(families):
        for right in families[left_index + 1:]:
            score = jaccard(ngrams[left["family_id"]], ngrams[right["family_id"]])
            if score >= 0.70:
                detail_left = f"{right['family_id']} (5-gram Jaccard={score:.3f})"
                detail_right = f"{left['family_id']} (5-gram Jaccard={score:.3f})"
                audits[left["family_id"]].add("review", "near_duplicate", detail_left)
                audits[right["family_id"]].add("review", "near_duplicate", detail_right)
                near_pairs.append({"left": left["family_id"],
                                   "right": right["family_id"],
                                   "jaccard": round(score, 6)})

    rows = [audits[item["family_id"]].as_dict() for item in families]
    severity_counts = Counter(
        finding["severity"] for row in rows for finding in row["findings"])
    code_counts = Counter(
        finding["code"] for row in rows for finding in row["findings"])
    hard_valid = sum(row["hard_valid"] for row in rows)
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(directory.relative_to(ROOT)),
        "blueprint_revision": revision,
        "artifacts": artifacts,
        "top_level_errors": top_errors,
        "summary": {
            "artifact_count": len(artifacts),
            "family_count": len(families),
            "hard_valid_families": hard_valid,
            "hard_invalid_families": len(families) - hard_valid,
            "finding_severity_counts": dict(sorted(severity_counts.items())),
            "finding_code_counts": dict(sorted(code_counts.items())),
            "near_duplicate_pair_count": len(near_pairs),
        },
        "families": rows,
        "near_duplicate_pairs": near_pairs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=OUT)
    parser.add_argument("--revision", type=int, default=1)
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "data/gate_families/blueprint_audit.json")
    parser.add_argument("--fail-on-hard", action="store_true")
    args = parser.parse_args(argv)

    report = audit_directory(args.input, args.revision)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {args.report}")
    failed = bool(report["top_level_errors"] or
                  report["summary"]["hard_invalid_families"])
    return int(args.fail_on_hard and failed)


if __name__ == "__main__":
    raise SystemExit(main())
