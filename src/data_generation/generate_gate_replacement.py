"""Generate one provenance-preserving pre-score blueprint replacement."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from .generate_gate_blueprints import (
        KINDS, REPLACEMENTS, ROOT, SOURCES, V2_KIND_CONSTRAINTS,
        V3_KIND_CONSTRAINTS, V4_KIND_CONSTRAINTS, load_env, output_path,
        parse_json_object, request_source, validate_families)
except ImportError:  # direct script execution
    from generate_gate_blueprints import (
        KINDS, REPLACEMENTS, ROOT, SOURCES, V2_KIND_CONSTRAINTS,
        V3_KIND_CONSTRAINTS, V4_KIND_CONSTRAINTS, load_env, output_path,
        parse_json_object, request_source, validate_families)


def prompt_for_replacement(kind: str, domain: str, fields: tuple[str, ...],
                           exclusions: list[str], revision: int) -> str:
    field_lines = "\n".join(f'- "{field}": string' for field in fields)
    return f"""Create exactly one replacement semantic blueprint for a scientific interpretability dataset.

Return exactly one JSON object with one key, "family", containing exactly these fields:
{field_lines}

The domain field must be exactly "{domain}". The scenario must not concern any
of these existing concepts or phrases: {json.dumps(exclusions)}.

Use an ordinary, non-famous, non-extreme situation with a concrete live
objective. No death, violence, crime, medical emergency, protected-class
content, hospital, ambulance, bleeding, eviction, homelessness, or physical
danger. Messages are under 24 words and avoid the words empathy, empathic,
welfare, moral, ethical, task, objective, urgent, resolved, simulated, and
archived. Do not write experiment labels, branch text, A/B options, or a final
decision. No Markdown and no extra keys.
{V2_KIND_CONSTRAINTS[kind]}
{V3_KIND_CONSTRAINTS.get(kind, "") if revision >= 3 else ""}
{V4_KIND_CONSTRAINTS.get(kind, "") if revision >= 4 else ""}
"""


def generate(kind: str, source: str, family_id: str, revision: int,
             reason: str, exclusions: list[str], retries: int = 8) -> Path:
    base_path = output_path(kind, source, revision)
    raw = base_path.read_bytes()
    base = json.loads(raw)
    target_index = next(
        index for index, row in enumerate(base["families"])
        if row["family_id"] == family_id)
    target = base["families"][target_index]
    prompt = prompt_for_replacement(
        kind, target["domain"], KINDS[kind]["fields"], exclusions, revision)
    errors, previous = [], None
    for attempt in range(1, retries + 1):
        try:
            repair = ""
            if errors:
                repair = ("\nThe previous candidate failed validation: " +
                          errors[-1]["error"] +
                          ". Return a corrected complete family.\nRejected JSON:\n" +
                          (previous or "<not parseable>"))
            text, metadata = request_source(source, prompt + repair)
            previous = text
            payload = parse_json_object(text)
            if set(payload) != {"family"}:
                raise ValueError("response must be exactly {'family': {...}}")
            candidate_rows = []
            for index, row in enumerate(base["families"]):
                candidate = payload["family"] if index == target_index else row
                candidate_rows.append({field: candidate[field]
                                       for field in KINDS[kind]["fields"]})
            validated = validate_families(
                kind, source, {"families": candidate_rows}, strict=True,
                revision=revision)
            replacement = validated[target_index]
            if any(term.lower() in " ".join(replacement.values()).lower()
                   for term in exclusions):
                raise ValueError("replacement contains an excluded phrase")
            artifact = {
                "schema": "empathy-action-probes/gate-blueprint-replacement/1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "kind": kind, "source": source, "revision": revision,
                "family_id": family_id, "reason": reason,
                "exclusions": exclusions,
                "base_path": str(base_path.relative_to(ROOT)),
                "base_sha256": hashlib.sha256(raw).hexdigest(),
                "prompt": prompt, "response_metadata": metadata,
                "errors_before_success": errors,
                "replacement": replacement,
            }
            REPLACEMENTS.mkdir(parents=True, exist_ok=True)
            path = REPLACEMENTS / f"{family_id}_r1.json"
            if path.exists():
                raise FileExistsError(path)
            path.write_text(json.dumps(artifact, indent=2) + "\n")
            print(f"wrote replacement -> {path}")
            return path
        except Exception as exc:
            errors.append({"attempt": attempt,
                           "error": f"{type(exc).__name__}: {exc}"})
            print(f"attempt {attempt}/{retries} failed: {errors[-1]['error']}",
                  flush=True)
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=sorted(KINDS), required=True)
    parser.add_argument("--source", choices=sorted(SOURCES), required=True)
    parser.add_argument("--family-id", required=True)
    parser.add_argument("--revision", type=int, default=2)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--exclude", action="append", default=[])
    args = parser.parse_args(argv)
    load_env()
    generate(args.kind, args.source, args.family_id, args.revision,
             args.reason, args.exclude)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
