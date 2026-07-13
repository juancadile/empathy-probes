"""Integrity helpers for deterministic cell builders (Integrity Repair A).

The 2026-07-13 science audit found that the T/T-confirm builders cycled
opener and closing variants in lockstep (``index % 2`` / ``index % 4`` over
``range(8)``), silently repeating variants 0-3 as 4-7: 40 rows but only 20
unique text pairs. Every deterministic builder now goes through this module:

  * uniqueness of ``(pos_text, neg_text)`` pairs is asserted before writing;
  * the expected scenario x variant grid is asserted before writing;
  * an existing artifact with different content is PRESERVED (copy + SHA-256
    + metadata under ``historical/``) before it is replaced — a builder can
    no longer silently overwrite scientific history;
  * preserved copies and their metadata are append-only: attempting to
    overwrite an existing preservation record raises.

Rows themselves stay timestamp-free so regeneration is byte-deterministic;
volatile facts (written_at, git commit) live in a ``*.provenance.json``
sidecar next to the artifact.

Integrity Repair A QA (Q8, 2026-07-13): an unchanged artifact no longer
returns early on byte-identity alone. The existing sidecar is VALIDATED
first — schema, artifact hash, row and unique-pair counts, family list, the
builder's asserted grid facts, and (when recorded) the preserved copy's
path/bytes. Contradictory or missing required provenance FAILS CLOSED; a
contradictory scientific record is never silently rewritten. A missing
sidecar can only be reconstructed by the explicit offline repair command
``python -m src.data_generation.builder_integrity repair-sidecar`` from
verified artifact + preserved-copy bytes, which leaves an audit entry inside
the reconstructed sidecar.
"""

import argparse
import hashlib
import itertools
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


class BuilderIntegrityError(AssertionError):
    """A deterministic builder produced structurally invalid stimuli."""


class SidecarValidationError(BuilderIntegrityError):
    """An existing sidecar contradicts the artifact or is missing/damaged."""


def rows_to_jsonl(rows):
    return "".join(json.dumps(row) + "\n" for row in rows)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def unique_pair_count(rows):
    return len({(row["pos_text"], row["neg_text"]) for row in rows})


def assert_unique_pairs(rows):
    """Every (pos_text, neg_text) pair must be unique across the cell."""
    seen, dupes = {}, []
    for i, row in enumerate(rows):
        key = (row["pos_text"], row["neg_text"])
        if key in seen:
            dupes.append((seen[key], i, row.get("scenario_id")))
        else:
            seen[key] = i
    if dupes:
        detail = "; ".join(
            f"row {j} duplicates row {i} (family {fam})" for i, j, fam in dupes
        )
        raise BuilderIntegrityError(
            f"{len(dupes)} duplicate (pos_text, neg_text) pairs: {detail}"
        )


def assert_variant_grid(rows, expected_families, variant_fields,
                        expected_variant_counts):
    """Every family must contain exactly the full variant Cartesian grid.

    ``variant_fields``: e.g. ("opener_variant", "closing_variant").
    ``expected_variant_counts``: matching sizes, e.g. (2, 4).
    """
    expected_grid = set(itertools.product(
        *[range(n) for n in expected_variant_counts]
    ))
    by_family = {}
    for row in rows:
        combo = tuple(row[field] for field in variant_fields)
        by_family.setdefault(row["scenario_id"], []).append(combo)
    families = sorted(by_family)
    if sorted(expected_families) != families:
        raise BuilderIntegrityError(
            f"family set mismatch: expected {sorted(expected_families)}, "
            f"got {families}"
        )
    for family, combos in by_family.items():
        if len(combos) != len(expected_grid) or set(combos) != expected_grid:
            missing = sorted(expected_grid - set(combos))
            extra = sorted(set(combos) - expected_grid)
            counts = {c: combos.count(c) for c in set(combos) if combos.count(c) > 1}
            raise BuilderIntegrityError(
                f"family {family!r} variant grid invalid: missing={missing} "
                f"extra={extra} duplicated={counts}"
            )


def assert_disjoint_families(rows_a, rows_b, label_a="a", label_b="b"):
    families_a = {row["scenario_id"] for row in rows_a}
    families_b = {row["scenario_id"] for row in rows_b}
    overlap = sorted(families_a & families_b)
    if overlap:
        raise BuilderIntegrityError(
            f"scenario families shared between {label_a} and {label_b}: {overlap}"
        )


def _git_commit(path):
    try:
        return subprocess.run(
            ("git", "-C", str(Path(path).resolve().parent), "rev-parse", "HEAD"),
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
    except Exception:
        return None


def _repo_relative(path):
    """Repo-relative path string when inside a git worktree, else absolute."""
    resolved = Path(path).resolve()
    try:
        top = subprocess.run(
            ("git", "-C", str(resolved.parent), "rev-parse", "--show-toplevel"),
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
        return str(resolved.relative_to(top))
    except Exception:
        return str(resolved)


def preserve_existing_artifact(path, reason, historical_root=None):
    """Copy ``path`` to an append-only historical location with metadata.

    Layout: ``<parent>/historical/<name>.<sha8>/`` containing the byte-exact
    copy plus ``preservation.json``. If the same content was already
    preserved, the existing record is verified and returned. A colliding
    directory with DIFFERENT content raises — preservation records are never
    overwritten.
    """
    path = Path(path)
    content = path.read_bytes()
    sha = hashlib.sha256(content).hexdigest()
    root = Path(historical_root) if historical_root else path.parent / "historical"
    dest_dir = root / f"{path.name}.{sha[:8]}"
    dest_file = dest_dir / path.name
    meta_file = dest_dir / "preservation.json"
    if dest_dir.exists():
        if not dest_file.exists() or hashlib.sha256(
            dest_file.read_bytes()
        ).hexdigest() != sha:
            raise BuilderIntegrityError(
                f"preservation collision at {dest_dir}: existing content "
                "differs; refusing to overwrite a historical record"
            )
        return json.loads(meta_file.read_text()) if meta_file.exists() else {
            "sha256": sha, "preserved_copy": str(dest_file)}
    dest_dir.mkdir(parents=True)
    dest_file.write_bytes(content)
    try:
        rows = [json.loads(line) for line in
                content.decode("utf-8").splitlines() if line.strip()]
        n_rows, n_unique = len(rows), unique_pair_count(rows)
        families = sorted({row.get("scenario_id") for row in rows})
    except Exception:
        n_rows = n_unique = families = None
    metadata = {
        "original_path": _repo_relative(path),
        "preserved_copy": _repo_relative(dest_file),
        "sha256": sha,
        "n_rows": n_rows,
        "n_unique_pairs": n_unique,
        "families": families,
        "reason": reason,
        "preserved_at": datetime.now(timezone.utc).isoformat(),
        "git_commit_at_preservation": _git_commit(path),
    }
    meta_file.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


#: Required sidecar provenance; a sidecar missing any of these fails closed.
REQUIRED_SIDECAR_FIELDS = (
    "artifact", "sha256", "n_rows", "n_unique_pairs", "families", "reason",
    "written_at", "replaced_artifact",
)


def _resolve_recorded_path(recorded, artifact_path):
    """Resolve a sidecar-recorded (possibly repo-relative) path to a file."""
    recorded_path = Path(recorded)
    candidates = [recorded_path]
    if not recorded_path.is_absolute():
        try:
            top = subprocess.run(
                ("git", "-C", str(Path(artifact_path).resolve().parent),
                 "rev-parse", "--show-toplevel"),
                capture_output=True, text=True, timeout=30, check=True,
            ).stdout.strip()
            candidates.append(Path(top) / recorded_path)
        except Exception:
            pass
        candidates.append(Path(artifact_path).resolve().parent / recorded_path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def validate_sidecar(path, rows, payload, expected_families):
    """Validate an existing sidecar against the artifact it describes (Q8).

    Checks schema, artifact hash, row/unique-pair counts, family list, the
    builder's asserted family grid fact, and the preserved copy's existence
    and bytes when one is recorded. Any contradiction raises
    ``SidecarValidationError`` — a contradictory scientific record is never
    silently rewritten.
    """
    path = Path(path)
    sidecar_path = path.with_suffix(path.suffix + ".provenance.json")
    if not sidecar_path.exists():
        raise SidecarValidationError(
            f"required provenance sidecar missing for {path}: refusing the "
            "unchanged-artifact shortcut. Reconstruct it explicitly with "
            "`python -m src.data_generation.builder_integrity repair-sidecar "
            f"--artifact {path}` (offline, audited) after verifying the "
            "artifact bytes."
        )
    try:
        record = json.loads(sidecar_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SidecarValidationError(
            f"sidecar {sidecar_path} is unreadable/unparseable: {exc}"
        ) from exc
    missing = [field for field in REQUIRED_SIDECAR_FIELDS
               if field not in record]
    if missing:
        raise SidecarValidationError(
            f"sidecar {sidecar_path} lacks required provenance fields: "
            f"{missing}")
    problems = []
    if not isinstance(record["reason"], str) or not record["reason"].strip():
        problems.append("reason must be a non-empty string")
    if not isinstance(record["written_at"], str) or not record["written_at"].strip():
        problems.append("written_at must be a non-empty string")
    if not (isinstance(record["sha256"], str)
            and len(record["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in record["sha256"])):
        problems.append("sha256 must be a lowercase 64-hex digest")
    if record["artifact"] != path.name:
        problems.append(f"artifact name {record['artifact']!r} != {path.name!r}")
    payload_sha = sha256_text(payload)
    if record["sha256"] != payload_sha:
        problems.append(
            f"artifact hash {record['sha256']} != recomputed {payload_sha}")
    if record["n_rows"] != len(rows):
        problems.append(f"n_rows {record['n_rows']} != recomputed {len(rows)}")
    recomputed_unique = unique_pair_count(rows)
    if record["n_unique_pairs"] != recomputed_unique:
        problems.append(
            f"n_unique_pairs {record['n_unique_pairs']} != recomputed "
            f"{recomputed_unique}")
    families = sorted({row["scenario_id"] for row in rows})
    if record["families"] != families:
        problems.append(
            f"families {record['families']} != recomputed {families}")
    if sorted(expected_families) != families:
        problems.append(
            f"builder grid fact violated: expected families "
            f"{sorted(expected_families)}, artifact has {families}")
    replaced = record["replaced_artifact"]
    if replaced is not None:
        if (not isinstance(replaced, dict) or "sha256" not in replaced
                or "preserved_copy" not in replaced):
            problems.append(
                "replaced_artifact must be null or "
                "{sha256, preserved_copy}")
        else:
            copy_path = _resolve_recorded_path(replaced["preserved_copy"], path)
            if copy_path is None:
                problems.append(
                    f"preserved copy {replaced['preserved_copy']!r} not found")
            else:
                copy_sha = hashlib.sha256(copy_path.read_bytes()).hexdigest()
                if copy_sha != replaced["sha256"]:
                    problems.append(
                        f"preserved copy {copy_path} hashes to {copy_sha}, "
                        f"sidecar records {replaced['sha256']} — the "
                        "historical record has been altered")
    if problems:
        raise SidecarValidationError(
            f"sidecar {sidecar_path} contradicts the artifact/preserved "
            "record (failing closed, not rewriting): " + "; ".join(problems)
        )
    return record


def write_jsonl_guarded(path, rows, reason, expected_families,
                        variant_fields, expected_variant_counts,
                        cross_disjoint_with=None):
    """Integrity-checked, preservation-guarded artifact write.

    Order of operations: validate rows -> (unchanged artifact: validate the
    existing sidecar and preserved copy, failing closed on contradiction) ->
    preserve any differing existing artifact -> write rows -> write
    ``*.provenance.json`` sidecar. Returns a status dict.
    """
    path = Path(path)
    assert_unique_pairs(rows)
    assert_variant_grid(rows, expected_families, variant_fields,
                        expected_variant_counts)
    if cross_disjoint_with is not None and Path(cross_disjoint_with).exists():
        other = [json.loads(line) for line in
                 Path(cross_disjoint_with).read_text().splitlines()
                 if line.strip()]
        assert_disjoint_families(rows, other, label_a=path.name,
                                 label_b=Path(cross_disjoint_with).name)
    payload = rows_to_jsonl(rows)
    status = {"path": str(path), "n_rows": len(rows),
              "n_unique_pairs": unique_pair_count(rows)}
    if path.exists():
        existing = path.read_text()
        if existing == payload:
            # QA Q8: byte-identity is not enough — the sidecar (and any
            # preserved copy it references) must validate before returning.
            validate_sidecar(path, rows, payload, expected_families)
            status["action"] = "unchanged"
            status["sidecar_validated"] = True
            return status
        preserved = preserve_existing_artifact(
            path, reason=f"pre-regeneration preservation: {reason}"
        )
        status["action"] = "replaced_with_preservation"
        status["preserved"] = preserved
    else:
        preserved = None
        status["action"] = "created"
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)
    _write_sidecar(path, rows, payload, reason, preserved)
    return status


def _write_sidecar(path, rows, payload, reason, preserved, extra=None):
    sidecar = path.with_suffix(path.suffix + ".provenance.json")
    families = sorted({row["scenario_id"] for row in rows})
    record = {
        "artifact": path.name,
        "sha256": sha256_text(payload),
        "n_rows": len(rows),
        "n_unique_pairs": unique_pair_count(rows),
        "families": families,
        "reason": reason,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "git_commit_at_write": _git_commit(path),
        "replaced_artifact": (
            {"sha256": preserved["sha256"],
             "preserved_copy": preserved["preserved_copy"]}
            if preserved else None
        ),
    }
    if extra:
        record.update(extra)
    sidecar.write_text(json.dumps(record, indent=2) + "\n")
    return record


def repair_sidecar(artifact_path, reason, replaced_sha=None,
                   historical_root=None, *, expected_families=None,
                   variant_fields=None, expected_variant_counts=None):
    """Explicit offline reconstruction of a MISSING sidecar (QA Q8).

    Only reconstructs from verified bytes: the artifact must round-trip
    through the deterministic serializer, and any preservation record it is
    linked to must verify (directory hash prefix, metadata hash, copy
    bytes). An existing sidecar is never touched. The reconstructed sidecar
    carries an explicit ``reconstructed`` audit entry.
    """
    artifact = Path(artifact_path)
    sidecar = artifact.with_suffix(artifact.suffix + ".provenance.json")
    if sidecar.exists():
        raise BuilderIntegrityError(
            f"{sidecar} already exists; repair-sidecar only reconstructs a "
            "MISSING sidecar and never rewrites an existing record")
    if not artifact.is_file():
        raise BuilderIntegrityError(f"artifact {artifact} not found")
    content = artifact.read_bytes()
    payload = content.decode("utf-8")
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    if rows_to_jsonl(rows) != payload:
        raise BuilderIntegrityError(
            f"{artifact} does not round-trip the deterministic-builder "
            "serialization; refusing to reconstruct provenance for a "
            "non-builder artifact")
    if not (expected_families and variant_fields and expected_variant_counts):
        raise BuilderIntegrityError(
            "sidecar repair requires builder-specific --expected-families, "
            "--variant-fields, and --expected-variant-counts")
    assert_unique_pairs(rows)
    assert_variant_grid(rows, expected_families, variant_fields,
                        expected_variant_counts)
    root = (Path(historical_root) if historical_root
            else artifact.parent / "historical")
    verified = []
    if root.is_dir():
        for dest_dir in sorted(root.glob(f"{artifact.name}.*")):
            copy = dest_dir / artifact.name
            if not copy.is_file():
                continue
            copy_sha = hashlib.sha256(copy.read_bytes()).hexdigest()
            if not dest_dir.name.endswith(copy_sha[:8]):
                raise BuilderIntegrityError(
                    f"preservation record {dest_dir} content hash "
                    f"{copy_sha[:8]} does not match its directory name — "
                    "the historical record has been altered")
            meta_file = dest_dir / "preservation.json"
            if meta_file.exists():
                meta = json.loads(meta_file.read_text())
                if meta.get("sha256") != copy_sha:
                    raise BuilderIntegrityError(
                        f"preservation metadata {meta_file} records sha256 "
                        f"{meta.get('sha256')} but the copy hashes to "
                        f"{copy_sha}")
            verified.append({"sha256": copy_sha,
                             "preserved_copy": _repo_relative(copy)})
    if not verified:
        raise BuilderIntegrityError(
            "sidecar repair requires at least one verified preserved-copy "
            "record; artifact bytes alone cannot reconstruct provenance")
    if replaced_sha is not None:
        matches = [v for v in verified if v["sha256"] == replaced_sha]
        if not matches:
            raise BuilderIntegrityError(
                f"no verified preservation record matches --replaced-sha "
                f"{replaced_sha}")
        preserved = matches[0]
    elif len(verified) == 1:
        preserved = verified[0]
    elif len(verified) > 1:
        raise BuilderIntegrityError(
            "multiple verified preservation records exist for "
            f"{artifact.name}: {[v['sha256'][:8] for v in verified]}; pass "
            "--replaced-sha to select which one this content replaced")
    else:  # unreachable: no-preservation case rejected above
        preserved = None
    record = _write_sidecar(
        artifact, rows, payload,
        reason=f"sidecar reconstruction: {reason}", preserved=preserved,
        extra={"reconstructed": {
            "at": datetime.now(timezone.utc).isoformat(),
            "by": "builder_integrity repair-sidecar",
            "reason": reason,
            "verified_from": {
                "artifact_sha256": sha256_text(payload),
                "verified_preservations": verified,
            },
        }},
    )
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Offline integrity utilities for deterministic builders.")
    sub = ap.add_subparsers(dest="command", required=True)
    rep = sub.add_parser(
        "repair-sidecar",
        help="reconstruct a MISSING sidecar from verified artifact and "
             "preserved-copy bytes, leaving an audit entry")
    rep.add_argument("--artifact", required=True)
    rep.add_argument("--reason", required=True,
                     help="why the sidecar is missing (recorded verbatim)")
    rep.add_argument("--replaced-sha", default=None,
                     help="full sha256 of the preservation record this "
                          "content replaced (required when several exist)")
    rep.add_argument("--historical-root", default=None)
    rep.add_argument("--expected-families", nargs="+", required=True)
    rep.add_argument("--variant-fields", nargs="+", required=True)
    rep.add_argument("--expected-variant-counts", nargs="+", type=int,
                     required=True)
    args = ap.parse_args(argv)
    record = repair_sidecar(args.artifact, args.reason,
                            replaced_sha=args.replaced_sha,
                            historical_root=args.historical_root,
                            expected_families=args.expected_families,
                            variant_fields=args.variant_fields,
                            expected_variant_counts=args.expected_variant_counts)
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
