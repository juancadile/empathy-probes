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
"""

import hashlib
import itertools
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


class BuilderIntegrityError(AssertionError):
    """A deterministic builder produced structurally invalid stimuli."""


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


def write_jsonl_guarded(path, rows, reason, expected_families,
                        variant_fields, expected_variant_counts,
                        cross_disjoint_with=None):
    """Integrity-checked, preservation-guarded artifact write.

    Order of operations: validate rows -> preserve any differing existing
    artifact -> write rows -> write ``*.provenance.json`` sidecar. Returns a
    status dict describing what happened.
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
            status["action"] = "unchanged"
            # keep the existing sidecar: it may hold the replaced_artifact
            # pointer from the write that produced this content
            if not path.with_suffix(path.suffix + ".provenance.json").exists():
                _write_sidecar(path, rows, payload, reason, preserved=None)
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


def _write_sidecar(path, rows, payload, reason, preserved):
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
    sidecar.write_text(json.dumps(record, indent=2) + "\n")
