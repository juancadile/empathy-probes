"""Shared accepted/exploratory evidence-run contract (Integrity Repair A QA Q6).

Every evidence-producing script declares an explicit run class:

  * ``accepted`` — the run may produce confirmatory evidence. Before any
    data/model load it must (a) pin immutable model/tokenizer/dataset
    revisions and verify the locally resolved commits against the requested
    values, (b) refuse pre-existing output artifacts, and (c) bind the
    executed scientific source to the recorded git commit (no unexplained
    dirty/untracked paths). Final artifacts are written through a temporary
    file and atomically renamed only after the serialized artifact
    re-validates; an existing artifact is never silently overwritten.
  * ``exploratory`` — omissions are allowed, but the missing pins and the
    exploratory status are persisted and the artifact is visibly marked
    ineligible for confirmatory evidence.

Unmigrated historical CLIs do not use this module; they carry the permanent
``historical_or_exploratory_only`` classification instead (see
``src/weight_edit_inventory.py``).
"""

import fnmatch
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

RUN_MODES = ("accepted", "exploratory")
RUN_CONTRACT_SCHEMA = "empathy-action-probes/run-contract/1"

ELIGIBILITY_ACCEPTED = "accepted_confirmatory"
ELIGIBILITY_EXPLORATORY = "exploratory_only_not_confirmatory"
ELIGIBILITY_HISTORICAL = "historical_or_exploratory_only"

#: resolve_hf_commit() methods that can satisfy an accepted exact-revision
#: check. ``single_snapshot`` is deliberately absent: a lone cached snapshot
#: says nothing about WHICH revision the run requested (QA Q6).
_ACCEPTED_RESOLUTION_METHODS = ("cache_refs", "explicit_commit_snapshot")

_FULL_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class EvidenceRunError(RuntimeError):
    """An accepted-mode requirement was violated; the run must not proceed."""


def is_immutable_revision(revision):
    """True only for a full 40-hex commit hash (mutable aliases and
    abbreviated hashes are not immutable pins)."""
    return bool(isinstance(revision, str)
                and _FULL_COMMIT_RE.match(revision.lower())
                and revision == revision.lower())


def validate_pinned_revision(label, requested):
    """Accepted mode: ``requested`` must be a full immutable commit hash."""
    if requested is None:
        raise EvidenceRunError(
            f"accepted mode requires an explicit immutable revision for "
            f"{label!r}; none was provided"
        )
    if not is_immutable_revision(requested):
        raise EvidenceRunError(
            f"accepted mode requires an immutable full 40-hex commit for "
            f"{label!r}; got {requested!r} (mutable aliases and abbreviated "
            "hashes are rejected)"
        )


def verify_revision_resolution(label, requested, record):
    """Accepted mode: the locally resolved commit must equal the request.

    ``record`` is a ``resolve_hf_commit()`` result. A ``single_snapshot``
    fallback never satisfies this check even if the lone cached snapshot
    happens to equal the request context (an unknown requested revision must
    not be laundered through cache contents).
    """
    if record is None:
        raise EvidenceRunError(
            f"accepted mode requires a local resolution record for {label!r} "
            "to verify the pinned revision; none was provided"
        )
    method = record.get("method")
    commit = record.get("commit")
    if method not in _ACCEPTED_RESOLUTION_METHODS:
        raise EvidenceRunError(
            f"accepted mode cannot verify {label!r}: resolution method "
            f"{method!r} is not an exact-revision match (commit={commit!r}, "
            f"requested={requested!r})"
        )
    if commit != requested:
        raise EvidenceRunError(
            f"requested/resolved commit mismatch for {label!r}: requested "
            f"{requested!r} but the local cache resolved {commit!r}"
        )


def require_fresh_output(*paths):
    """Accepted mode fails closed if any result artifact already exists."""
    existing = [str(p) for p in paths if Path(p).exists()]
    if existing:
        raise EvidenceRunError(
            "accepted mode refuses to overwrite existing output artifacts: "
            + ", ".join(existing)
            + " (choose a fresh output path; prior accepted runs are never "
            "silently replaced)"
        )


def atomic_write_json(path, obj, *, require_fresh=True, indent=2,
                      validate_fn=None):
    """Write JSON through a temp file and atomically rename after validation.

    ``validate_fn(parsed)`` may raise to abort finalization; the temp file is
    removed and the destination untouched. With ``require_fresh`` (accepted
    semantics) an existing destination aborts before and after serialization.
    """
    path = Path(path)
    if require_fresh and path.exists():
        raise EvidenceRunError(
            f"refusing to overwrite existing artifact {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, indent=indent)
    tmp = path.parent / f".{path.name}.tmp.{os.getpid()}"
    tmp.write_text(payload)
    try:
        parsed = json.loads(tmp.read_text())  # complete-artifact validation
        if validate_fn is not None:
            validate_fn(parsed)
        if require_fresh:
            # link(2) creates the destination only if it does not already
            # exist. Unlike exists()+replace(), this is atomic no-clobber even
            # when another process publishes a sentinel during validation.
            try:
                os.link(tmp, path)
            except FileExistsError as exc:
                raise EvidenceRunError(
                    f"artifact {path} appeared during finalization; refusing "
                    "to overwrite") from exc
            tmp.unlink()
        else:
            os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def require_keys_validator(*required_keys):
    """Build a minimal schema validator for atomic artifact finalization."""
    required = set(required_keys)

    def validate(payload):
        if not isinstance(payload, dict):
            raise ValueError("artifact root must be a JSON object")
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"incomplete artifact: missing keys {missing}")
    return validate


# ---------------------------------------------------------------------------
# Source binding (QA Q6): accepted runs must execute recorded code/data
# ---------------------------------------------------------------------------

def _git(root, *argv):
    return subprocess.run(
        ("git", "-C", str(root)) + argv,
        capture_output=True, text=True, timeout=60, check=True,
    ).stdout


def _match_rule(path, rules):
    for rule in rules:
        if rule.endswith("/"):
            if path.startswith(rule) or fnmatch.fnmatch(path, rule + "*"):
                return rule
        if fnmatch.fnmatch(path, rule) or path == rule:
            return rule
    return None


_SCIENTIFIC_ROOTS = (
    "src/", "tests/", "data/", "third_party/", "scripts/", "configs/",
    "pyproject.toml", "requirements", "environment", "Dockerfile",
)
_GENERATED_ROOTS = ("results/", "logs/", "artifacts/")


def validate_source_rules(rules):
    """Accepted exclusions may cover generated outputs, never science code.

    Restricting rules by their literal root also closes wildcard spellings
    such as ``s*/`` that would otherwise match ``src/``.
    """
    for rule in rules:
        normalized = str(rule).lstrip("./")
        if not any(normalized.startswith(root) for root in _GENERATED_ROOTS):
            raise EvidenceRunError(
                f"source exclusion {rule!r} is not under a generated-output "
                f"root {_GENERATED_ROOTS}; scientific code/data cannot be "
                "excluded from accepted source binding")
        sentinels = [
            "src/science.py", "tests/test_science.py", "data/input.jsonl",
            "third_party/game/runtime.py", "scripts/run.sh", "configs/run.json",
            "pyproject.toml", "requirements.txt", "environment.yml", "Dockerfile",
        ]
        if any(_match_rule(path, [rule]) for path in sentinels):
            raise EvidenceRunError(
                f"source exclusion {rule!r} can match scientific code/data")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bind_inputs(paths):
    """Content-address declared inputs for start/end drift detection."""
    out = {}
    for value in paths:
        path = Path(value).resolve()
        if not path.is_file():
            raise EvidenceRunError(f"declared input is not a file: {path}")
        out[str(path)] = sha256_file(path)
    return out


def source_binding(repo_root=None, allowed_dirty_rules=()):
    """Bind the executed source tree to the recorded commit.

    Every dirty or untracked path must be excluded by an EXPLICIT, persisted
    rule (e.g. the run's own output directory); a bare dirty-path count is
    insufficient. Returns the persisted report.
    """
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    rules = list(allowed_dirty_rules)
    try:
        validate_source_rules(rules)
    except EvidenceRunError as exc:
        return {"ok": False, "error": str(exc), "rules": rules,
                "paths": [], "violations": []}
    try:
        commit = _git(root, "rev-parse", "HEAD").strip()
        branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD").strip()
        status = _git(root, "status", "--porcelain")
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "rules": rules}
    paths, violations = [], []
    for line in status.splitlines():
        if not line.strip():
            continue
        state, path = line[:2], line[3:].strip()
        if " -> " in path:  # rename records old -> new
            path = path.split(" -> ", 1)[1]
        if path.startswith('"') and path.endswith('"'):
            path = path[1:-1]
        rule = _match_rule(path, rules)
        entry = {"path": path, "status": state.strip() or state,
                 "excluded_by_rule": rule}
        paths.append(entry)
        if rule is None:
            violations.append(entry)
    return {
        "ok": not violations,
        "commit": commit,
        "branch": branch,
        "rules": rules,
        "paths": paths,
        "violations": violations,
    }


def require_source_binding(repo_root=None, allowed_dirty_rules=()):
    report = source_binding(repo_root, allowed_dirty_rules)
    if not report["ok"]:
        detail = report.get("error") or ", ".join(
            f"{v['path']} ({v['status']})" for v in report["violations"])
        raise EvidenceRunError(
            "accepted mode requires the executed source to be bound to the "
            f"recorded commit; unexplained modified/untracked paths: {detail}. "
            "Commit the changes or pass an explicit persisted exclusion rule "
            "for generated outputs."
        )
    return report


# ---------------------------------------------------------------------------
# Contract evaluation (one call per script, before any data/model load)
# ---------------------------------------------------------------------------

def evaluate_run_contract(run_mode, *, revisions=None, output_paths=(),
                          repo_root=None, source_rules=(),
                          check_source=None, input_paths=()):
    """Validate the run class BEFORE loading anything; return the persisted
    contract block.

    ``revisions``: mapping label -> {"requested": str|None,
    "resolution": resolve_hf_commit-record|None}. Labels without a local
    resolution record (e.g. HF datasets) are pin-format-validated only.
    ``check_source`` defaults to True in accepted mode, False otherwise.
    """
    if run_mode not in RUN_MODES:
        raise EvidenceRunError(
            f"unknown run mode {run_mode!r}; expected one of {RUN_MODES}")
    revisions = dict(revisions or {})
    contract = {
        "schema": RUN_CONTRACT_SCHEMA,
        "run_mode": run_mode,
        "evidence_eligibility": (
            ELIGIBILITY_ACCEPTED if run_mode == "accepted"
            else ELIGIBILITY_EXPLORATORY),
        "revisions": {},
        "output_paths": [str(p) for p in output_paths],
        "declared_inputs": bind_inputs(input_paths) if input_paths else {},
    }
    missing = []
    for label, spec in sorted(revisions.items()):
        requested = spec.get("requested")
        record = {"requested": requested,
                  "immutable": is_immutable_revision(requested)}
        resolution = spec.get("resolution")
        if resolution is not None:
            record["resolution"] = resolution
        if run_mode == "accepted":
            validate_pinned_revision(label, requested)
            if resolution is not None:
                verify_revision_resolution(label, requested, resolution)
                record["verified"] = True
        elif not record["immutable"]:
            missing.append(label)
        contract["revisions"][label] = record
    if run_mode == "accepted":
        require_fresh_output(*output_paths)
        if check_source is None or check_source:
            contract["source_binding"] = require_source_binding(
                repo_root, source_rules)
    else:
        contract["missing_pins"] = missing
        contract["warning"] = (
            "EXPLORATORY RUN: this artifact is ineligible for confirmatory "
            "evidence"
            + (f"; unpinned revisions: {missing}" if missing else "")
        )
        if check_source:
            contract["source_binding"] = source_binding(
                repo_root, source_rules)
    return contract


def revalidate_run_contract(contract, *, repo_root=None, source_rules=(),
                            input_paths=()):
    """Re-bind source and inputs immediately before accepted finalization."""
    if contract.get("run_mode") != "accepted":
        return contract
    end_source = require_source_binding(repo_root, source_rules)
    start_source = contract.get("source_binding")
    if start_source and end_source.get("commit") != start_source.get("commit"):
        raise EvidenceRunError(
            "source commit changed between accepted run start and finalization")
    end_inputs = bind_inputs(input_paths)
    if end_inputs != contract.get("declared_inputs", {}):
        raise EvidenceRunError(
            "declared input bytes changed between accepted run start and "
            "finalization")
    contract["final_source_binding"] = end_source
    contract["final_declared_inputs"] = end_inputs
    return contract
