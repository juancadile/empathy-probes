"""QA Q6: immutable revisions, fail-closed outputs, and source binding.

Pure validation tests — no network, no model load, no HF cache assumptions.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import src.utils.evidence_run as evidence_run  # noqa: E402

from src.utils.evidence_run import (  # noqa: E402
    ELIGIBILITY_ACCEPTED, ELIGIBILITY_EXPLORATORY, EvidenceRunError,
    atomic_write_json, evaluate_run_contract, is_immutable_revision,
    revalidate_run_contract, require_fresh_output, source_binding,
    validate_pinned_revision,
    verify_revision_resolution,
)

FULL = "a" * 40
OTHER = "b" * 40


# ---------------------------------------------------------------------------
# revision pinning
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("revision,ok", [
    (FULL, True),
    ("11c9b309abf73637e4b6f9a3fa1e92e615547819", True),
    (None, False),
    ("main", False),                # mutable alias
    ("refs/pr/1", False),           # mutable alias
    (FULL[:12], False),             # abbreviated hash
    (FULL.upper(), False),          # not the canonical lowercase form
])
def test_is_immutable_revision(revision, ok):
    assert is_immutable_revision(revision) is ok


def test_accepted_rejects_absent_and_mutable_revisions():
    with pytest.raises(EvidenceRunError, match="requires an explicit"):
        validate_pinned_revision("model", None)
    with pytest.raises(EvidenceRunError, match="immutable"):
        validate_pinned_revision("model", "main")
    with pytest.raises(EvidenceRunError, match="immutable"):
        validate_pinned_revision("model", FULL[:8])


def test_accepted_rejects_requested_resolved_mismatch():
    record = {"commit": OTHER, "method": "cache_refs"}
    with pytest.raises(EvidenceRunError, match="mismatch"):
        verify_revision_resolution("model", FULL, record)


def test_single_snapshot_fallback_never_satisfies_accepted():
    # even when the lone cached snapshot happens to equal the request, the
    # method is not an exact-revision verification (QA Q6)
    record = {"commit": FULL, "method": "single_snapshot"}
    with pytest.raises(EvidenceRunError, match="single_snapshot"):
        verify_revision_resolution("model", FULL, record)
    with pytest.raises(EvidenceRunError, match="cache_miss"):
        verify_revision_resolution("model", FULL,
                                   {"commit": None, "method": "cache_miss"})
    with pytest.raises(EvidenceRunError, match="unresolved"):
        verify_revision_resolution("model", FULL,
                                   {"commit": None, "method": "unresolved"})


def test_exact_resolution_passes():
    for method in ("cache_refs", "explicit_commit_snapshot"):
        verify_revision_resolution("model", FULL,
                                   {"commit": FULL, "method": method})


# ---------------------------------------------------------------------------
# contract evaluation (before any load)
# ---------------------------------------------------------------------------

def test_accepted_contract_fails_before_load_on_unpinned_revision(tmp_path):
    with pytest.raises(EvidenceRunError, match="immutable"):
        evaluate_run_contract(
            "accepted",
            revisions={"model": {"requested": "main", "resolution": None}},
            output_paths=(tmp_path / "out.json",),
            check_source=False,
        )


def test_accepted_contract_fails_on_resolution_mismatch(tmp_path):
    with pytest.raises(EvidenceRunError, match="mismatch"):
        evaluate_run_contract(
            "accepted",
            revisions={"model": {
                "requested": FULL,
                "resolution": {"commit": OTHER, "method": "cache_refs"}}},
            output_paths=(tmp_path / "out.json",),
            check_source=False,
        )


def test_accepted_contract_fails_closed_on_existing_output(tmp_path):
    existing = tmp_path / "out.json"
    existing.write_text("{}")
    with pytest.raises(EvidenceRunError, match="refuses to overwrite"):
        evaluate_run_contract(
            "accepted",
            revisions={"model": {
                "requested": FULL,
                "resolution": {"commit": FULL, "method": "cache_refs"}}},
            output_paths=(existing,),
            check_source=False,
        )


def test_accepted_contract_passes_and_is_marked_eligible(tmp_path):
    contract = evaluate_run_contract(
        "accepted",
        revisions={"model": {
            "requested": FULL,
            "resolution": {"commit": FULL, "method": "cache_refs"}},
            "mmlu_dataset": {"requested": OTHER}},
        output_paths=(tmp_path / "out.json",),
        check_source=False,
    )
    assert contract["evidence_eligibility"] == ELIGIBILITY_ACCEPTED
    assert contract["revisions"]["model"]["verified"] is True
    assert contract["revisions"]["mmlu_dataset"]["immutable"] is True


def test_exploratory_contract_persists_missing_pins():
    contract = evaluate_run_contract(
        "exploratory",
        revisions={"model": {"requested": None, "resolution": None},
                   "tokenizer": {"requested": FULL, "resolution": None}},
    )
    assert contract["evidence_eligibility"] == ELIGIBILITY_EXPLORATORY
    assert contract["missing_pins"] == ["model"]
    assert "ineligible for confirmatory evidence" in contract["warning"]


def test_unknown_run_mode_rejected():
    with pytest.raises(EvidenceRunError, match="unknown run mode"):
        evaluate_run_contract("confirmatory")


# ---------------------------------------------------------------------------
# atomic no-overwrite output
# ---------------------------------------------------------------------------

def test_require_fresh_output(tmp_path):
    fresh = tmp_path / "a.json"
    require_fresh_output(fresh)  # ok
    fresh.write_text("{}")
    with pytest.raises(EvidenceRunError, match="refuses to overwrite"):
        require_fresh_output(fresh)


def test_atomic_write_refuses_existing_and_validates(tmp_path):
    out = tmp_path / "artifact.json"
    atomic_write_json(out, {"x": 1})
    assert json.loads(out.read_text()) == {"x": 1}
    with pytest.raises(EvidenceRunError, match="refusing to overwrite"):
        atomic_write_json(out, {"x": 2})
    assert json.loads(out.read_text()) == {"x": 1}  # untouched

    def reject(parsed):
        raise ValueError("incomplete artifact")

    target = tmp_path / "validated.json"
    with pytest.raises(ValueError, match="incomplete artifact"):
        atomic_write_json(target, {"x": 3}, validate_fn=reject)
    assert not target.exists()  # nothing finalized
    assert not list(tmp_path.glob(".*.tmp.*"))  # temp cleaned up


def test_atomic_write_overwrite_mode_is_explicit(tmp_path):
    out = tmp_path / "explore.json"
    atomic_write_json(out, {"v": 1}, require_fresh=False)
    atomic_write_json(out, {"v": 2}, require_fresh=False)
    assert json.loads(out.read_text()) == {"v": 2}


def test_atomic_write_race_preserves_competing_sentinel(tmp_path, monkeypatch):
    out = tmp_path / "artifact.json"
    real_link = evidence_run.os.link

    def racing_link(src, dst):
        Path(dst).write_text('{"sentinel": true}')
        return real_link(src, dst)

    monkeypatch.setattr(evidence_run.os, "link", racing_link)
    with pytest.raises(EvidenceRunError, match="appeared during finalization"):
        atomic_write_json(out, {"accepted": True})
    assert json.loads(out.read_text()) == {"sentinel": True}


# ---------------------------------------------------------------------------
# source binding (temp git repos; no assumptions about this checkout)
# ---------------------------------------------------------------------------

def make_repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"],
                   check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "t"],
                   check=True)
    (tmp_path / "code.py").write_text("x = 1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "init"],
                   check=True)
    return tmp_path


def test_source_binding_clean_repo_ok(tmp_path):
    repo = make_repo(tmp_path)
    report = source_binding(repo)
    assert report["ok"] is True
    assert report["commit"]


def test_source_binding_fails_on_unexplained_dirty_or_untracked(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "code.py").write_text("x = 2\n")  # modified tracked file
    report = source_binding(repo)
    assert report["ok"] is False
    assert report["violations"][0]["path"] == "code.py"
    (repo / "code.py").write_text("x = 1\n")
    (repo / "stray.txt").write_text("data")  # untracked
    report = source_binding(repo)
    assert report["ok"] is False


def test_source_binding_requires_explicit_persisted_rules(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "results").mkdir()
    (repo / "results" / "out.json").write_text("{}")
    report = source_binding(repo, allowed_dirty_rules=["results/"])
    assert report["ok"] is True
    assert report["rules"] == ["results/"]
    assert report["paths"][0]["excluded_by_rule"] == "results/"
    # a bare count would hide this; without the rule it must fail
    with pytest.raises(EvidenceRunError, match="unexplained"):
        evaluate_run_contract("accepted",
                              revisions={}, output_paths=(),
                              repo_root=repo, source_rules=())


@pytest.mark.parametrize("rule", [
    "src/", "s*/", "data/", "third_party/", "scripts/", "configs/",
    "pyproject.toml", "requirements*", "environment*", "Dockerfile",
])
def test_source_binding_cannot_exclude_scientific_roots(tmp_path, rule):
    repo = make_repo(tmp_path)
    report = source_binding(repo, allowed_dirty_rules=[rule])
    assert report["ok"] is False
    assert "scientific code/data" in report["error"] or "generated-output" in report["error"]


def test_revalidation_rejects_input_drift(tmp_path):
    repo = make_repo(tmp_path)
    input_path = tmp_path / "results" / "input.npy"
    input_path.parent.mkdir()
    input_path.write_bytes(b"before")
    subprocess.run(["git", "-C", str(repo), "add", "results/input.npy"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "input"], check=True)
    contract = evaluate_run_contract(
        "accepted", revisions={}, output_paths=(), repo_root=repo,
        source_rules=("results/input.npy",), input_paths=(input_path,))
    input_path.write_bytes(b"after")
    with pytest.raises(EvidenceRunError, match="input bytes changed"):
        revalidate_run_contract(
            contract, repo_root=repo, source_rules=("results/input.npy",),
            input_paths=(input_path,))
