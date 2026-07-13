"""QA Q4: component registries are bound to their direction.

Acceptance criteria covered here:
  * current set + current direction passes (and the live current key resolves
    the tracked file's recorded hash when it exists);
  * current set + historical direction fails;
  * literal component specs require explicit direction provenance;
  * the exploratory override is visible in the returned/persisted resolution
    and refused in accepted mode;
  * a superseded registry key is rejected in accepted mode (warning alone is
    insufficient).

Temporary fixtures are used for hash-mismatch cases so the tests do not
assume project result files; the real-file assertions are guarded.
"""

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from component_sets import (  # noqa: E402
    COMPONENT_SET_VERSIONS, ComponentSetError, bind_direction,
    normalize_direction_path, resolve_component_sets,
)

CURRENT_KEY = "gemma2_9b_it_resid_2026-07-12"
SUPERSEDED_KEY = "gemma2_9b_it_precorrection_2026-07-11"
CURRENT_ENTRY = COMPONENT_SET_VERSIONS[CURRENT_KEY]
CURRENT_DIR = CURRENT_ENTRY["direction"]
HISTORICAL_DIR = COMPONENT_SET_VERSIONS[SUPERSEDED_KEY]["direction"]
REAL_FILE = ROOT / CURRENT_DIR


def make_fixture_repo(tmp_path, rel_path, content=b"direction-bytes"):
    """Temp repo root containing rel_path with known content; returns
    (repo_root, sha256)."""
    target = tmp_path / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return tmp_path, hashlib.sha256(content).hexdigest()


def test_registry_entries_record_direction_hashes():
    for key, entry in COMPONENT_SET_VERSIONS.items():
        assert entry.get("direction_sha256"), key
        assert len(entry["direction_sha256"]) == 64, key


def test_current_set_plus_current_direction_passes():
    resolved = resolve_component_sets(
        ("positive_writers", "suppressors"), explicit={},
        set_key=CURRENT_KEY, model="google/gemma-2-9b-it",
        direction_path=CURRENT_DIR,
    )
    binding = resolved["source"]["direction_binding"]
    assert binding["path_match"] is True
    assert binding["mismatch_override"] is False
    assert resolved["sets"]["suppressors"] == "L18H13,L20H10,L19H12,L17H7"


@pytest.mark.skipif(not REAL_FILE.exists(),
                    reason="tracked direction file not present")
def test_live_current_key_resolves_recorded_hash():
    resolved = resolve_component_sets(
        ("positive_writers",), explicit={},
        set_key=CURRENT_KEY, model="google/gemma-2-9b-it",
        direction_path=CURRENT_DIR,
    )
    binding = resolved["source"]["direction_binding"]
    assert binding["observed_sha256"] == (
        "1b6d692e0e933d76c15f722fe996d01e1ca2ee8c8b72f19daf2f122cda294b42")
    assert binding["hash_match"] is True


def test_current_set_plus_historical_direction_fails():
    with pytest.raises(ComponentSetError, match="direction path mismatch"):
        resolve_component_sets(
            ("positive_writers",), explicit={},
            set_key=CURRENT_KEY, model="google/gemma-2-9b-it",
            direction_path=HISTORICAL_DIR,
        )


def test_hash_mismatch_fails_even_when_path_matches(tmp_path):
    repo, _sha = make_fixture_repo(tmp_path, CURRENT_DIR, b"WRONG BYTES")
    with pytest.raises(ComponentSetError, match="direction hash mismatch"):
        resolve_component_sets(
            ("positive_writers",), explicit={},
            set_key=CURRENT_KEY, model="google/gemma-2-9b-it",
            direction_path=CURRENT_DIR, repo_root=repo,
        )


def test_literal_specs_require_direction_provenance():
    with pytest.raises(ComponentSetError, match="direction provenance"):
        resolve_component_sets(
            ("suppressors",), explicit={"suppressors": "L18H13,L20H10"})


def test_exploratory_override_is_visible_and_non_confirmatory():
    resolved = resolve_component_sets(
        ("positive_writers",), explicit={},
        set_key=CURRENT_KEY, model="google/gemma-2-9b-it",
        direction_path=HISTORICAL_DIR, run_mode="exploratory",
        allow_direction_mismatch=True,
    )
    source = resolved["source"]
    assert source["direction_binding"]["mismatch_override"] is True
    assert source["direction_binding"]["mismatch_details"]
    assert source["confirmatory_evidence_eligible"] is False
    assert any("direction mismatch override" in r
               for r in source["ineligibility_reasons"])
    assert "DIRECTION MISMATCH OVERRIDE" in source["warning"]


def test_accepted_mode_refuses_the_override():
    with pytest.raises(ComponentSetError, match="exploratory-only override"):
        resolve_component_sets(
            ("positive_writers",), explicit={},
            set_key=CURRENT_KEY, model="google/gemma-2-9b-it",
            direction_path=HISTORICAL_DIR, run_mode="accepted",
            allow_direction_mismatch=True,
        )


def test_accepted_mode_rejects_superseded_key():
    with pytest.raises(ComponentSetError, match="rejected in accepted mode"):
        resolve_component_sets(
            ("positive_writers",), explicit={},
            set_key=SUPERSEDED_KEY, model="google/gemma-2-9b-it",
            direction_path=HISTORICAL_DIR, run_mode="accepted",
        )


def test_superseded_key_still_resolves_in_exploratory():
    resolved = resolve_component_sets(
        ("positive_writers",), explicit={},
        set_key=SUPERSEDED_KEY, model="google/gemma-2-9b-it",
        direction_path=HISTORICAL_DIR, run_mode="exploratory",
    )
    assert resolved["sets"]["positive_writers"] == "L19MLP,L20H15"
    assert resolved["source"]["confirmatory_evidence_eligible"] is False
    assert "superseded" in resolved["source"]["warning"]


def test_accepted_mode_requires_direction_file_to_exist(tmp_path):
    with pytest.raises(ComponentSetError, match="direction file to exist"):
        resolve_component_sets(
            ("positive_writers",), explicit={},
            set_key=CURRENT_KEY, model="google/gemma-2-9b-it",
            direction_path=CURRENT_DIR, run_mode="accepted",
            repo_root=tmp_path,  # empty repo: file missing
        )


def test_accepted_mode_passes_with_correct_fixture_bytes(tmp_path):
    repo, sha = make_fixture_repo(tmp_path, CURRENT_DIR)
    entry = dict(CURRENT_ENTRY, direction_sha256=sha)
    binding = bind_direction(CURRENT_DIR, entry=entry, set_key=CURRENT_KEY,
                             run_mode="accepted", repo_root=repo)
    assert binding["path_match"] is True and binding["hash_match"] is True


def test_persisted_binding_records_expected_and_observed(tmp_path):
    repo, sha = make_fixture_repo(tmp_path, CURRENT_DIR)
    resolved = resolve_component_sets(
        ("positive_writers",), explicit={},
        set_key=CURRENT_KEY, model="google/gemma-2-9b-it",
        direction_path=CURRENT_DIR, repo_root=repo,
        run_mode="exploratory", allow_direction_mismatch=True,
    )
    binding = resolved["source"]["direction_binding"]
    assert binding["registry_path"] == CURRENT_DIR
    assert binding["registry_sha256"] == CURRENT_ENTRY["direction_sha256"]
    assert binding["observed_sha256"] == sha
    assert binding["normalized_path"] == CURRENT_DIR


def test_normalize_direction_path_variants(tmp_path):
    assert normalize_direction_path("results/x/y.npy", tmp_path) == "results/x/y.npy"
    assert normalize_direction_path(
        str(tmp_path / "results/x/y.npy"), tmp_path) == "results/x/y.npy"
    outside = normalize_direction_path("/somewhere/else/y.npy", tmp_path)
    assert outside.startswith("/")
