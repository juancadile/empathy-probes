"""Environment-lock / run-provenance pure logic (Integrity Repair A).

Local, offline: injected package/platform facts, tmp-dir HF cache layouts,
and serialize/parse round-trips. The real Spark `empathy` capture is a
Gate-0B action and is deliberately NOT exercised here.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from utils.run_provenance import (  # noqa: E402
    ENV_LOCK_SCHEMA, build_environment_lock, collect_run_provenance,
    main as provenance_main, package_versions, parse_cached_refs, parse_lock,
    resolve_hf_commit, resolve_model_and_tokenizer, serialize_lock,
)


FAKE_FACTS = dict(
    packages={"torch": "2.13.0+cu130", "transformers": "5.13.0",
              "sae-lens": None},
    python_info={"python": "3.12.13", "platform": "linux-test"},
    cuda_info={"torch_available": True, "cuda_available": False},
    git_info={"commit": "abc123", "dirty": False},
)


def test_lock_build_serialize_parse_roundtrip():
    lock = build_environment_lock(
        **FAKE_FACTS,
        models=[{"model_id": "google/gemma-2-9b-it", "commit": "11c9b309"}],
        label="unit-test", created_at="2026-07-13T00:00:00+00:00",
    )
    text = serialize_lock(lock)
    assert text.endswith("\n")
    parsed = parse_lock(text)
    assert parsed == lock
    assert parsed["schema"] == ENV_LOCK_SCHEMA
    # absent packages are recorded, not dropped
    assert parsed["packages"]["sae-lens"] is None


def test_lock_serialization_is_deterministic():
    lock_a = build_environment_lock(**FAKE_FACTS, label="x",
                                    created_at="t0")
    lock_b = build_environment_lock(**FAKE_FACTS, label="x",
                                    created_at="t0")
    assert serialize_lock(lock_a) == serialize_lock(lock_b)


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d.update(schema="wrong/0"), "unsupported lock schema"),
    (lambda d: d.pop("packages"), "missing sections"),
    (lambda d: d.update(packages=[1, 2]), "wrong types"),
])
def test_parse_lock_rejects_invalid(mutate, match):
    lock = build_environment_lock(**FAKE_FACTS, created_at="t0")
    mutate(lock)
    with pytest.raises(ValueError, match=match):
        parse_lock(json.dumps(lock))


def test_parse_lock_rejects_non_json():
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_lock("{nope")


def test_package_versions_records_absence():
    versions = package_versions(("numpy", "definitely-not-a-real-pkg-xyz"))
    assert versions["numpy"] is not None
    assert versions["definitely-not-a-real-pkg-xyz"] is None


def make_hub_cache(tmp_path, model_id="google/gemma-2-9b-it",
                   commit="11c9b309abf73637e4b6f9a3fa1e92e615547819"):
    repo = tmp_path / ("models--" + model_id.replace("/", "--"))
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text(commit + "\n")
    (repo / "snapshots" / commit).mkdir(parents=True)
    return repo, commit


def test_parse_cached_refs_and_resolve(tmp_path):
    repo, commit = make_hub_cache(tmp_path)
    parsed = parse_cached_refs(repo)
    assert parsed["refs"] == {"main": commit}
    assert parsed["snapshots"] == [commit]

    record = resolve_hf_commit("google/gemma-2-9b-it", cache_dir=tmp_path)
    assert record["commit"] == commit
    assert record["method"] == "cache_refs"


def test_resolve_hf_commit_cache_miss(tmp_path):
    record = resolve_hf_commit("nope/never-downloaded", cache_dir=tmp_path)
    assert record["commit"] is None
    assert record["method"] == "cache_miss"


def test_resolve_model_and_tokenizer_separately(tmp_path):
    _, gemma_commit = make_hub_cache(tmp_path)
    _, tok_commit = make_hub_cache(tmp_path, model_id="other/tokenizer-repo",
                                   commit="feedbead" * 5)
    record = resolve_model_and_tokenizer(
        "google/gemma-2-9b-it", tokenizer_id="other/tokenizer-repo",
        cache_dir=tmp_path)
    assert record["model"]["commit"] == gemma_commit
    assert record["tokenizer"]["commit"] == tok_commit


def test_cli_export_and_validate(tmp_path, capsys):
    out = tmp_path / "lock.json"
    assert provenance_main(["export", "--out", str(out),
                            "--label", "unit-test",
                            "--omit-timestamp"]) == 0
    lock = parse_lock(out.read_text())
    assert lock["label"] == "unit-test"
    assert lock["created_at"] is None
    assert provenance_main(["validate", "--lock", str(out)]) == 0
    assert "valid lock" in capsys.readouterr().out


def test_collect_run_provenance_hashes_files(tmp_path):
    f = tmp_path / "direction.npy"
    f.write_bytes(b"\x01\x02\x03")
    prov = collect_run_provenance(files={"direction": f})
    assert prov["files"]["direction"]["sha256"] == (
        "039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81")
    missing = collect_run_provenance(files={"gone": tmp_path / "missing"})
    assert "error" in missing["files"]["gone"]
