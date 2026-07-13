"""Shared run-provenance and environment-lock helpers (Integrity Repair A).

Every accepted run must record exactly which code, environment, and model
weights produced it (`EXECUTION_ORDER_2026-07-13.md` rule 2). This module
centralizes that so scripts stop re-implementing partial versions:

  * ``collect_run_provenance()`` — compact dict for embedding in result JSONs:
    Python/platform, CUDA/device, git commit + dirty state, exact versions of
    the tracked scientific packages, input-file hashes, and locally resolved
    Hugging Face model/tokenizer commit hashes.
  * ``build_environment_lock()`` / ``serialize_lock()`` / ``parse_lock()`` —
    pure, deterministic environment-lock schema (testable offline).
  * CLI: ``python -m src.utils.run_provenance export --out <file>`` writes the
    lock for the CURRENT interpreter/host. The accepted Spark ``empathy``
    lock is captured in Gate 0B on the Spark itself; running the exporter
    elsewhere documents that other environment, never the Spark one.

Optional packages that are absent are RECORDED as absent (version ``null``);
their absence must never make this module fail to import or collect.

Note: ``requirements.txt`` holds broad developer install constraints; it is
not an environment lock and never substitutes for one.

Audit reference values expected at the Gate-0B Spark capture (verify against
the exported lock; do not hard-code as facts): Gemma cache revision
11c9b309abf73637e4b6f9a3fa1e92e615547819, Llama cache revision
0e9e39f249a16976918f6564b8830bc894c89659; Python 3.12.13, Torch
2.13.0+cu130, Transformers 5.13.0, TransformerLens 3.5.1, SAELens 6.45.3,
Datasets 5.0.0.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ENV_LOCK_SCHEMA = "empathy-action-probes/env-lock/1"
RUN_PROVENANCE_SCHEMA = "empathy-action-probes/run-provenance/1"

#: Exact-version packages named by the science audit, plus the hub stack that
#: determines which weights actually load. Absent -> version None.
TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "numpy",
    "scikit-learn",
    "datasets",
    "transformer-lens",
    "sae-lens",
    "accelerate",
    "huggingface-hub",
    "tokenizers",
    "safetensors",
)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions(packages=TRACKED_PACKAGES):
    """Exact installed versions; absent packages recorded as None."""
    from importlib import metadata

    versions = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
        except Exception as exc:  # metadata corruption must not kill a run
            versions[name] = f"error:{type(exc).__name__}"
    return versions


def python_platform_info():
    return {
        "python": sys.version.split()[0],
        "python_full": sys.version.replace("\n", " "),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def cuda_device_info():
    """CUDA/device facts via torch when present; absence is recorded."""
    try:
        import torch
    except Exception as exc:
        return {"torch_available": False, "detail": f"{type(exc).__name__}"}
    info = {
        "torch_available": True,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
    }
    try:
        if info["cuda_available"]:
            info["device_count"] = torch.cuda.device_count()
            info["devices"] = [
                {
                    "name": torch.cuda.get_device_name(i),
                    "capability": ".".join(
                        str(x) for x in torch.cuda.get_device_capability(i)
                    ),
                    "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory,
                }
                for i in range(torch.cuda.device_count())
            ]
    except Exception as exc:
        info["device_query_error"] = f"{type(exc).__name__}: {exc}"
    return info


def git_state(repo_root=None):
    """Current commit, branch, and dirty state; failures are recorded."""
    root = str(repo_root) if repo_root else str(Path(__file__).resolve().parents[2])

    def _git(*argv):
        return subprocess.run(
            ("git", "-C", root) + argv,
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()

    try:
        status = _git("status", "--porcelain")
        return {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status),
            "n_dirty_paths": len(status.splitlines()) if status else 0,
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Hugging Face cached-revision resolution (local only; no network)
# ---------------------------------------------------------------------------

def hf_cache_repo_dir(model_id, cache_dir=None):
    if cache_dir is None:
        try:
            from huggingface_hub.constants import HF_HUB_CACHE
            cache_dir = HF_HUB_CACHE
        except Exception:
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    return Path(cache_dir) / ("models--" + model_id.replace("/", "--"))


def parse_cached_refs(repo_dir):
    """Pure parse of a hub-cache repo dir -> {refs, snapshots} (no network).

    Layout: ``<repo>/refs/<ref>`` holds a commit hash; ``<repo>/snapshots/``
    holds one directory per cached commit.
    """
    repo_dir = Path(repo_dir)
    refs, snapshots = {}, []
    refs_dir = repo_dir / "refs"
    if refs_dir.is_dir():
        for ref_file in sorted(p for p in refs_dir.rglob("*") if p.is_file()):
            refs[str(ref_file.relative_to(refs_dir))] = (
                ref_file.read_text().strip()
            )
    snap_dir = repo_dir / "snapshots"
    if snap_dir.is_dir():
        snapshots = sorted(p.name for p in snap_dir.iterdir() if p.is_dir())
    return {"refs": refs, "snapshots": snapshots}


def resolve_hf_commit(model_id, cache_dir=None, revision="main"):
    """Resolve a model id to the locally cached commit hash (never network).

    Accepted reruns must then pass ``revision=<commit>`` explicitly to
    ``from_pretrained`` so the mutable id can never drift mid-project.
    """
    record = {
        "model_id": model_id,
        "requested_revision": revision,
        "commit": None,
        "method": None,
    }
    repo_dir = hf_cache_repo_dir(model_id, cache_dir)
    if not repo_dir.is_dir():
        record["method"] = "cache_miss"
        return record
    parsed = parse_cached_refs(repo_dir)
    record["cached_refs"] = parsed["refs"]
    record["cached_snapshots"] = parsed["snapshots"]
    if revision in parsed["refs"]:
        record["commit"] = parsed["refs"][revision]
        record["method"] = "cache_refs"
    elif revision in parsed["snapshots"]:
        record["commit"] = revision
        record["method"] = "explicit_commit_snapshot"
    elif len(parsed["snapshots"]) == 1 and not parsed["refs"]:
        record["commit"] = parsed["snapshots"][0]
        record["method"] = "single_snapshot"
    else:
        record["method"] = "unresolved"
    return record


def resolve_model_and_tokenizer(model_id, tokenizer_id=None, cache_dir=None,
                                revision="main", tokenizer_revision=None):
    """Model and tokenizer commit hashes resolved SEPARATELY (audit A2)."""
    return {
        "model": resolve_hf_commit(model_id, cache_dir, revision),
        "tokenizer": resolve_hf_commit(
            tokenizer_id or model_id, cache_dir,
            tokenizer_revision or revision,
        ),
    }


# ---------------------------------------------------------------------------
# Environment lock (pure build/serialize/parse; deterministic)
# ---------------------------------------------------------------------------

REQUIRED_LOCK_SECTIONS = ("schema", "label", "python", "packages", "cuda", "git", "models")


def build_environment_lock(*, packages, python_info, cuda_info, git_info,
                           models=(), label=None, created_at=None):
    """Assemble a lock dict from already-collected facts (pure, testable)."""
    lock = {
        "schema": ENV_LOCK_SCHEMA,
        "label": label,
        "created_at": created_at,
        "python": dict(python_info),
        "packages": dict(packages),
        "cuda": dict(cuda_info),
        "git": dict(git_info),
        "models": [dict(m) for m in models],
    }
    return lock


def serialize_lock(lock):
    """Deterministic serialization: sorted keys, fixed separators, newline."""
    return json.dumps(lock, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def parse_lock(text):
    """Parse + validate a serialized lock; raises ValueError on mismatch."""
    try:
        lock = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"environment lock is not valid JSON: {exc}") from exc
    if not isinstance(lock, dict):
        raise ValueError("environment lock must be a JSON object")
    if lock.get("schema") != ENV_LOCK_SCHEMA:
        raise ValueError(
            f"unsupported lock schema {lock.get('schema')!r}; "
            f"expected {ENV_LOCK_SCHEMA!r}"
        )
    missing = [key for key in REQUIRED_LOCK_SECTIONS if key not in lock]
    if missing:
        raise ValueError(f"environment lock missing sections: {missing}")
    if not isinstance(lock["packages"], dict) or not isinstance(lock["models"], list):
        raise ValueError("environment lock sections have wrong types")
    return lock


def collect_environment_lock(model_ids=(), repo_root=None, cache_dir=None,
                             label=None, include_timestamp=True):
    created = datetime.now(timezone.utc).isoformat() if include_timestamp else None
    return build_environment_lock(
        packages=package_versions(),
        python_info=python_platform_info(),
        cuda_info=cuda_device_info(),
        git_info=git_state(repo_root),
        models=[resolve_hf_commit(m, cache_dir) for m in model_ids],
        label=label,
        created_at=created,
    )


# ---------------------------------------------------------------------------
# Embedded per-run provenance
# ---------------------------------------------------------------------------

def collect_run_provenance(files=None, model_ids=(), repo_root=None,
                           cache_dir=None, extra=None):
    """Compact provenance block for embedding into a result artifact.

    ``files``: mapping name -> path; each is SHA-256 hashed.
    ``model_ids``: HF ids whose cached commits are resolved locally.
    """
    prov = {
        "schema": RUN_PROVENANCE_SCHEMA,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "argv": list(sys.argv),
        "python": python_platform_info(),
        "packages": package_versions(),
        "cuda": cuda_device_info(),
        "git": git_state(repo_root),
    }
    if files:
        prov["files"] = {}
        for name, path in files.items():
            entry = {"path": str(path)}
            try:
                entry["sha256"] = sha256_file(path)
            except OSError as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
            prov["files"][name] = entry
    if model_ids:
        prov["hf_models"] = [resolve_hf_commit(m, cache_dir) for m in model_ids]
    if extra:
        prov["extra"] = extra
    return prov


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Environment-lock export/validation (no network access)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    exp = sub.add_parser("export", help="write the lock for THIS environment")
    exp.add_argument("--out", required=True)
    exp.add_argument("--label", default=None,
                     help="e.g. 'spark-empathy' when captured on the Spark")
    exp.add_argument("--models", nargs="*", default=[],
                     help="HF model ids to resolve from the local cache")
    exp.add_argument("--cache-dir", default=None)
    exp.add_argument("--omit-timestamp", action="store_true",
                     help="byte-deterministic output for identical envs")

    val = sub.add_parser("validate", help="parse/validate an existing lock")
    val.add_argument("--lock", required=True)

    args = parser.parse_args(argv)
    if args.command == "export":
        lock = collect_environment_lock(
            model_ids=args.models,
            cache_dir=args.cache_dir,
            label=args.label,
            include_timestamp=not args.omit_timestamp,
        )
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(serialize_lock(lock))
        print(f"wrote {out}")
        return 0
    lock = parse_lock(Path(args.lock).read_text())
    absent = sorted(k for k, v in lock["packages"].items() if v is None)
    print(f"valid lock: schema={lock['schema']} label={lock.get('label')!r} "
          f"python={lock['python'].get('python')} "
          f"packages={sum(v is not None for v in lock['packages'].values())} "
          f"absent={absent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
