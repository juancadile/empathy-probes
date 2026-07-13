"""Versioned component-set registry (Integrity Repair A, 2026-07-13).

Weight-edit scripts must resolve their component sets EXPLICITLY: either from
a named, versioned entry in this registry (``--component-set <key>``) or from
full literal component specs on the command line. Silent module-level
defaults are forbidden — the 2026-07-13 science audit found that an
unqualified ``weight_orthogonalization.py`` rerun silently edited the
superseded pre-correction sets.

Rules:
  * A registry key is immutable once any accepted result references it.
    New selections get NEW keys; nothing is edited in place.
  * ``status`` documents which entry is current for a model. It is
    documentation only — scripts never fall back to "the current set";
    callers always name the exact key (or pass explicit specs) and the
    resolved key/specs are persisted in the result artifact.
  * Superseded entries remain so historical runs stay exactly reproducible,
    clearly labeled, and can never masquerade as current defaults.

Integrity Repair A QA (Q4, 2026-07-13): a component set is only meaningful
together with the direction it was selected against, so every resolution now
BINDS the supplied direction to the registry entry:

  * the caller must pass ``direction_path`` — literal component specs without
    explicit direction provenance are rejected;
  * the normalized direction path is compared against the registry entry, and
    the file's SHA-256 is verified against the entry's recorded hash whenever
    the file exists; expected and observed path/hash are persisted;
  * a mismatch aborts the resolution. The only escape is the explicit
    ``allow_direction_mismatch=True`` exploratory override, which is refused
    in ``accepted`` mode, prominently warned, and persisted as making the run
    ineligible for confirmatory evidence;
  * ``run_mode="accepted"`` additionally rejects superseded registry keys
    (historical reproduction stays possible in exploratory mode only) and
    requires the direction file to exist so its hash can be verified.
"""

import hashlib
from pathlib import Path

#: Direction files of record, SHA-256 hashed from the tracked repository
#: artifacts (independently verified at architect HEAD a8d68c2).
_DIR_GEMMA_PRECORRECTION = (
    "results/controlled_directions_gemma2_9b_it/direction_M_block20.npy")
_SHA_GEMMA_PRECORRECTION = (
    "ed4d034fc88fee241e61d90ae106ab6e416758f99fa02dc38f5f564ad5b60a96")
_DIR_GEMMA_RESID = (
    "results/controlled_directions_gemma2_9b_it/direction_M_resid_block20.npy")
_SHA_GEMMA_RESID = (
    "1b6d692e0e933d76c15f722fe996d01e1ca2ee8c8b72f19daf2f122cda294b42")
_DIR_LLAMA_GROUPED = (
    "results/controlled_directions_llama31_8b_it/direction_M_grouped_block15.npy")
_SHA_LLAMA_GROUPED = (
    "99b4daa965b914ae4aa014991fd47f179441a210a8bb31368cde30bb4458d78a")

COMPONENT_SET_VERSIONS = {
    "gemma2_9b_it_precorrection_2026-07-11": {
        "model": "google/gemma-2-9b-it",
        "status": "superseded",
        "superseded_by": "gemma2_9b_it_resid_2026-07-12",
        "direction": _DIR_GEMMA_PRECORRECTION,
        "direction_sha256": _SHA_GEMMA_PRECORRECTION,
        "positive_writers": "L19MLP,L20H15",
        "suppressors": "L19H12,L15H15,L17H13,L18H13",
        "targeted": "L19MLP,L19H12,L15H15,L17H13,L18H13,L20H15",
        "random": "L1MLP,L19H7,L15H9,L17H5,L18H12,L20H12",
        "notes": (
            "Pre-E24 (AUROC polarity bug) selection from the contaminated "
            "direction_M_block20 derivation. Kept only for exact historical "
            "reproduction of E13-E23-era artifacts; never for accepted runs."
        ),
    },
    "gemma2_9b_it_resid_2026-07-12": {
        "model": "google/gemma-2-9b-it",
        "status": "current",
        "superseded_by": None,
        "direction": _DIR_GEMMA_RESID,
        "direction_sha256": _SHA_GEMMA_RESID,
        "positive_writers": "L19MLP,L20MLP",
        "suppressors": "L18H13,L20H10,L19H12,L17H7",
        "targeted": "L19MLP,L20MLP,L18H13,L20H10,L19H12,L17H7",
        "random": "L1MLP,L2MLP,L18H8,L20H5,L19H7,L17H2",
        "notes": (
            "Rescue3 re-derivation (E25 stage 3) on d_resid under the "
            "pre-registered selection rule; sets of record for E25b/E26/E26b/"
            "E28/rescue3c. `random` is the composition-matched (2 MLP + 4 "
            "head) comparator used in E28/rescue3c — a fixed comparator, "
            "not a null."
        ),
    },
    "llama31_8b_it_grouped_2026-07-12": {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "status": "current",
        "superseded_by": None,
        "direction": _DIR_LLAMA_GROUPED,
        "direction_sha256": _SHA_LLAMA_GROUPED,
        "positive_writers": "L15MLP,L12H4",
        "suppressors": "L15H6,L14H27,L12H20,L11MLP",
        "targeted": "L15MLP,L12H4,L15H6,L14H27,L12H20,L11MLP",
        "random": "L1MLP,L12H15,L15H17,L14H6,L12H31,L2MLP",
        "notes": (
            "E17 amendment-v3 Llama sets (pre-registered stage 1-2 "
            "derivation). Llama-specific; the E17 stage-3 confirmatory "
            "result was a non-replication at confirmatory level."
        ),
    },
}

#: Roles a registry entry can provide to a script.
ROLES = ("positive_writers", "suppressors", "targeted", "random")


#: Run modes recognized by the resolution contract (QA Q4/Q6). ``None`` is
#: treated as exploratory for eligibility but still fails on a mismatch.
RUN_MODES = ("accepted", "exploratory")


class ComponentSetError(ValueError):
    """Raised when component sets cannot be resolved explicitly."""


def get_component_set(key):
    """Return the registry entry for ``key`` or raise with the known keys."""
    try:
        return COMPONENT_SET_VERSIONS[key]
    except KeyError:
        known = ", ".join(sorted(COMPONENT_SET_VERSIONS))
        raise ComponentSetError(
            f"unknown component-set version {key!r}; known keys: {known}"
        ) from None


def _default_repo_root():
    return Path(__file__).resolve().parents[1]


def normalize_direction_path(path, repo_root=None):
    """Normalize a direction path to a repo-relative POSIX string.

    Paths outside the repository stay absolute (still POSIX-normalized) so
    registry comparison fails loudly instead of by accident.
    """
    repo_root = Path(repo_root) if repo_root else _default_repo_root()
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute()
                else repo_root / candidate).resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bind_direction(direction_path, entry=None, set_key=None, run_mode=None,
                   allow_direction_mismatch=False, repo_root=None):
    """Verify and record the direction binding for a resolution (QA Q4).

    Returns the persisted binding record; raises ``ComponentSetError`` on a
    path or hash mismatch unless the explicit exploratory override applies.
    """
    repo_root = Path(repo_root) if repo_root else _default_repo_root()
    if direction_path is None:
        raise ComponentSetError(
            "direction provenance is required: pass direction_path so the "
            "component resolution is bound to the direction it was selected "
            "against (Integrity Repair A QA Q4). Literal component specs "
            "without an explicit direction are not accepted."
        )
    if run_mode is not None and run_mode not in RUN_MODES:
        raise ComponentSetError(
            f"unknown run_mode {run_mode!r}; expected one of {RUN_MODES}")
    if allow_direction_mismatch and run_mode == "accepted":
        raise ComponentSetError(
            "allow_direction_mismatch is an exploratory-only override and is "
            "refused in accepted mode"
        )
    normalized = normalize_direction_path(direction_path, repo_root)
    supplied = Path(direction_path)
    file_path = supplied if supplied.is_absolute() else repo_root / supplied
    observed_sha = _sha256_file(file_path) if file_path.is_file() else None
    binding = {
        "supplied_path": str(direction_path),
        "normalized_path": normalized,
        "observed_sha256": observed_sha,
        "registry_path": entry["direction"] if entry else None,
        "registry_sha256": entry.get("direction_sha256") if entry else None,
        "path_match": None,
        "hash_match": None,
        "mismatch_override": False,
    }
    problems = []
    if entry is not None:
        binding["path_match"] = normalized == entry["direction"]
        if not binding["path_match"]:
            problems.append(
                f"direction path mismatch: registry entry {set_key!r} is "
                f"bound to {entry['direction']!r} but the run supplied "
                f"{normalized!r}"
            )
        expected_sha = entry.get("direction_sha256")
        if expected_sha and observed_sha is not None:
            binding["hash_match"] = observed_sha == expected_sha
            if not binding["hash_match"]:
                problems.append(
                    f"direction hash mismatch: registry entry {set_key!r} "
                    f"records sha256 {expected_sha} but {normalized!r} "
                    f"hashes to {observed_sha}"
                )
    if run_mode == "accepted" and observed_sha is None:
        problems.append(
            f"accepted mode requires the direction file to exist for hash "
            f"verification; {normalized!r} was not found"
        )
    if problems:
        if allow_direction_mismatch and run_mode != "accepted":
            binding["mismatch_override"] = True
            binding["mismatch_details"] = problems
        else:
            raise ComponentSetError(
                "; ".join(problems)
                + ". Evidence-eligible runs must fail on a direction "
                "mismatch; pass allow_direction_mismatch=True only for an "
                "explicitly exploratory, non-confirmatory run."
            )
    return binding


def resolve_component_sets(roles, explicit, set_key=None, model=None,
                           direction_path=None, run_mode=None,
                           allow_direction_mismatch=False, repo_root=None):
    """Resolve component specs for ``roles`` with no silent defaults.

    Parameters
    ----------
    roles : sequence of role names required by the calling script.
    explicit : mapping role -> spec string (or None) from the CLI.
    set_key : optional registry key naming a versioned set-of-record.
    model : optional model id; mismatches against the registry entry abort.
    direction_path : REQUIRED direction provenance; compared (path and, when
        the file exists, SHA-256) against the registry entry (QA Q4).
    run_mode : "accepted" | "exploratory" | None. Accepted mode rejects
        superseded keys, mismatch overrides, and missing direction files.
    allow_direction_mismatch : explicit exploratory override for a direction
        mismatch; prominently warned and persisted as non-confirmatory.

    Returns a dict persisted verbatim into result artifacts::

        {"source": {...}, "sets": {role: "L19MLP,L20MLP", ...}}

    Every role must come from the named registry entry or an explicit CLI
    spec; if neither is available this raises ``ComponentSetError`` — there
    is deliberately no fallback.
    """
    explicit = dict(explicit or {})
    resolved, origins = {}, {}
    entry = None
    if set_key is not None:
        entry = get_component_set(set_key)
        if model is not None and entry["model"] != model:
            raise ComponentSetError(
                f"component set {set_key!r} is registered for "
                f"{entry['model']!r} but the run targets {model!r}; pass "
                "explicit component specs if this is intentional"
            )
        if run_mode == "accepted" and entry["status"] != "current":
            raise ComponentSetError(
                f"component set {set_key!r} is marked {entry['status']!r}"
                + (f" (superseded by {entry['superseded_by']})"
                   if entry.get("superseded_by") else "")
                + " and is rejected in accepted mode; exact historical "
                "reproduction must run as exploratory"
            )
    binding = bind_direction(
        direction_path, entry=entry, set_key=set_key, run_mode=run_mode,
        allow_direction_mismatch=allow_direction_mismatch,
        repo_root=repo_root,
    )
    for role in roles:
        spec = explicit.get(role)
        if spec:
            resolved[role] = spec
            origins[role] = "explicit"
        elif entry is not None and entry.get(role):
            resolved[role] = entry[role]
            origins[role] = f"registry:{set_key}"
        else:
            raise ComponentSetError(
                f"component set for role {role!r} was not specified. Pass "
                f"--component-set <key> (known: "
                f"{', '.join(sorted(COMPONENT_SET_VERSIONS))}) or an "
                f"explicit spec; silent defaults were removed by Integrity "
                "Repair A (2026-07-13)."
            )
    warnings = []
    ineligibility = []
    if entry is not None and entry["status"] != "current":
        warnings.append(
            f"{set_key} is marked {entry['status']!r}"
            + (f" (superseded by {entry['superseded_by']})"
               if entry.get("superseded_by") else "")
            + "; do not use for new accepted runs"
        )
        ineligibility.append(f"superseded registry key {set_key!r}")
    if binding["mismatch_override"]:
        warnings.append(
            "DIRECTION MISMATCH OVERRIDE ACTIVE: the supplied direction does "
            "not match the registry binding ("
            + "; ".join(binding.get("mismatch_details", ()))
            + "). This run is exploratory only and INELIGIBLE for "
            "confirmatory evidence."
        )
        ineligibility.append("direction mismatch override in effect")
    if run_mode != "accepted":
        ineligibility.append(
            "run_mode is not 'accepted'" if run_mode is None or run_mode in RUN_MODES
            else f"unknown run_mode {run_mode!r}"
        )
    source = {
        "registry_key": set_key,
        "registry_status": entry["status"] if entry else None,
        "origins": origins,
        "run_mode": run_mode,
        "direction_binding": binding,
        "confirmatory_evidence_eligible": not ineligibility,
        "ineligibility_reasons": ineligibility,
    }
    if warnings:
        source["warning"] = " | ".join(warnings)
    return {"source": source, "sets": resolved}
