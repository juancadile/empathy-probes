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
"""

COMPONENT_SET_VERSIONS = {
    "gemma2_9b_it_precorrection_2026-07-11": {
        "model": "google/gemma-2-9b-it",
        "status": "superseded",
        "superseded_by": "gemma2_9b_it_resid_2026-07-12",
        "direction": "results/controlled_directions_gemma2_9b_it/direction_M_block20.npy",
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
        "direction": "results/controlled_directions_gemma2_9b_it/direction_M_resid_block20.npy",
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
        "direction": "results/controlled_directions_llama31_8b_it/direction_M_grouped_block15.npy",
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


def resolve_component_sets(roles, explicit, set_key=None, model=None):
    """Resolve component specs for ``roles`` with no silent defaults.

    Parameters
    ----------
    roles : sequence of role names required by the calling script.
    explicit : mapping role -> spec string (or None) from the CLI.
    set_key : optional registry key naming a versioned set-of-record.
    model : optional model id; mismatches against the registry entry abort.

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
    source = {
        "registry_key": set_key,
        "registry_status": entry["status"] if entry else None,
        "origins": origins,
    }
    if entry is not None and entry["status"] != "current":
        source["warning"] = (
            f"{set_key} is marked {entry['status']!r}"
            + (f" (superseded by {entry['superseded_by']})"
               if entry.get("superseded_by") else "")
            + "; do not use for new accepted runs"
        )
    return {"source": source, "sets": resolved}
