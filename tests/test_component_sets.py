"""Versioned component-set registry + explicit resolution (Integrity Repair A).

Acceptance criterion covered: no accepted experiment can silently use old
component defaults — every resolution path either names a versioned registry
entry, passes explicit specs, or raises.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from component_sets import (  # noqa: E402
    COMPONENT_SET_VERSIONS, ComponentSetError, get_component_set,
    resolve_component_sets,
)
from weight_orthogonalization import parse_component  # noqa: E402


def test_registry_entries_are_well_formed():
    for key, entry in COMPONENT_SET_VERSIONS.items():
        assert entry["status"] in ("current", "superseded"), key
        for role in ("positive_writers", "suppressors", "targeted", "random"):
            for spec in entry[role].split(","):
                parsed = parse_component(spec)  # raises on malformed specs
                assert parsed["name"] == spec
        # targeted is exactly writers + suppressors as a SET; historical
        # entries keep their original CLI ordering byte-exact
        assert set(entry["targeted"].split(",")) == (
            set(entry["positive_writers"].split(","))
            | set(entry["suppressors"].split(","))), key
        assert len(entry["targeted"].split(",")) == (
            len(entry["positive_writers"].split(","))
            + len(entry["suppressors"].split(","))), key


def test_registry_matches_sets_of_record():
    current = get_component_set("gemma2_9b_it_resid_2026-07-12")
    assert current["status"] == "current"
    assert current["positive_writers"] == "L19MLP,L20MLP"
    assert current["suppressors"] == "L18H13,L20H10,L19H12,L17H7"
    assert current["random"] == "L1MLP,L2MLP,L18H8,L20H5,L19H7,L17H2"

    superseded = get_component_set("gemma2_9b_it_precorrection_2026-07-11")
    assert superseded["status"] == "superseded"
    assert superseded["positive_writers"] == "L19MLP,L20H15"
    assert superseded["suppressors"] == "L19H12,L15H15,L17H13,L18H13"

    llama = get_component_set("llama31_8b_it_grouped_2026-07-12")
    assert llama["positive_writers"] == "L15MLP,L12H4"
    assert llama["suppressors"] == "L15H6,L14H27,L12H20,L11MLP"


def test_unknown_registry_key_raises():
    with pytest.raises(ComponentSetError, match="unknown component-set"):
        get_component_set("no_such_key")


def test_resolution_requires_explicitness():
    with pytest.raises(ComponentSetError, match="not specified"):
        resolve_component_sets(("positive_writers",), explicit={})
    with pytest.raises(ComponentSetError, match="not specified"):
        resolve_component_sets(("positive_writers", "suppressors"),
                               explicit={"positive_writers": "L1MLP"})


def test_resolution_from_registry_and_overrides():
    resolved = resolve_component_sets(
        ("positive_writers", "suppressors"),
        explicit={"positive_writers": None, "suppressors": "L9H9"},
        set_key="gemma2_9b_it_resid_2026-07-12",
        model="google/gemma-2-9b-it",
    )
    assert resolved["sets"]["positive_writers"] == "L19MLP,L20MLP"
    assert resolved["sets"]["suppressors"] == "L9H9"
    assert resolved["source"]["origins"]["positive_writers"] == (
        "registry:gemma2_9b_it_resid_2026-07-12")
    assert resolved["source"]["origins"]["suppressors"] == "explicit"
    assert "warning" not in resolved["source"]


def test_resolution_model_mismatch_raises():
    with pytest.raises(ComponentSetError, match="registered for"):
        resolve_component_sets(
            ("positive_writers",), explicit={},
            set_key="gemma2_9b_it_resid_2026-07-12",
            model="meta-llama/Llama-3.1-8B-Instruct",
        )


def test_superseded_set_carries_warning():
    resolved = resolve_component_sets(
        ("targeted", "random"), explicit={},
        set_key="gemma2_9b_it_precorrection_2026-07-11",
        model="google/gemma-2-9b-it",
    )
    assert "superseded" in resolved["source"]["warning"]
    assert resolved["source"]["registry_status"] == "superseded"


def test_explicit_only_resolution_records_origins():
    resolved = resolve_component_sets(
        ("random",), explicit={"random": "L1MLP,L2MLP"})
    assert resolved["sets"] == {"random": "L1MLP,L2MLP"}
    assert resolved["source"]["registry_key"] is None
    assert resolved["source"]["origins"] == {"random": "explicit"}
