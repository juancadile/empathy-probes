"""Offline integrity tests for both EIA game drivers."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import src.eia_validation.e27_game_variants as e27  # noqa: E402
import src.eia_validation.run_eia_local as local  # noqa: E402

CURRENT_KEY = "gemma2_9b_it_resid_2026-07-12"
CURRENT_DIRECTION = (
    "results/controlled_directions_gemma2_9b_it/direction_M_resid_block20.npy")


@pytest.mark.parametrize("main", [local.main, e27.main])
def test_driver_help_imports_without_model_load(main):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_edit_condition_omission_and_accepted_baseline_omission_fail():
    with pytest.raises(local.ComponentSetError):
        local.resolve_condition_components(
            "suppressors", set_key=None,
            explicit={"suppressors": None}, model="google/gemma-2-9b-it",
            direction=CURRENT_DIRECTION, run_mode="exploratory")
    with pytest.raises(local.ComponentSetError, match="baseline included"):
        local.resolve_condition_components(
            "baseline", set_key=None, explicit={},
            model="google/gemma-2-9b-it", direction=CURRENT_DIRECTION,
            run_mode="accepted")


def test_baseline_and_edit_share_registry_direction_identity():
    baseline = local.resolve_condition_components(
        "baseline", set_key=CURRENT_KEY, explicit={},
        model="google/gemma-2-9b-it", direction=CURRENT_DIRECTION,
        run_mode="exploratory")
    edited = local.resolve_condition_components(
        "suppressors", set_key=CURRENT_KEY,
        explicit={"suppressors": None}, model="google/gemma-2-9b-it",
        direction=CURRENT_DIRECTION, run_mode="exploratory")
    assert baseline["source"]["registry_key"] == edited["source"]["registry_key"]
    assert (baseline["source"]["direction_binding"] ==
            edited["source"]["direction_binding"])


@pytest.mark.parametrize("scenarios,seeds,match", [
    (["the_listener", "the_listener"], [1], "scenarios contain duplicates"),
    (["the_listener"], [1, 1], "seeds contain duplicates"),
    (["the_listener"], ["1"], "seeds must be integers"),
])
def test_run_dimensions_are_unique_and_typed(scenarios, seeds, match):
    with pytest.raises(local.EvidenceRunError, match=match):
        local.validate_unique_run_dimensions(scenarios, seeds)


def test_player_seed_derivation_resets_per_fresh_player():
    assert local.derive_player_call_seed(22, 1) == 22 * 100003 + 1
    assert local.derive_player_call_seed(22, 1) == local.derive_player_call_seed(22, 1)
    assert local.derive_player_call_seed(22, 2) != local.derive_player_call_seed(22, 1)


def test_manifest_validation_and_behavioral_tree_hash(tmp_path):
    with pytest.raises(local.EvidenceRunError, match="manifest incomplete"):
        local.validate_run_manifest({"schema": local.RUN_MANIFEST_SCHEMA})
    tree = tmp_path / "eia"
    tree.mkdir()
    (tree / "game.py").write_text("x = 1\n")
    (tree / "map.png").write_bytes(b"map")
    first = local.content_tree_hash(tree)
    (tree / "map.png").write_bytes(b"changed")
    assert local.content_tree_hash(tree) != first


def test_e27_declares_adapted_assay_not_original():
    assert "ADAPTED listener assay" in e27.__doc__
    assert "ORIGINAL listener" not in e27.__doc__
