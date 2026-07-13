"""E18c need-level LEVEL estimands: family mean effects, paired contrasts,
and the writer need-gated gate (mean-effect contrast + resolved equivalence),
kept distinct from the cost-slope estimands. Pure numpy, no model."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "analysis"))

from e18c_slope_interaction import (  # noqa: E402
    MEAN_EQUIV_BOUND, family_cost_slopes, family_mean_effects, paired_contrast,
    writer_need_gate,
)


def _cell(fams_effects, ranks_per_fam=(0, 1, 2, 3), slope=0.0, noise=None):
    """Per-pair deltas: family level effect + slope*rank (+ optional noise)."""
    fams, ranks, deltas = [], [], []
    for i, (fam, level) in enumerate(sorted(fams_effects.items())):
        for r in ranks_per_fam:
            fams.append(fam)
            ranks.append(r)
            deltas.append(level + slope * r
                          + (noise[len(deltas)] if noise is not None else 0.0))
    return np.array(deltas), np.array(fams), np.array(ranks)


def test_family_mean_effects_recovers_levels():
    levels = {"famA": -0.2, "famB": 0.1, "famC": 0.0}
    d, fams, _ = _cell(levels)
    means = family_mean_effects(d, fams)
    for f, lv in levels.items():
        assert abs(means[f] - lv) < 1e-12


def test_mean_effect_is_slope_invariant_and_slopes_are_level_invariant():
    # the two estimands must not leak into each other
    levels = {f"fam{i}": 0.05 * i for i in range(4)}
    d, fams, ranks = _cell(levels, slope=-0.08)
    means = family_mean_effects(d, fams)
    slopes = family_cost_slopes(d, fams, ranks)
    for f, lv in levels.items():
        assert abs(means[f] - (lv + -0.08 * 1.5)) < 1e-9  # level + slope*mean(rank)
        assert abs(slopes[f] - (-0.08)) < 1e-9  # slope untouched by the level


def test_paired_contrast_matches_family_differences():
    urgent = {f"fam{i}": -0.2 + 0.01 * i for i in range(8)}
    resolved = {f"fam{i}": 0.01 * i for i in range(8)}
    stat = paired_contrast(urgent, resolved)
    assert stat["n_paired_families"] == 8
    assert abs(stat["mean"] - (-0.2)) < 1e-12
    assert stat["ci95_excludes_zero"] is True
    assert stat["sign_positive_families"] == 0


def test_paired_contrast_uses_common_families_only():
    stat = paired_contrast({"a": 1.0, "b": 2.0, "only_a": 9.0},
                           {"a": 0.5, "b": 1.0, "only_b": -9.0})
    assert stat["n_paired_families"] == 2
    assert abs(stat["mean"] - 0.75) < 1e-12


def test_writer_need_gate_requires_both_conditions():
    hit = {"ci95": [-0.3, -0.1], "ci95_excludes_zero": True}
    miss = {"ci95": [-0.1, 0.05], "ci95_excludes_zero": False}
    resolved_null = {"ci95": [-0.02, 0.03]}
    resolved_nonnull = {"ci95": [0.06, 0.12]}
    assert writer_need_gate(hit, resolved_null)["satisfied"] is True
    # contrast significant but resolved effect not practically null -> fail
    assert writer_need_gate(hit, resolved_nonnull)["satisfied"] is False
    # resolved null but contrast CI covers zero -> fail
    assert writer_need_gate(miss, resolved_null)["satisfied"] is False
    # boundary: resolved CI exactly on the equivalence bound counts as within
    at_bound = {"ci95": [-MEAN_EQUIV_BOUND, MEAN_EQUIV_BOUND]}
    assert writer_need_gate(hit, at_bound)[
        "resolved_mean_ci_within_equivalence_bound"] is True
    beyond = {"ci95": [-MEAN_EQUIV_BOUND - 1e-6, 0.0]}
    assert writer_need_gate(hit, beyond)[
        "resolved_mean_ci_within_equivalence_bound"] is False


def test_gate_reports_predeclared_bound():
    g = writer_need_gate({"ci95": [-0.3, -0.1], "ci95_excludes_zero": True},
                         {"ci95": [0.0, 0.01]})
    assert g["equivalence_bound"] == MEAN_EQUIV_BOUND
