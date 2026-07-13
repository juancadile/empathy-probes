"""CPU-only tests for LB1/LB2 auditability helpers.

half_crystallization_index: every point from the returned index onward must
BOTH retain the final nonzero sign AND reach half the final magnitude; a zero
final value returns None (no sign to retain). Regression: a late sign crossing
with large magnitude must NOT count as crystallized before the crossing.

save_raw_npz (both modules): compressed round trip preserves arrays and ids
exactly; the returned pointer's sha256 matches the written bytes.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "analysis"))

import behavioral_dla  # noqa: E402
import logit_lens_trajectory  # noqa: E402
from logit_lens_trajectory import half_crystallization_index  # noqa: E402


class TestHalfCrystallizationIndex:
    def test_monotone_rise(self):
        assert half_crystallization_index([0.0, 0.2, 0.6, 1.0]) == 2

    def test_sign_crossing_after_half_magnitude_not_crystallized(self):
        # |-0.6| >= half(1.0) but the sign flips: the magnitude-only rule
        # (old bug) returned 0; sign retention must push the index past it
        assert half_crystallization_index([0.6, -0.6, 1.0]) == 2

    def test_zero_final_returns_none(self):
        assert half_crystallization_index([0.5, 0.3, 0.0]) is None

    def test_negative_final(self):
        assert half_crystallization_index([-0.1, -0.6, -0.8]) == 1

    def test_dip_below_half_resets_index(self):
        assert half_crystallization_index([0.6, 0.3, 0.8, 1.0]) == 2

    def test_crystallized_from_start(self):
        assert half_crystallization_index([0.6, 0.7, 1.0]) == 0

    def test_zero_intermediate_point_lacks_sign(self):
        # sign(0) == 0 != sign(final): a zero point cannot be crystallized
        assert half_crystallization_index([0.6, 0.0, 1.0]) == 2

    def test_final_point_always_qualifies_when_nonzero(self):
        assert half_crystallization_index([-1.0, 0.1]) == 1

    def test_returns_plain_int(self):
        idx = half_crystallization_index(np.array([0.2, 0.6, 1.0]))
        assert type(idx) is int


SAVERS = [behavioral_dla.save_raw_npz, logit_lens_trajectory.save_raw_npz]


@pytest.mark.parametrize("save_raw_npz", SAVERS,
                         ids=["behavioral_dla", "logit_lens_trajectory"])
def test_save_raw_npz_round_trip_and_sha256(save_raw_npz, tmp_path):
    rng = np.random.default_rng(0)
    arrays = {
        "H": rng.standard_normal((5, 3, 4)),
        "traj": rng.standard_normal((5, 7)),
        "flip_signs": np.array([1.0, -1.0, 1.0, -1.0, 1.0]),
        "scenario_ids": np.array(["mc_a", "mc_a", "mc_b", "mc_c", "mc_c"]),
        "pair_indices": np.arange(5),
    }
    path = tmp_path / "raw.npz"
    ptr = save_raw_npz(path, **arrays)

    assert {"path", "sha256"} <= set(ptr)
    assert ptr["path"] == str(path)
    assert ptr["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    loaded = np.load(path)
    assert set(arrays) <= set(loaded.files)
    for key, arr in arrays.items():
        assert np.array_equal(loaded[key], arr), key
