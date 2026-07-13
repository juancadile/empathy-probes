"""Static acceptance tests for every direct parameter-edit caller."""

import ast
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.weight_edit_inventory import (  # noqa: E402
    HISTORICAL_ONLY, WEIGHT_EDIT_CALLER_CLASSIFICATION, _calls_weight_edit,
    scan_direct_edit_callers,
)
from src.analysis.logit_lens_trajectory import (  # noqa: E402
    EVIDENCE_ELIGIBILITY, save_raw_npz,
)


def test_frozen_inventory_exactly_matches_ast_scan():
    assert scan_direct_edit_callers(ROOT) == sorted(
        WEIGHT_EDIT_CALLER_CLASSIFICATION)


def test_import_alias_calls_are_detected():
    tree = ast.parse(
        "from weight_orthogonalization import orthogonalize_component as edit\n"
        "edit(model, component, direction)\n")
    assert _calls_weight_edit(tree) is True


def test_new_unclassified_caller_fails_inventory_equality(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "new_science.py").write_text(
        "from weight_orthogonalization import orthogonalize_component as e\n"
        "e(model, component, direction)\n")
    assert scan_direct_edit_callers(tmp_path) == ["src/new_science.py"]
    assert set(scan_direct_edit_callers(tmp_path)) != set(
        WEIGHT_EDIT_CALLER_CLASSIFICATION)


def test_historical_callers_have_no_accepted_mode_and_persist_marker():
    for rel, classification in WEIGHT_EDIT_CALLER_CLASSIFICATION.items():
        if classification != HISTORICAL_ONLY:
            continue
        text = (ROOT / rel).read_text()
        assert 'EVIDENCE_ELIGIBILITY = "historical_or_exploratory_only"' in text
        assert '"evidence_eligibility": EVIDENCE_ELIGIBILITY' in text
        assert 'add_argument("--run-mode"' not in text
        assert "refusing to overwrite historical" in text


def test_logit_lens_npz_embeds_nonconfirmatory_eligibility(tmp_path):
    path = tmp_path / "raw.npz"
    pointer = save_raw_npz(path, traj=np.zeros((2, 3)), flip_signs=np.ones(2))
    payload = np.load(path)
    assert payload["evidence_eligibility"].item() == EVIDENCE_ELIGIBILITY
    assert pointer["evidence_eligibility"] == EVIDENCE_ELIGIBILITY
