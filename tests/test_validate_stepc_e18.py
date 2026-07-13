"""resume2 stepC skip-validation: a skip is only allowed when the existing
e18.json exactly matches the current chain config (conditions_spec, all four
cells, provenance hashes); ANY mismatch must exit nonzero so the chain aborts
instead of silently skipping or overwriting."""

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_stepc_e18.py"

WRITERS = "L20MLP,L19MLP"
SUPPS = "L18H13,L20H10,L19H12,L17H7"
RAND = "L1MLP,L2MLP,L18H8,L20H5,L19H7,L17H2"
CELL_NAMES = ["cost_axis", "nonsocial_axis", "need_mild", "need_resolved"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_fixture(tmp_path):
    direction = tmp_path / "direction.npy"
    direction.write_bytes(b"fake-direction-bytes")
    cells = {}
    for n in CELL_NAMES:
        p = tmp_path / f"{n}.jsonl"
        p.write_text(json.dumps({"scenario_id": "fam0", "cost_level": "free"}) + "\n")
        cells[n] = str(p)
    spec = {"positive_writers_k2": WRITERS, "suppressors_k4": SUPPS,
            "targeted_k6": f"{WRITERS},{SUPPS}", "random_k6": RAND}
    run = {
        "conditions_spec": spec,
        "provenance": {
            "direction": {"path": str(direction), "sha256": sha256(direction)},
            "cell_files": {n: {"path": p, "sha256": sha256(p)}
                           for n, p in cells.items()},
        },
        "conditions": {c: {n: {"per_pair_delta": [0.1, -0.2]} for n in CELL_NAMES}
                       for c in spec},
    }
    run_path = tmp_path / "e18.json"
    run_path.write_text(json.dumps(run))
    return run, run_path, direction, cells


def validate(run_path, direction, cells):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--run", str(run_path),
         "--direction", str(direction), "--writers", WRITERS,
         "--suppressors", SUPPS, "--random", RAND,
         "--cells"] + [f"{n}={p}" for n, p in cells.items()],
        capture_output=True, text=True)


def rewrite(run_path, run):
    run_path.write_text(json.dumps(run))


def test_valid_artifact_passes(tmp_path):
    run, run_path, direction, cells = make_fixture(tmp_path)
    r = validate(run_path, direction, cells)
    assert r.returncode == 0, r.stderr
    assert "VALID" in r.stdout


def test_conditions_spec_mismatch_fails(tmp_path):
    run, run_path, direction, cells = make_fixture(tmp_path)
    bad = copy.deepcopy(run)
    bad["conditions_spec"]["suppressors_k4"] = "L18H13,L20H10,L19H12,L17H0"
    rewrite(run_path, bad)
    r = validate(run_path, direction, cells)
    assert r.returncode != 0 and "conditions_spec mismatch" in r.stderr


def test_legacy_artifact_without_spec_fails(tmp_path):
    run, run_path, direction, cells = make_fixture(tmp_path)
    bad = copy.deepcopy(run)
    del bad["conditions_spec"]
    rewrite(run_path, bad)
    assert validate(run_path, direction, cells).returncode != 0


def test_missing_cell_in_condition_fails(tmp_path):
    run, run_path, direction, cells = make_fixture(tmp_path)
    bad = copy.deepcopy(run)
    del bad["conditions"]["suppressors_k4"]["need_resolved"]
    rewrite(run_path, bad)
    r = validate(run_path, direction, cells)
    assert r.returncode != 0 and "missing cell" in r.stderr


def test_direction_hash_mismatch_fails(tmp_path):
    run, run_path, direction, cells = make_fixture(tmp_path)
    direction.write_bytes(b"DIFFERENT-direction-bytes")
    r = validate(run_path, direction, cells)
    assert r.returncode != 0 and "direction sha256" in r.stderr


def test_cell_file_hash_mismatch_fails(tmp_path):
    run, run_path, direction, cells = make_fixture(tmp_path)
    Path(cells["need_mild"]).write_text("edited after the run\n")
    r = validate(run_path, direction, cells)
    assert r.returncode != 0 and "sha256" in r.stderr


def test_unparseable_json_fails(tmp_path):
    run, run_path, direction, cells = make_fixture(tmp_path)
    run_path.write_text("{truncated")
    r = validate(run_path, direction, cells)
    assert r.returncode != 0 and "cannot parse" in r.stderr
