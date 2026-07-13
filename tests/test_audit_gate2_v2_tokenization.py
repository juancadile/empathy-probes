import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/gate_families/gate2_v2"


def test_gate2_v2_tokenizer_audit_is_pinned_and_passes():
    report = json.loads((DATA / "tokenizer_audit.json").read_text())
    assert report["passed"]
    assert report["model"] == "google/gemma-2-9b-it"
    assert report["revision"] == "11c9b309abf73637e4b6f9a3fa1e92e615547819"
    assert all(not item["failures"] for item in report["files"])


def test_gate2_v2_tokenizer_audit_binds_current_files():
    report = json.loads((DATA / "tokenizer_audit.json").read_text())
    for item in report["files"]:
        path = ROOT / item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]


def test_primary_status_actuality_cells_have_equal_prompt_positions():
    report = json.loads((DATA / "tokenizer_audit.json").read_text())
    by_name = {Path(item["path"]).name: item for item in report["files"]}
    assert by_name["wp1_families.jsonl"]["prompt_token_range_by_label"]["L_new"]["max"] == 0
    assert by_name["wp3_families.jsonl"]["prompt_token_range_by_label"]["observation"]["max"] == 0
    assert by_name["wp3_families.jsonl"]["prompt_token_range_by_label"]["agency"]["max"] == 0

