import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "evaluation"))
import audit_gate1_r2b_v2 as audit  # noqa: E402


def test_primary_judge_is_pinned_and_independent():
    assert audit.MODEL == "Qwen/Qwen3-14B"
    assert len(audit.REVISION) == 40
    assert audit.SEED == 3010918835
