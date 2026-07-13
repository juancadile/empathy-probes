import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data_generation"))
import audit_gate1_tokenization as token_audit  # noqa: E402


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return list(range(len(text.split())))

    def convert_ids_to_tokens(self, ids):
        return [f"t{item}" for item in ids]


def test_audit_file_records_exact_tail_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(token_audit, "ROOT", tmp_path)
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        '{"family_id":"f1","variant_id":"v0","partition":"p",'
        '"positive_tail":"one two three","negative_tail":"one twothree"}\n')
    report = token_audit.audit_file(path, FakeTokenizer())
    assert report["passed"]
    assert report["unique_tail_pairs"][0]["positive_token_ids"] == [0, 1, 2]
    assert report["unique_tail_pairs"][0]["token_length_difference"] == 1
