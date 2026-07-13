import json
import shutil
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "analysis"))

import reanalyze_family_matched_opus as matched  # noqa: E402


def test_family_matched_reanalysis_uses_exact_frozen_pools():
    result = matched.analyze()
    manifest = json.loads((matched.BUNDLE / "manifest.json").read_text())
    selected = manifest["forms"]["A"]["selected_families"]
    for label, item in result["results"].items():
        assert item["families"] == sorted(selected[label])
        for check in item["checks"].values():
            assert check["family_count"] == len(selected[label])
            assert sum(check["counts"].values()) == len(selected[label])
    assert result["role"].startswith("sensitivity only")


def test_manifest_tampering_fails_closed(tmp_path):
    bundle = tmp_path / "bundle"
    shutil.copytree(matched.BUNDLE, bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["seed"] += 1
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        matched.analyze(bundle=bundle)
