"""Create or verify a deterministic SHA-256 manifest for artifact trees."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(roots: list[Path]) -> list[dict[str, object]]:
    files: list[Path] = []
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"artifact root is not a directory: {root}")
        files.extend(path for path in root.rglob("*") if path.is_file())
    unique = sorted(set(files), key=lambda path: str(path.relative_to(ROOT)))
    return [
        {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in unique
    ]


def aggregate_digest(files: list[dict[str, object]]) -> str:
    lines = "".join(f"{item['sha256']}  {item['path']}\n" for item in files)
    return hashlib.sha256(lines.encode()).hexdigest()


def build(roots: list[Path]) -> dict[str, object]:
    files = collect(roots)
    return {
        "schema": "empathy-action-probes/artifact-tree-manifest/1",
        "roots": [str(root.relative_to(ROOT)) for root in roots],
        "file_count": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "aggregate_sha256": aggregate_digest(files),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    roots = [path.resolve() for path in args.root]
    output = args.output.resolve()
    current = build(roots)
    if args.verify:
        expected = json.loads(output.read_text())
        if current != expected:
            raise SystemExit("artifact manifest verification failed")
        print(f"verified {current['file_count']} files: {current['aggregate_sha256']}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(current, indent=2) + "\n")
    print(f"wrote {current['file_count']} files: {current['aggregate_sha256']}")


if __name__ == "__main__":
    main()
