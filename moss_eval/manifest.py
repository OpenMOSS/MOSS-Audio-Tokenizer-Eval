from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Any

MANIFEST_NAME = "manifest.json"


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def manifest_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / MANIFEST_NAME


def load_manifest(output_dir: str | Path) -> dict | None:
    path = manifest_path(output_dir)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_manifest(output_dir: str | Path, manifest: dict) -> None:
    path = manifest_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)


def build_manifest(dataset, model_meta: dict[str, Any], nq: int | None) -> dict:
    return {
        "schema_version": 1,
        "dataset": {
            "name": dataset.name,
            "jsonl_path": str(dataset.jsonl_path),
            "jsonl_sha256": file_sha256(dataset.jsonl_path),
            "num_items": len(dataset.items),
            "output_names": [item.output_name for item in dataset.items],
        },
        "model": model_meta,
        "nq": nq,
    }


def output_complete(output_dir: str | Path, expected: dict) -> bool:
    output_dir = Path(output_dir)
    if load_manifest(output_dir) != expected:
        return False
    names = expected.get("dataset", {}).get("output_names", [])
    for name in names:
        if not (output_dir / "gt_audios" / name).is_file():
            return False
        if not (output_dir / "syn_audios" / name).is_file():
            return False
    return True
