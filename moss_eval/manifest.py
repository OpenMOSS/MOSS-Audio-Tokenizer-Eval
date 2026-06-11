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


def shard_manifest_path(output_dir: str | Path, rank: int, world_size: int) -> Path:
    return Path(output_dir) / f"manifest.rank{rank:05d}-of-{world_size:05d}.json"


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


def write_shard_manifest(output_dir: str | Path, manifest: dict, *, rank: int, world_size: int, output_names: list[str]) -> None:
    payload = dict(manifest)
    payload["shard"] = {
        "rank": int(rank),
        "world_size": int(world_size),
        "output_names": list(output_names),
    }
    path = shard_manifest_path(output_dir, rank, world_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


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


def files_exist(output_dir: str | Path, output_names: list[str]) -> bool:
    output_dir = Path(output_dir)
    for name in output_names:
        if not (output_dir / "gt_audios" / name).is_file():
            return False
        if not (output_dir / "syn_audios" / name).is_file():
            return False
    return True


def output_complete(output_dir: str | Path, expected: dict) -> bool:
    if load_manifest(output_dir) != expected:
        return False
    names = expected.get("dataset", {}).get("output_names", [])
    return files_exist(output_dir, names)


def output_files_complete(output_dir: str | Path, expected: dict) -> bool:
    names = expected.get("dataset", {}).get("output_names", [])
    return files_exist(output_dir, names)


def shard_complete(output_dir: str | Path, expected: dict, *, rank: int, world_size: int, output_names: list[str]) -> bool:
    if output_complete(output_dir, expected) and files_exist(output_dir, output_names):
        return True
    path = shard_manifest_path(output_dir, rank, world_size)
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    expected_payload = dict(expected)
    expected_payload["shard"] = {
        "rank": int(rank),
        "world_size": int(world_size),
        "output_names": list(output_names),
    }
    return payload == expected_payload and files_exist(output_dir, output_names)
