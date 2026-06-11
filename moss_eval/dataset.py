from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import Iterable

AUDIO_KEYS = ("audio_path", "audio_file", "audio", "wav", "path")


@dataclass(frozen=True)
class DatasetItem:
    index: int
    audio_path: Path
    output_name: str
    record: dict


@dataclass(frozen=True)
class JsonlDataset:
    name: str
    jsonl_path: Path
    items: list[DatasetItem]

    def __len__(self) -> int:
        return len(self.items)


def _sanitize_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name or "audio"


def _audio_path_from_record(record: dict, line_no: int, jsonl_path: Path) -> Path:
    for key in AUDIO_KEYS:
        value = record.get(key)
        if value:
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                path = (jsonl_path.parent / path).resolve()
            return path
    raise ValueError(
        f"{jsonl_path}:{line_no} does not contain an audio path field. "
        f"Supported keys: {', '.join(AUDIO_KEYS)}"
    )


def load_jsonl_dataset(jsonl_path: str | Path, name: str | None = None, check_exists: bool = True) -> JsonlDataset:
    jsonl_path = Path(jsonl_path).expanduser().resolve()
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    items: list[DatasetItem] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                raise ValueError(f"{jsonl_path}:{line_no} is empty; remove blank lines for reproducible indexing")
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{jsonl_path}:{line_no} is not valid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{jsonl_path}:{line_no} must be a JSON object")
            audio_path = _audio_path_from_record(record, line_no, jsonl_path)
            if check_exists and not audio_path.is_file():
                raise FileNotFoundError(f"Audio file not found at {jsonl_path}:{line_no}: {audio_path}")
            stem = _sanitize_name(audio_path.stem)
            output_name = f"{len(items):06d}_{stem}.flac"
            items.append(DatasetItem(len(items), audio_path, output_name, record))

    if not items:
        raise ValueError(f"No audio records found in JSONL: {jsonl_path}")
    dataset_name = name or jsonl_path.stem
    return JsonlDataset(dataset_name, jsonl_path, items)


def discover_datasets(config: dict, check_exists: bool = True) -> list[JsonlDataset]:
    datasets_cfg = config.get("datasets")
    if datasets_cfg:
        datasets = []
        for item in datasets_cfg:
            if isinstance(item, str):
                datasets.append(load_jsonl_dataset(item, check_exists=check_exists))
            else:
                datasets.append(load_jsonl_dataset(item["jsonl"], name=item.get("name"), check_exists=check_exists))
        return datasets

    dataset_dir = config.get("dataset_dir") or config.get("test_dataset_dir")
    if not dataset_dir:
        raise ValueError("Config must define either `datasets` or `dataset_dir`")
    root = Path(dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    jsonl_files = sorted(root.rglob("*.jsonl"))
    if not jsonl_files:
        raise ValueError(f"No .jsonl files found under {root}")
    return [load_jsonl_dataset(path, check_exists=check_exists) for path in jsonl_files]


def validate_dataset(dataset: JsonlDataset) -> list[str]:
    errors: list[str] = []
    names = set()
    for item in dataset.items:
        if item.output_name in names:
            errors.append(f"Duplicate output name: {item.output_name}")
        names.add(item.output_name)
        if not item.audio_path.is_file():
            errors.append(f"Missing audio: {item.audio_path}")
    return errors
