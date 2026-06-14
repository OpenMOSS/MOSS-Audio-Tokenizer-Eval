#!/usr/bin/env python3
"""Prepare AudioLLMs/librispeech_test_clean for MOSS codec evaluation.

The dataset stores audio as WAV bytes inside parquet files. This script
downloads or reads those parquet files, extracts WAV files, and writes a JSONL
with {"audio_path": "..."} records accepted by moss-eval.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

import pyarrow.parquet as pq


DEFAULT_DATASET_ID = "AudioLLMs/librispeech_test_clean"
DEFAULT_REVISION = "main"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for parquet files, extracted WAV files, JSONL, and reports.",
    )
    parser.add_argument(
        "--local-parquet-dir",
        help="Use already downloaded parquet files instead of downloading from Hugging Face.",
    )
    parser.add_argument(
        "--download-method",
        choices=("auto", "hf_hub", "curl"),
        default="auto",
        help="Download backend when --local-parquet-dir is not set.",
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Parquet batch size for extraction.",
    )
    parser.add_argument(
        "--skip-existing-audio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip writing extracted WAV files that already exist.",
    )
    parser.add_argument(
        "--slug",
        default="librispeech_test_clean",
        help="Prefix used for generated JSONL/report filenames.",
    )
    return parser.parse_args()


def sanitize_stem(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    value = value.strip("._")
    return value or "audio"


def list_repo_parquets(dataset_id: str, revision: str) -> list[str]:
    try:
        from huggingface_hub import list_repo_files
    except ImportError as exc:
        raise RuntimeError("huggingface-hub is required for downloading") from exc

    files = list_repo_files(dataset_id, repo_type="dataset", revision=revision)
    parquets = sorted(f for f in files if f.endswith(".parquet"))
    if not parquets:
        raise RuntimeError(f"No parquet files found in dataset repo: {dataset_id}@{revision}")
    return parquets


def download_with_hf_hub(dataset_id: str, revision: str, filename: str, output_dir: Path, force: bool) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface-hub is required for hf_hub download") from exc

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
    path = hf_hub_download(
        repo_id=dataset_id,
        repo_type="dataset",
        revision=revision,
        filename=filename,
        local_dir=str(output_dir),
        force_download=force,
    )
    return Path(path)


def download_with_curl(dataset_id: str, revision: str, filename: str, output_dir: Path) -> Path:
    curl = shutil.which("curl")
    if not curl:
        raise RuntimeError("curl is not available for fallback download")
    dest = output_dir / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = (
        f"https://huggingface.co/datasets/{dataset_id}/resolve/"
        f"{quote(revision)}/{quote(filename, safe='/')}"
    )
    cmd = [
        curl,
        "-L",
        "--retry",
        "20",
        "--retry-delay",
        "5",
        "--connect-timeout",
        "30",
        "--speed-time",
        "120",
        "--speed-limit",
        "1024",
        "-C",
        "-",
        "-o",
        str(dest),
        url,
    ]
    subprocess.run(cmd, check=True)
    return dest


def prepare_parquet_dir(args: argparse.Namespace, output_dir: Path) -> Path:
    if args.local_parquet_dir:
        parquet_dir = Path(args.local_parquet_dir).expanduser().resolve()
        if not parquet_dir.is_dir():
            raise FileNotFoundError(f"local parquet dir not found: {parquet_dir}")
        return parquet_dir

    parquet_dir = output_dir / "hf_dataset"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    parquet_files = list_repo_parquets(args.dataset_id, args.revision)

    for filename in parquet_files:
        dest = parquet_dir / filename
        if dest.exists() and not args.force_download:
            print(f"skip existing parquet: {dest}", flush=True)
            continue

        last_error: Exception | None = None
        if args.download_method in ("auto", "hf_hub"):
            try:
                path = download_with_hf_hub(
                    args.dataset_id, args.revision, filename, parquet_dir, args.force_download
                )
                print(f"downloaded via hf_hub: {path}", flush=True)
                continue
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
                if args.download_method == "hf_hub":
                    raise
                print(f"hf_hub download failed for {filename}: {exc}", file=sys.stderr, flush=True)

        if args.download_method in ("auto", "curl"):
            try:
                path = download_with_curl(args.dataset_id, args.revision, filename, parquet_dir)
                print(f"downloaded via curl: {path}", flush=True)
                continue
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
                if args.download_method == "curl":
                    raise

        raise RuntimeError(f"Failed to download {filename}") from last_error

    return parquet_dir


def find_parquets(parquet_dir: Path) -> list[Path]:
    files = sorted(parquet_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {parquet_dir}")
    return files


def context_stem(context: dict, row_index: int, slug: str) -> str:
    source = context.get("path") or context.get("filename") or context.get("file")
    if source:
        stem = sanitize_stem(Path(str(source)).stem)
        if stem:
            return f"{row_index:06d}_{stem}"
    return f"{sanitize_stem(slug)}_{row_index:06d}"


def unique_output_path(directory: Path, stem: str, suffix: str, used: set[Path]) -> Path:
    path = directory / f"{stem}{suffix}"
    if path not in used:
        used.add(path)
        return path
    idx = 1
    while True:
        candidate = directory / f"{stem}_{idx:03d}{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        idx += 1


def write_jsonl(path: Path, audio_paths: list[Path]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for audio_path in audio_paths:
            f.write(json.dumps({"audio_path": str(audio_path)}, ensure_ascii=False) + "\n")


def extract_dataset(args: argparse.Namespace, parquet_dir: Path, output_dir: Path) -> dict:
    audio_dir = output_dir / "extracted_wav"
    audio_dir.mkdir(parents=True, exist_ok=True)

    parquets = find_parquets(parquet_dir)
    audio_paths: list[Path] = []
    used_output_paths: set[Path] = set()
    row_index = 0

    for parquet_path in parquets:
        pf = pq.ParquetFile(parquet_path)
        schema_names = set(pf.schema_arrow.names)
        if "context" not in schema_names:
            raise ValueError(f"{parquet_path} has no context column")

        for batch in pf.iter_batches(batch_size=args.batch_size, columns=["context"]):
            for row in batch.to_pylist():
                context = row.get("context") or {}
                blob = context.get("bytes")
                if blob is None:
                    raise ValueError(f"row {row_index} in {parquet_path} has no context.bytes")

                stem = context_stem(context, row_index, args.slug)
                output_path = unique_output_path(audio_dir, stem, ".wav", used_output_paths)
                if not output_path.exists() or not args.skip_existing_audio:
                    output_path.write_bytes(blob)

                audio_paths.append(output_path)
                row_index += 1
                if row_index % 500 == 0:
                    print(f"processed rows: {row_index}", flush=True)

    jsonl_path = output_dir / f"z-{sanitize_stem(args.slug)}_hf_order.jsonl"
    write_jsonl(jsonl_path, audio_paths)

    return {
        "dataset_id": args.dataset_id,
        "revision": args.revision,
        "parquet_dir": str(parquet_dir),
        "parquet_files": [str(p) for p in parquets],
        "rows": len(audio_paths),
        "extracted_audio_dir": str(audio_dir),
        "jsonl": str(jsonl_path),
    }


def write_summary(output_dir: Path, summary: dict) -> None:
    summary_path = output_dir / "prepare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    report_path = output_dir / "prepare_report.md"
    lines = [
        "# AudioLLMs/librispeech_test_clean Preparation Report",
        "",
        "## Summary",
        "",
        f"- Dataset: `{summary['dataset_id']}`",
        f"- Revision: `{summary['revision']}`",
        f"- Rows: `{summary['rows']}`",
        f"- Extracted audio dir: `{summary['extracted_audio_dir']}`",
        f"- JSONL: `{summary['jsonl']}`",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir = prepare_parquet_dir(args, output_dir)
    summary = extract_dataset(args, parquet_dir, output_dir)
    write_summary(output_dir, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
