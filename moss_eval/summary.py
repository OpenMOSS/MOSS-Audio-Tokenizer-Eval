from __future__ import annotations

from pathlib import Path
import csv
import json
from collections import defaultdict

DEFAULT_METRICS = [
    "sim", "stoi", "pesq-nb", "pesq-wb", "mel_loss", "spectral_convergence",
    "sdr", "sisdr", "utmos", "stft", "visqol_speech", "visqol_audio",
]


def summarize(exp_dir: str | Path, output_dir: str | Path, metrics: list[str] | None = None) -> list[Path]:
    exp_dir = Path(exp_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = metrics or DEFAULT_METRICS
    records_by_dataset = defaultdict(list)

    for result_path in sorted(exp_dir.rglob("results.json")):
        model_name = str(result_path.parent.relative_to(exp_dir)).replace("\\", "/")
        with result_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        bps = payload.get("bps")
        meta = payload.get("_meta", {})
        if bps is None:
            bps = meta.get("bps")
        for dataset_name, values in payload.items():
            if dataset_name.startswith("_") or dataset_name == "bps" or not isinstance(values, dict):
                continue
            row = {"model": model_name, "bps": bps}
            for metric in metrics:
                row[metric] = values.get(metric)
            records_by_dataset[dataset_name].append(row)

    written: list[Path] = []
    header = ["model", "bps"] + metrics
    for dataset_name, rows in records_by_dataset.items():
        path = output_dir / f"{dataset_name}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        written.append(path)
    return written
