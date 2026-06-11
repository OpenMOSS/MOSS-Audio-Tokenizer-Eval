from __future__ import annotations

from pathlib import Path
import logging

from tqdm import tqdm

from moss_eval import audio as audio_utils
from moss_eval.config import config_models
from moss_eval.dataset import discover_datasets
from moss_eval.adapters import build_adapter
from moss_eval.manifest import build_manifest, output_complete, write_manifest
from moss_eval.rvq import nq_tag, parse_nq_spec


def _model_tag(model_cfg: dict, adapter) -> str:
    return str(model_cfg.get("tag") or model_cfg.get("name") or adapter.name).strip().replace(" ", "_")


def run_reconstruct(config: dict, *, force: bool = False, device: str | None = None, nq_override=None) -> list[Path]:
    exp_root = Path(config.get("exp_root", "exp")).expanduser().resolve()
    datasets = discover_datasets(config, check_exists=True)
    generated: list[Path] = []

    for model_cfg in config_models(config):
        adapter = build_adapter(model_cfg, device=device or config.get("device"))
        adapter.load()
        nq_spec = nq_override if nq_override is not None else model_cfg.get("nq", config.get("nq"))
        nq_values = parse_nq_spec(nq_spec, max_nq=adapter.max_nq, is_tokenizer=adapter.is_tokenizer)
        tag = _model_tag(model_cfg, adapter)

        for nq in nq_values:
            model_meta = adapter.metadata()
            for dataset in datasets:
                out_dir = exp_root / tag / nq_tag(nq, adapter.is_tokenizer) / dataset.name
                expected_manifest = build_manifest(dataset, model_meta=model_meta, nq=nq)
                if not force and output_complete(out_dir, expected_manifest):
                    logging.info("Skip reconstruction; manifest matches: %s", out_dir)
                    generated.append(out_dir)
                    continue

                gt_dir = out_dir / "gt_audios"
                syn_dir = out_dir / "syn_audios"
                gt_dir.mkdir(parents=True, exist_ok=True)
                syn_dir.mkdir(parents=True, exist_ok=True)
                logging.info("Reconstructing model=%s nq=%s dataset=%s -> %s", tag, nq, dataset.name, out_dir)

                for item in tqdm(dataset.items, desc=f"{tag}/{nq_tag(nq, adapter.is_tokenizer)}/{dataset.name}"):
                    wav, sample_rate = audio_utils.load_audio(item.audio_path)
                    result = adapter.reconstruct(wav, sample_rate, nq=nq)
                    audio_utils.save_audio(gt_dir / item.output_name, result.reference, result.sample_rate)
                    audio_utils.save_audio(syn_dir / item.output_name, result.audio, result.sample_rate)

                # Re-read metadata after reconstruction because some adapters infer max_nq lazily.
                expected_manifest = build_manifest(dataset, model_meta=adapter.metadata(), nq=nq)
                write_manifest(out_dir, expected_manifest)
                generated.append(out_dir)
    return generated
