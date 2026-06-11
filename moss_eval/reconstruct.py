from __future__ import annotations

from pathlib import Path
import logging
import time

from tqdm import tqdm

from moss_eval import audio as audio_utils
from moss_eval.config import config_models
from moss_eval.dataset import discover_datasets
from moss_eval.adapters import build_adapter
from moss_eval.distributed import DistributedContext, log_distributed_context
from moss_eval.manifest import (
    build_manifest,
    output_complete,
    output_files_complete,
    shard_complete,
    write_manifest,
    write_shard_manifest,
)
from moss_eval.rvq import nq_tag, parse_nq_spec


class IncompleteOutputError(RuntimeError):
    pass


def _model_tag(model_cfg: dict, adapter) -> str:
    return str(model_cfg.get("tag") or model_cfg.get("name") or adapter.name).strip().replace(" ", "_")


def wait_for_complete_outputs(
    jobs: list[tuple[Path, dict]],
    *,
    timeout_s: float,
    interval_s: float = 10.0,
) -> None:
    """Wait until every planned output has all gt/syn files, then write full manifests."""

    if timeout_s <= 0:
        incomplete = [str(out_dir) for out_dir, expected in jobs if not output_files_complete(out_dir, expected)]
        if incomplete:
            raise IncompleteOutputError("Outputs are incomplete: " + ", ".join(incomplete[:10]))
        for out_dir, expected in jobs:
            write_manifest(out_dir, expected)
        return

    deadline = time.monotonic() + timeout_s
    while True:
        incomplete = [(out_dir, expected) for out_dir, expected in jobs if not output_files_complete(out_dir, expected)]
        if not incomplete:
            for out_dir, expected in jobs:
                write_manifest(out_dir, expected)
            return
        if time.monotonic() >= deadline:
            sample = ", ".join(str(out_dir) for out_dir, _ in incomplete[:10])
            raise IncompleteOutputError(f"Timed out waiting for {len(incomplete)} incomplete output dirs: {sample}")
        logging.info("Waiting for %d output dirs to complete before metrics/final manifest", len(incomplete))
        time.sleep(interval_s)


def run_reconstruct(
    config: dict,
    *,
    force: bool = False,
    device: str | None = None,
    nq_override=None,
    dist: DistributedContext | None = None,
    wait_timeout_s: float = 0.0,
    wait_interval_s: float = 10.0,
) -> list[Path]:
    exp_root = Path(config.get("exp_root", "exp")).expanduser().resolve()
    datasets = discover_datasets(config, check_exists=True)
    dist = dist or DistributedContext()
    log_distributed_context(dist)

    generated: list[Path] = []
    final_manifest_jobs: list[tuple[Path, dict]] = []

    for model_cfg in config_models(config):
        resolved_device = dist.resolve_device(device or model_cfg.get("device") or config.get("device"))
        adapter = build_adapter(model_cfg, device=resolved_device)
        adapter.load()
        nq_spec = nq_override if nq_override is not None else model_cfg.get("nq", config.get("nq"))
        nq_values = parse_nq_spec(nq_spec, max_nq=adapter.max_nq, is_tokenizer=adapter.is_tokenizer)
        tag = _model_tag(model_cfg, adapter)

        for nq in nq_values:
            model_meta = adapter.metadata()
            for dataset in datasets:
                out_dir = exp_root / tag / nq_tag(nq, adapter.is_tokenizer) / dataset.name
                expected_manifest = build_manifest(dataset, model_meta=model_meta, nq=nq)
                final_manifest_jobs.append((out_dir, expected_manifest))

                shard_items = dist.shard(dataset.items)
                shard_names = [item.output_name for item in shard_items]

                if not force and shard_complete(
                    out_dir,
                    expected_manifest,
                    rank=dist.rank,
                    world_size=dist.world_size,
                    output_names=shard_names,
                ):
                    logging.info("Skip shard; manifest matches: %s (%s)", out_dir, dist.describe())
                    generated.append(out_dir)
                    continue

                gt_dir = out_dir / "gt_audios"
                syn_dir = out_dir / "syn_audios"
                gt_dir.mkdir(parents=True, exist_ok=True)
                syn_dir.mkdir(parents=True, exist_ok=True)
                logging.info(
                    "Reconstructing model=%s nq=%s dataset=%s shard_items=%d/%d -> %s",
                    tag,
                    nq,
                    dataset.name,
                    len(shard_items),
                    len(dataset.items),
                    out_dir,
                )

                if shard_items:
                    desc = f"{tag}/{nq_tag(nq, adapter.is_tokenizer)}/{dataset.name}/rank{dist.rank}"
                    for item in tqdm(shard_items, desc=desc, disable=not dist.is_primary):
                        wav, sample_rate = audio_utils.load_audio(item.audio_path)
                        result = adapter.reconstruct(wav, sample_rate, nq=nq)
                        audio_utils.save_audio(gt_dir / item.output_name, result.reference, result.sample_rate)
                        audio_utils.save_audio(syn_dir / item.output_name, result.audio, result.sample_rate)

                # Re-read metadata after reconstruction because some adapters infer max_nq lazily.
                expected_manifest = build_manifest(dataset, model_meta=adapter.metadata(), nq=nq)
                write_shard_manifest(
                    out_dir,
                    expected_manifest,
                    rank=dist.rank,
                    world_size=dist.world_size,
                    output_names=shard_names,
                )
                final_manifest_jobs[-1] = (out_dir, expected_manifest)
                if not dist.is_distributed and output_files_complete(out_dir, expected_manifest):
                    write_manifest(out_dir, expected_manifest)
                generated.append(out_dir)

    if dist.is_primary and wait_timeout_s is not None:
        try:
            wait_for_complete_outputs(final_manifest_jobs, timeout_s=float(wait_timeout_s), interval_s=float(wait_interval_s))
        except IncompleteOutputError:
            if wait_timeout_s > 0:
                raise
            logging.info("Full outputs are not complete yet; shard manifests were written")

    return generated
