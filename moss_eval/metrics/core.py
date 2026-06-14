from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import logging
import os
import shlex
import subprocess
import tempfile
from typing import Callable

import numpy as np
import torch
import librosa
from tqdm import tqdm

from moss_eval import audio as audio_utils

DEFAULT_METRICS = ["stoi", "pesq_nb", "pesq_wb", "mel_loss", "spectral_convergence", "sdr", "sisdr", "stft"]
DEFAULT_EXTERNAL_SCRIPT = "do_reconstruct_evalation_given_dir.py"


@dataclass
class Pair:
    name: str
    ref_path: Path
    syn_path: Path


def collect_pairs(output_dir: str | Path) -> list[Pair]:
    output_dir = Path(output_dir)
    ref_dir = output_dir / "gt_audios"
    syn_dir = output_dir / "syn_audios"
    if not ref_dir.is_dir() or not syn_dir.is_dir():
        raise FileNotFoundError(f"Expected {ref_dir} and {syn_dir}")
    ref_files = {p.name: p for p in sorted(ref_dir.glob("*")) if p.is_file()}
    syn_files = {p.name: p for p in sorted(syn_dir.glob("*")) if p.is_file()}
    if ref_files.keys() != syn_files.keys():
        missing_syn = sorted(ref_files.keys() - syn_files.keys())
        missing_ref = sorted(syn_files.keys() - ref_files.keys())
        raise ValueError(f"Mismatched audio files. missing_syn={missing_syn[:5]}, missing_ref={missing_ref[:5]}")
    if not ref_files:
        raise ValueError(f"No audio files found in {ref_dir}")
    return [Pair(name, ref_files[name], syn_files[name]) for name in sorted(ref_files)]


def _read_pair(pair: Pair, target_sr: int | None, device: str) -> tuple[np.ndarray, np.ndarray, int]:
    ref, ref_sr = audio_utils.load_audio(pair.ref_path)
    syn, syn_sr = audio_utils.load_audio(pair.syn_path)
    if target_sr is None:
        target_sr = ref_sr
    ref = audio_utils.resample_audio(ref, ref_sr, target_sr, device=device)
    syn = audio_utils.resample_audio(syn, syn_sr, target_sr, device=device)
    ref_np = audio_utils.to_mono_numpy(ref)
    syn_np = audio_utils.to_mono_numpy(syn)
    length = min(ref_np.shape[-1], syn_np.shape[-1])
    if length <= 0:
        raise ValueError(f"Empty aligned audio for {pair.name}")
    return ref_np[:length], syn_np[:length], int(target_sr)


def _mean_or_raise(values: list[float], metric: str) -> float:
    if not values:
        raise RuntimeError(f"All files failed for metric {metric}")
    return float(np.mean(np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)))


def metric_stoi(pairs: list[Pair], device: str, target_sr: int = 16000) -> tuple[float, dict, list[str]]:
    try:
        from pystoi.stoi import stoi
    except ImportError as exc:
        raise RuntimeError("STOI requires `pystoi`. Install it or disable the stoi metric.") from exc
    values, per_file, errors = [], {}, []
    for pair in tqdm(pairs, desc="stoi"):
        try:
            ref, syn, sr = _read_pair(pair, target_sr, device)
            value = float(stoi(ref, syn, sr, extended=False))
            values.append(value)
            per_file[pair.name] = value
        except Exception as exc:
            errors.append(f"{pair.name}: {exc}")
    return _mean_or_raise(values, "stoi"), per_file, errors


def _metric_pesq(pairs: list[Pair], device: str, mode: str) -> tuple[float, dict, list[str]]:
    try:
        from pesq import pesq
    except ImportError as exc:
        raise RuntimeError("PESQ requires `pesq`. Install it or disable pesq metrics.") from exc
    target_sr = 16000
    key = f"pesq-{mode}"
    values, per_file, errors = [], {}, []
    for pair in tqdm(pairs, desc=key):
        try:
            ref, syn, sr = _read_pair(pair, target_sr, device)
            value = float(pesq(sr, ref, syn, mode))
            values.append(value)
            per_file[pair.name] = value
        except Exception as exc:
            errors.append(f"{pair.name}: {exc}")
    return _mean_or_raise(values, key), per_file, errors


def metric_pesq_nb(pairs: list[Pair], device: str) -> tuple[float, dict, list[str]]:
    return _metric_pesq(pairs, device, "nb")


def metric_pesq_wb(pairs: list[Pair], device: str) -> tuple[float, dict, list[str]]:
    return _metric_pesq(pairs, device, "wb")


def metric_spectral_convergence(pairs: list[Pair], device: str, target_sr: int = 16000) -> tuple[float, dict, list[str]]:
    values, per_file, errors = [], {}, []
    for pair in tqdm(pairs, desc="spectral_convergence"):
        try:
            ref, syn, _ = _read_pair(pair, target_sr, device)
            ref_mag = np.abs(librosa.stft(ref))
            syn_mag = np.abs(librosa.stft(syn))
            value = float(np.linalg.norm(ref_mag - syn_mag) / (np.linalg.norm(ref_mag) + 1e-12))
            values.append(value)
            per_file[pair.name] = value
        except Exception as exc:
            errors.append(f"{pair.name}: {exc}")
    return _mean_or_raise(values, "spectral_convergence"), per_file, errors


def _sisdr_value(ref: np.ndarray, syn: np.ndarray) -> float:
    ref = ref / (np.linalg.norm(ref) + 1e-10)
    syn = syn / (np.linalg.norm(syn) + 1e-10)
    projection = np.dot(ref, syn) * ref
    distortion = syn - projection
    return float(10 * np.log10(np.mean(ref ** 2) / (np.mean(distortion ** 2) + 1e-10)))


def metric_sisdr(pairs: list[Pair], device: str, target_sr: int = 16000) -> tuple[float, dict, list[str]]:
    values, per_file, errors = [], {}, []
    for pair in tqdm(pairs, desc="sisdr"):
        try:
            ref, syn, _ = _read_pair(pair, target_sr, device)
            value = _sisdr_value(ref, syn)
            values.append(value)
            per_file[pair.name] = value
        except Exception as exc:
            errors.append(f"{pair.name}: {exc}")
    return _mean_or_raise(values, "sisdr"), per_file, errors


def _plain_sdr_value(ref: np.ndarray, syn: np.ndarray) -> float:
    noise = ref - syn
    return float(10 * np.log10((np.sum(ref ** 2) + 1e-12) / (np.sum(noise ** 2) + 1e-12)))


def metric_sdr(pairs: list[Pair], device: str, target_sr: int = 16000) -> tuple[float, dict, list[str]]:
    try:
        from mir_eval.separation import bss_eval_sources
    except ImportError:
        bss_eval_sources = None
        logging.warning("mir_eval is not installed; falling back to plain waveform SDR")
    values, per_file, errors = [], {}, []
    for pair in tqdm(pairs, desc="sdr"):
        try:
            ref, syn, _ = _read_pair(pair, target_sr, device)
            if bss_eval_sources is None:
                value = _plain_sdr_value(ref, syn)
            else:
                value = float(bss_eval_sources(np.asarray([ref]), np.asarray([syn]))[0][0])
            values.append(value)
            per_file[pair.name] = value
        except Exception as exc:
            errors.append(f"{pair.name}: {exc}")
    return _mean_or_raise(values, "sdr"), per_file, errors


def metric_mel_loss(pairs: list[Pair], device: str, target_sr: int = 16000) -> tuple[float, dict, list[str]]:
    run_device = audio_utils.choose_device(device)
    values, per_file, errors = [], {}, []
    n_mels = [150, 80]
    window_lengths = [2048, 512]
    loss_fn = torch.nn.L1Loss()
    clamp_eps = 1e-5
    pow_value = 2.0
    for pair in tqdm(pairs, desc="mel_loss"):
        try:
            ref, syn, _ = _read_pair(pair, target_sr, run_device)
            ref_t = torch.from_numpy(ref).float().to(run_device).unsqueeze(0)
            syn_t = torch.from_numpy(syn).float().to(run_device).unsqueeze(0)
            loss = torch.zeros((), device=run_device)
            for n_mel, window_length in zip(n_mels, window_lengths):
                hop = window_length // 4
                window = torch.sqrt(
                    torch.hann_window(window_length, periodic=True, device=run_device)
                )
                ref_stft = torch.stft(
                    ref_t,
                    n_fft=window_length,
                    hop_length=hop,
                    window=window,
                    return_complex=True,
                    center=True,
                )
                syn_stft = torch.stft(
                    syn_t,
                    n_fft=window_length,
                    hop_length=hop,
                    window=window,
                    return_complex=True,
                    center=True,
                )
                ref_mag = ref_stft.abs()
                syn_mag = syn_stft.abs()
                mel = librosa.filters.mel(
                    sr=target_sr,
                    n_fft=window_length,
                    n_mels=n_mel,
                    fmin=0.0,
                    fmax=target_sr / 2,
                )
                mel_t = torch.from_numpy(mel).float().to(run_device)
                ref_mel = (
                    torch.matmul(ref_mag.transpose(1, 2), mel_t.T)
                    .transpose(1, 2)
                    .unsqueeze(1)
                )
                syn_mel = (
                    torch.matmul(syn_mag.transpose(1, 2), mel_t.T)
                    .transpose(1, 2)
                    .unsqueeze(1)
                )
                loss = loss + loss_fn(
                    ref_mel.clamp(clamp_eps).pow(pow_value).log10(),
                    syn_mel.clamp(clamp_eps).pow(pow_value).log10(),
                )
                loss = loss + loss_fn(ref_mel, syn_mel)
            value = float(loss.detach().cpu())
            values.append(value)
            per_file[pair.name] = value
        except Exception as exc:
            errors.append(f"{pair.name}: {exc}")
    return _mean_or_raise(values, "mel_loss"), per_file, errors


def metric_speaker_similarity(
    pairs: list[Pair],
    device: str,
    model_path: str | None = None,
    target_sr: int = 16000,
) -> tuple[float, dict, list[str]]:
    if not model_path:
        raise RuntimeError("speaker_similarity requires `model_path` in metric config")
    from stopes.eval.vocal_style_similarity.vocal_style_sim_tool import get_embedder, compute_cosine_similarity

    names, ref_paths, syn_paths, errors = [], [], [], []
    with tempfile.TemporaryDirectory(prefix="moss_eval_sim_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        for idx, pair in enumerate(pairs):
            try:
                ref, syn, sr = _read_pair(pair, target_sr, device)
                ref_path = tmp_dir_path / f"{idx:08d}_ref.wav"
                syn_path = tmp_dir_path / f"{idx:08d}_syn.wav"
                audio_utils.save_audio(ref_path, torch.from_numpy(ref).unsqueeze(0), sr)
                audio_utils.save_audio(syn_path, torch.from_numpy(syn).unsqueeze(0), sr)
                names.append(pair.name)
                ref_paths.append(str(ref_path))
                syn_paths.append(str(syn_path))
            except Exception as exc:
                errors.append(f"{pair.name}: {exc}")

        if not ref_paths:
            raise RuntimeError("All files failed for metric sim")

        embedder = get_embedder(model_name="valle", model_path=model_path)
        sims = compute_cosine_similarity(embedder(ref_paths), embedder(syn_paths))
        per_file = {name: float(value) for name, value in zip(names, sims)}
        return float(np.mean(sims)), per_file, errors


def _infer_dataset_output_dir(pairs: list[Pair]) -> Path:
    ref_dirs = {pair.ref_path.parent for pair in pairs}
    syn_dirs = {pair.syn_path.parent for pair in pairs}
    if len(ref_dirs) != 1 or len(syn_dirs) != 1:
        raise ValueError("External metrics require all pairs to come from one output_dir")
    ref_dir = next(iter(ref_dirs))
    syn_dir = next(iter(syn_dirs))
    dataset_output_dir = ref_dir.parent
    if ref_dir.name != "gt_audios" or syn_dir.name != "syn_audios" or syn_dir.parent != dataset_output_dir:
        raise ValueError("External metrics expect output_dir/gt_audios and output_dir/syn_audios")
    return dataset_output_dir.expanduser().resolve()


def _resolve_python_bin(python_bin: str | None, env_dir: str | None) -> str:
    if python_bin:
        path = Path(python_bin).expanduser()
        return str(path) if path.parts else python_bin
    if env_dir:
        candidate = Path(env_dir).expanduser() / "bin" / "python"
        if candidate.is_file():
            return str(candidate)
    return "python"


def _format_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _tail(text: str | None, limit: int = 4000) -> str:
    if not text:
        return ""
    return text[-limit:]


def _prepare_external_metric_audio(
    pairs: list[Pair],
    output_dir: Path,
    *,
    target_sr: int,
    channels: int,
    device: str,
) -> None:
    gt_dir = output_dir / "gt_audios"
    syn_dir = output_dir / "syn_audios"
    gt_dir.mkdir(parents=True, exist_ok=True)
    syn_dir.mkdir(parents=True, exist_ok=True)
    for pair in pairs:
        for src_path, dst_dir in ((pair.ref_path, gt_dir), (pair.syn_path, syn_dir)):
            wav, sr = audio_utils.load_audio(src_path)
            wav = audio_utils.resample_audio(wav, sr, target_sr, device=device)
            wav = audio_utils.ensure_channels(wav, channels)
            audio_utils.save_audio(dst_dir / pair.name, wav, target_sr)


def _run_external_dataset_metric(
    pairs: list[Pair],
    metric_key: str,
    *,
    device: str,
    work_dir: str | None,
    env_dir: str | None,
    python_bin: str | None,
    script: str,
    script_args: dict[str, str | int | float | Path] | None = None,
    extra_args: list[str] | None = None,
    env_vars: dict[str, str] | None = None,
    preprocess_sample_rate: int | None = None,
    preprocess_channels: int | None = None,
    timeout: float | None = None,
) -> tuple[float, dict, list[str]]:
    source_output_dir = _infer_dataset_output_dir(pairs)
    dataset_name = source_output_dir.name

    work_path = Path(work_dir or ".").expanduser().resolve()
    if not work_path.is_dir():
        raise FileNotFoundError(f"{metric_key} work_dir not found: {work_path}")

    script_path = Path(script).expanduser()
    if not script_path.is_absolute():
        script_path = work_path / script_path
    if not script_path.is_file():
        raise FileNotFoundError(f"{metric_key} script not found: {script_path}")

    tmp_ctx = None
    try:
        dataset_output_dir = source_output_dir
        if preprocess_sample_rate is not None or preprocess_channels is not None:
            target_sr = int(preprocess_sample_rate or 24000)
            channels = int(preprocess_channels or 1)
            tmp_ctx = tempfile.TemporaryDirectory(prefix=f"moss_eval_{metric_key}_")
            dataset_output_dir = Path(tmp_ctx.name) / dataset_name
            logging.info(
                "Preparing external metric audio metric=%s sr=%s channels=%s -> %s",
                metric_key,
                target_sr,
                channels,
                dataset_output_dir,
            )
            _prepare_external_metric_audio(
                pairs,
                dataset_output_dir,
                target_sr=target_sr,
                channels=channels,
                device=device,
            )

        result_path = dataset_output_dir.parent / "results.json"
        cmd = [_resolve_python_bin(python_bin, env_dir), str(script_path), "--output_dir", str(dataset_output_dir)]
        for key, value in (script_args or {}).items():
            cmd.extend([f"--{key}", str(value)])
        if extra_args:
            cmd.extend(str(arg) for arg in extra_args)

        env = os.environ.copy()
        env["PYTHONPATH"] = str(work_path) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        if env_vars:
            env.update({str(key): str(value) for key, value in env_vars.items()})
        logging.info("Running external metric %s: %s", metric_key, _format_command(cmd))
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(work_path),
                env=env,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"External metric {metric_key} failed with exit code {exc.returncode}: {_format_command(cmd)}\n"
                f"stdout tail:\n{_tail(exc.stdout)}\n"
                f"stderr tail:\n{_tail(exc.stderr)}"
            ) from exc
        if completed.stdout:
            logging.debug("%s stdout tail:\n%s", metric_key, _tail(completed.stdout))
        if completed.stderr:
            logging.debug("%s stderr tail:\n%s", metric_key, _tail(completed.stderr))

        if not result_path.is_file():
            raise RuntimeError(f"External metric {metric_key} did not create {result_path}")
        with result_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if dataset_name not in payload or metric_key not in payload[dataset_name]:
            raise RuntimeError(f"External metric {metric_key} missing from {result_path} under dataset {dataset_name}")
        return float(payload[dataset_name][metric_key]), {}, []
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


def _external_preprocess_options(
    preprocess_sample_rate: int | None,
    preprocess_channels: int | None,
) -> tuple[int | None, int | None]:
    if preprocess_sample_rate is False or preprocess_channels is False:
        return None, None
    return preprocess_sample_rate, preprocess_channels


def metric_utmos(
    pairs: list[Pair],
    device: str,
    python_bin: str | None = None,
    env_dir: str | None = None,
    conda_env: str | None = None,
    work_dir: str | None = None,
    script: str = DEFAULT_EXTERNAL_SCRIPT,
    ckpt_path: str | None = None,
    bs: int | None = None,
    num_workers: int | None = None,
    preprocess_sample_rate: int | None = 24000,
    preprocess_channels: int | None = 1,
    extra_args: list[str] | None = None,
    timeout: float | None = None,
) -> tuple[float, dict, list[str]]:
    if not work_dir:
        raise RuntimeError("utmos requires `work_dir` in metric options")
    script_args: dict[str, str | int | float | Path] = {}
    if ckpt_path is not None:
        script_args["ckpt_path"] = ckpt_path
    if bs is not None:
        script_args["bs"] = bs
    if num_workers is not None:
        script_args["num_workers"] = num_workers
    preprocess_sample_rate, preprocess_channels = _external_preprocess_options(
        preprocess_sample_rate,
        preprocess_channels,
    )
    return _run_external_dataset_metric(
        pairs,
        "utmos",
        device=device,
        work_dir=work_dir,
        env_dir=env_dir or conda_env,
        python_bin=python_bin,
        script=script,
        script_args=script_args,
        extra_args=extra_args,
        env_vars={"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"},
        preprocess_sample_rate=preprocess_sample_rate,
        preprocess_channels=preprocess_channels,
        timeout=timeout,
    )


def _multi_scale_stft_loss(
    ref: np.ndarray,
    syn: np.ndarray,
    *,
    device: str,
    window_lengths: tuple[int, ...] = (2048, 512),
    clamp_eps: float = 1e-5,
    pow_value: float = 2.0,
) -> float:
    run_device = audio_utils.choose_device(device)
    ref_t = torch.from_numpy(ref).float().to(run_device).unsqueeze(0)
    syn_t = torch.from_numpy(syn).float().to(run_device).unsqueeze(0)
    loss = torch.zeros((), device=run_device)
    loss_fn = torch.nn.L1Loss()
    for window_length in window_lengths:
        hop_length = window_length // 4
        # Match audiotools AudioSignal default: scipy periodic Hann, square-rooted.
        window = torch.sqrt(torch.hann_window(window_length, periodic=True, device=run_device))
        ref_stft = torch.stft(
            ref_t,
            n_fft=window_length,
            hop_length=hop_length,
            window=window,
            return_complex=True,
            center=True,
        )
        syn_stft = torch.stft(
            syn_t,
            n_fft=window_length,
            hop_length=hop_length,
            window=window,
            return_complex=True,
            center=True,
        )
        ref_mag = ref_stft.abs()
        syn_mag = syn_stft.abs()
        loss = loss + loss_fn(
            ref_mag.clamp(clamp_eps).pow(pow_value).log10(),
            syn_mag.clamp(clamp_eps).pow(pow_value).log10(),
        )
        loss = loss + loss_fn(ref_mag, syn_mag)
    return float(loss.detach().cpu())


def metric_stft(
    pairs: list[Pair],
    device: str,
    target_sr: int = 16000,
    window_lengths: list[int] | tuple[int, ...] = (2048, 512),
) -> tuple[float, dict, list[str]]:
    values, per_file, errors = [], {}, []
    window_lengths_tuple = tuple(int(w) for w in window_lengths)
    for pair in tqdm(pairs, desc="stft"):
        try:
            ref, syn, _ = _read_pair(pair, target_sr, device)
            value = _multi_scale_stft_loss(
                ref,
                syn,
                device=device,
                window_lengths=window_lengths_tuple,
            )
            values.append(value)
            per_file[pair.name] = value
        except Exception as exc:
            errors.append(f"{pair.name}: {exc}")
    return _mean_or_raise(values, "stft"), per_file, errors


METRIC_FUNCS: dict[str, Callable] = {
    "stoi": metric_stoi,
    "pesq_nb": metric_pesq_nb,
    "pesq-nb": metric_pesq_nb,
    "pesq_wb": metric_pesq_wb,
    "pesq-wb": metric_pesq_wb,
    "mel_loss": metric_mel_loss,
    "spectral_convergence": metric_spectral_convergence,
    "sdr": metric_sdr,
    "sisdr": metric_sisdr,
    "speaker_similarity": metric_speaker_similarity,
    "sim": metric_speaker_similarity,
    "utmos": metric_utmos,
    "stft": metric_stft,
}

RESULT_KEY = {
    "pesq_nb": "pesq-nb",
    "pesq-nb": "pesq-nb",
    "pesq_wb": "pesq-wb",
    "pesq-wb": "pesq-wb",
    "speaker_similarity": "sim",
    "sim": "sim",
}


def evaluate_output_dir(output_dir: str | Path, metrics: list[str] | None = None, *, device: str = "cpu", metric_options: dict | None = None) -> tuple[dict, dict, dict]:
    pairs = collect_pairs(output_dir)
    metric_options = metric_options or {}
    enabled = metrics or DEFAULT_METRICS
    results: dict[str, float] = {}
    per_file: dict[str, dict] = {}
    errors: dict[str, list[str]] = {}
    for metric in enabled:
        metric = metric.strip()
        if not metric:
            continue
        func = METRIC_FUNCS.get(metric)
        if func is None:
            raise ValueError(f"Unknown metric: {metric}. Available: {sorted(METRIC_FUNCS)}")
        key = RESULT_KEY.get(metric, metric)
        options = dict(metric_options.get(metric, {}))
        logging.info("Evaluating metric=%s output_dir=%s", key, output_dir)
        mean_value, metric_per_file, metric_errors = func(pairs, device=device, **options)
        results[key] = mean_value
        per_file[key] = metric_per_file
        if metric_errors:
            errors[key] = metric_errors
            logging.warning("Metric %s had %d file errors", key, len(metric_errors))
    return results, per_file, errors


def _infer_bps_from_manifest(output_dir: Path) -> int | None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    model = manifest.get("model") or {}
    nq = manifest.get("nq")
    if nq is None:
        return None

    if model.get("bps") is not None:
        return int(round(float(model["bps"])))
    if model.get("bps_per_quantizer") is not None:
        return int(round(float(model["bps_per_quantizer"]) * int(nq)))

    frame_rate = model.get("frame_rate")
    codebook_bits = model.get("codebook_bits")
    if frame_rate is not None and codebook_bits is not None:
        return int(round(float(frame_rate) * float(codebook_bits) * int(nq)))

    repo_id = str(model.get("repo_id") or "")
    adapter = str(model.get("adapter") or "")
    if adapter == "moss_audio_tokenizer" or "MOSS-Audio-Tokenizer" in repo_id:
        return int(round(12.5 * 10 * int(nq)))
    return None


def write_results(output_dir: str | Path, results: dict, *, metric_errors: dict | None = None, metadata: dict | None = None) -> Path:
    output_dir = Path(output_dir)
    dataset_name = output_dir.name
    result_path = output_dir.parent / "results.json"
    if result_path.is_file():
        with result_path.open("r", encoding="utf-8") as f:
            all_results = json.load(f)
    else:
        all_results = {}
    all_results.setdefault(dataset_name, {})
    all_results[dataset_name].update({k: float(v) for k, v in results.items()})
    if metric_errors:
        all_results[dataset_name].setdefault("_metric_errors", {}).update(metric_errors)
    inferred_bps = _infer_bps_from_manifest(output_dir)
    if inferred_bps is not None:
        all_results.setdefault("_meta", {})["bps"] = inferred_bps
    if metadata:
        all_results.setdefault("_meta", {}).update(metadata)
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, sort_keys=True)
    return result_path
