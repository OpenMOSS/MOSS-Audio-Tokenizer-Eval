from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F


def choose_device(device: str | None) -> str:
    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    data, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    # soundfile returns [T, C], internal convention is [C, T].
    wav = torch.from_numpy(data.T.copy())
    return wav, int(sample_rate)


def resample_audio(wav: torch.Tensor, orig_sr: int, target_sr: Optional[int], device: str = "cpu") -> torch.Tensor:
    if target_sr is None or int(orig_sr) == int(target_sr):
        return wav.float()
    run_device = choose_device(device)
    wav_device = wav.float().to(run_device)
    out = F.resample(wav_device, orig_freq=int(orig_sr), new_freq=int(target_sr))
    return out.cpu()


def ensure_channels(wav: torch.Tensor, channels: Optional[int]) -> torch.Tensor:
    if channels is None:
        return wav
    channels = int(channels)
    if channels <= 0:
        raise ValueError(f"channels must be positive, got {channels}")
    if wav.ndim != 2:
        raise ValueError(f"audio tensor must have shape [C, T], got {tuple(wav.shape)}")
    current = wav.shape[0]
    if current == channels:
        return wav
    if channels == 1:
        return wav.mean(dim=0, keepdim=True)
    if current == 1:
        return wav.repeat(channels, 1)
    if current > channels:
        return wav[:channels]
    repeats = (channels + current - 1) // current
    return wav.repeat(repeats, 1)[:channels]


def trim_to_length(wav: torch.Tensor, length: int) -> torch.Tensor:
    if wav.shape[-1] >= length:
        return wav[..., :length]
    pad = length - wav.shape[-1]
    return torch.nn.functional.pad(wav, (0, pad))


def save_audio(path: str | Path, wav: torch.Tensor, sample_rate: int, subtype: str = "PCM_16") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.ndim != 2:
        raise ValueError(f"audio tensor must have shape [C, T], got {tuple(wav.shape)}")
    wav = wav.detach().cpu().float().clamp(-1.0, 1.0)
    data = wav.transpose(0, 1).numpy()
    sf.write(str(path), data, int(sample_rate), subtype=subtype)


def to_mono_numpy(wav: torch.Tensor) -> np.ndarray:
    if wav.ndim == 2:
        wav = wav.mean(dim=0)
    return wav.detach().cpu().float().numpy()
