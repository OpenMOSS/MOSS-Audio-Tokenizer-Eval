from __future__ import annotations

import torch

from moss_eval import audio as audio_utils
from .base import AudioModelAdapter, ReconstructOutput


class IdentityAdapter(AudioModelAdapter):
    """A deterministic adapter used for smoke tests and dataset validation."""

    is_tokenizer = False

    def __init__(self, config: dict, device: str = "cpu") -> None:
        super().__init__(config, device=device)
        self.sample_rate = config.get("sample_rate")
        self.channels = config.get("channels")

    def reconstruct(self, wav: torch.Tensor, sample_rate: int, nq: int | None = None) -> ReconstructOutput:
        if nq is not None:
            raise ValueError("IdentityAdapter is not a tokenizer and does not accept nq")
        ref = audio_utils.resample_audio(wav, sample_rate, self.sample_rate, device=self.device)
        ref = audio_utils.ensure_channels(ref, self.channels)
        return ReconstructOutput(audio=ref.clone(), reference=ref, sample_rate=int(self.sample_rate or sample_rate))

    def metadata(self) -> dict:
        meta = super().metadata()
        meta.update({"sample_rate": self.sample_rate, "channels": self.channels})
        return meta


class FakeTokenizerAdapter(AudioModelAdapter):
    """Small tokenizer-like adapter for CI; lower nq applies stronger smoothing."""

    is_tokenizer = True

    def __init__(self, config: dict, device: str = "cpu") -> None:
        super().__init__(config, device=device)
        self.sample_rate = int(config.get("sample_rate", 16000))
        self.channels = int(config.get("channels", 1))
        self.max_nq = int(config.get("max_nq", 4))

    def reconstruct(self, wav: torch.Tensor, sample_rate: int, nq: int | None = None) -> ReconstructOutput:
        ref = audio_utils.resample_audio(wav, sample_rate, self.sample_rate, device=self.device)
        ref = audio_utils.ensure_channels(ref, self.channels)
        if nq is None or nq >= self.max_nq:
            out = ref.clone()
        else:
            kernel = max(1, 2 * (self.max_nq - int(nq)) + 1)
            pad = kernel // 2
            x = torch.nn.functional.pad(ref.unsqueeze(0), (pad, pad), mode="reflect")
            weight = torch.ones(ref.shape[0], 1, kernel) / kernel
            out = torch.nn.functional.conv1d(x, weight, groups=ref.shape[0]).squeeze(0)
            out = audio_utils.trim_to_length(out, ref.shape[-1])
        return ReconstructOutput(audio=out, reference=ref, sample_rate=self.sample_rate)

    def metadata(self) -> dict:
        meta = super().metadata()
        meta.update({"sample_rate": self.sample_rate, "channels": self.channels})
        return meta
