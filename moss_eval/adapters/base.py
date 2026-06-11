from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ReconstructOutput:
    audio: torch.Tensor
    reference: torch.Tensor
    sample_rate: int
    extra: dict | None = None


class AudioModelAdapter:
    """Base class for codec/tokenizer/VAE/vocoder reconstruction adapters.

    Subclasses receive raw audio as a [C, T] float tensor and must return both the
    reconstructed audio and the reference audio that should be used for metrics.
    Keeping the reference in the adapter is important because each model defines
    its own fair input policy, e.g. 24 kHz mono or 48 kHz stereo.
    """

    is_tokenizer: bool = False
    max_nq: Optional[int] = None

    def __init__(self, config: dict, device: str = "cpu") -> None:
        self.config = dict(config)
        self.device = device
        self.name = self.config.get("name") or self.config.get("adapter") or self.__class__.__name__

    def load(self) -> None:
        return None

    def reconstruct(self, wav: torch.Tensor, sample_rate: int, nq: int | None = None) -> ReconstructOutput:
        raise NotImplementedError

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "adapter": self.config.get("adapter", self.__class__.__name__),
            "is_tokenizer": self.is_tokenizer,
            "max_nq": self.max_nq,
        }
