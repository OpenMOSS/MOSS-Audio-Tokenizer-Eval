from __future__ import annotations

from typing import Any

import torch

from moss_eval import audio as audio_utils
from .base import AudioModelAdapter, ReconstructOutput

MOSS_ALIASES = {
    "moss-audio-tokenizer": {
        "repo_id": "OpenMOSS-Team/MOSS-Audio-Tokenizer",
        "sample_rate": 24000,
        "channels": 1,
        "max_nq": 32,
    },
    "moss-audio-tokenizer-v2": {
        "repo_id": "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2",
        "sample_rate": 48000,
        "channels": 2,
        "max_nq": 32,
    },
    "moss-audio-tokenizer-nano": {
        "repo_id": "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
        "sample_rate": 48000,
        "channels": 2,
        "max_nq": 32,
    },
}


def _torch_dtype(name: str | None):
    if not name:
        return None
    table = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return table[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch_dtype: {name}") from exc


class MossAudioTokenizerAdapter(AudioModelAdapter):
    is_tokenizer = True

    def __init__(self, config: dict, device: str = "cpu") -> None:
        merged = dict(config)
        alias = (merged.get("model") or merged.get("adapter") or merged.get("name") or "").lower()
        defaults = MOSS_ALIASES.get(alias, {})
        for key, value in defaults.items():
            merged.setdefault(key, value)
        super().__init__(merged, device=device)
        self.repo_id = merged.get("repo_id")
        if not self.repo_id:
            raise ValueError("MOSS adapter requires `repo_id` or a known model alias")
        self.revision = merged.get("revision")
        self.sample_rate = int(merged.get("sample_rate", 24000))
        self.channels = int(merged.get("channels", 1))
        self.max_nq = int(merged.get("max_nq", 32)) if merged.get("max_nq") is not None else None
        self.chunk_duration = merged.get("chunk_duration")
        self.torch_dtype = _torch_dtype(merged.get("torch_dtype"))
        self.model = None

    def load(self) -> None:
        if self.model is not None:
            return
        from transformers import AutoModel

        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self.revision:
            kwargs["revision"] = self.revision
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype
        self.model = AutoModel.from_pretrained(self.repo_id, **kwargs).to(self.device).eval()
        self.sample_rate = int(getattr(self.model, "sampling_rate", self.sample_rate))
        config = getattr(self.model, "config", None)
        self.channels = int(getattr(config, "number_channels", self.channels))
        for attr in ("num_quantizers", "n_quantizers", "num_codebooks"):
            value = getattr(config, attr, None)
            if value is not None:
                self.max_nq = int(value)
                break

    def _prepare(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        wav = audio_utils.resample_audio(wav, sample_rate, self.sample_rate, device=self.device)
        wav = audio_utils.ensure_channels(wav, self.channels)
        return wav

    @torch.inference_mode()
    def reconstruct(self, wav: torch.Tensor, sample_rate: int, nq: int | None = None) -> ReconstructOutput:
        self.load()
        assert self.model is not None
        ref = self._prepare(wav, sample_rate)
        input_length = ref.shape[-1]
        model_input = ref.unsqueeze(0).to(self.device)
        encode_kwargs = {"return_dict": True}
        decode_kwargs = {"return_dict": True}
        if self.chunk_duration is not None:
            encode_kwargs["chunk_duration"] = self.chunk_duration
            decode_kwargs["chunk_duration"] = self.chunk_duration
        enc = self.model.encode(model_input, **encode_kwargs)
        codes = enc.audio_codes
        if codes.ndim < 3:
            raise RuntimeError(f"Unexpected MOSS audio_codes shape: {tuple(codes.shape)}")
        available_nq = int(codes.shape[0])
        self.max_nq = available_nq
        if nq is not None:
            nq = int(nq)
            if nq <= 0 or nq > available_nq:
                raise ValueError(f"nq must be in [1, {available_nq}], got {nq}")
            codes = codes[:nq]
        dec = self.model.decode(codes, **decode_kwargs)
        audio = dec.audio
        if audio.ndim == 3:
            audio = audio.squeeze(0)
        elif audio.ndim == 2:
            pass
        else:
            raise RuntimeError(f"Unexpected decoded audio shape: {tuple(audio.shape)}")
        audio = audio_utils.trim_to_length(audio.detach().cpu(), input_length)
        return ReconstructOutput(audio=audio, reference=ref.cpu(), sample_rate=self.sample_rate)

    def metadata(self) -> dict:
        meta = super().metadata()
        meta.update({
            "adapter": "moss_audio_tokenizer",
            "repo_id": self.repo_id,
            "revision": self.revision,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "frame_rate": 12.5,
            "codebook_bits": 10,
            "chunk_duration": self.chunk_duration,
        })
        return meta
