from __future__ import annotations

import importlib

from moss_eval.audio import choose_device
from .fake import FakeTokenizerAdapter, IdentityAdapter


def _load_custom(path: str):
    module_name, _, class_name = path.partition(":")
    if not module_name or not class_name:
        raise ValueError("Custom adapter must be specified as `module:ClassName`")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def build_adapter(model_config: dict, device: str | None = None):
    cfg = dict(model_config)
    run_device = choose_device(device or cfg.get("device"))
    adapter_name = (cfg.get("adapter") or cfg.get("type") or cfg.get("name") or "").strip()
    adapter_key = adapter_name.lower()

    if adapter_key in {"identity", "copy", "fake_identity"}:
        return IdentityAdapter(cfg, device=run_device)
    if adapter_key in {"fake_tokenizer", "toy_tokenizer"}:
        return FakeTokenizerAdapter(cfg, device=run_device)
    if adapter_key.startswith("moss") or adapter_key in {
        "moss-audio-tokenizer",
        "moss-audio-tokenizer-v2",
        "moss-audio-tokenizer-nano",
    } or str(cfg.get("repo_id", "")).startswith("OpenMOSS-Team/MOSS-Audio-Tokenizer"):
        from .moss import MossAudioTokenizerAdapter
        return MossAudioTokenizerAdapter(cfg, device=run_device)
    if ":" in adapter_name:
        cls = _load_custom(adapter_name)
        return cls(cfg, device=run_device)
    raise ValueError(
        f"Unknown adapter {adapter_name!r}. Use one of identity, fake_tokenizer, "
        "moss-audio-tokenizer, moss-audio-tokenizer-v2, moss-audio-tokenizer-nano, "
        "or a custom `module:ClassName`."
    )
