from __future__ import annotations

from pathlib import Path
import json
import os
from typing import Any


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


def load_config(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to read YAML configs. Install pyyaml or use JSON config.") from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    data = _expand(data)
    data.setdefault("config_path", str(path))
    return data


def config_models(config: dict) -> list[dict]:
    models = config.get("models")
    if models is None:
        model = config.get("model")
        if model is None:
            raise ValueError("Config must define `models` or `model`")
        models = [model]
    if isinstance(models, dict):
        models = [models]
    out = []
    for model in models:
        if isinstance(model, str):
            out.append({"name": model, "adapter": model})
        elif isinstance(model, dict):
            out.append(dict(model))
        else:
            raise TypeError(f"Model config must be a string or mapping, got {type(model)!r}")
    return out


def metric_config(config: dict) -> dict:
    metrics = config.get("metrics", {})
    if metrics is None:
        return {}
    if isinstance(metrics, list):
        return {"enabled": metrics}
    if isinstance(metrics, str):
        return {"enabled": [m.strip() for m in metrics.split(",") if m.strip()]}
    if not isinstance(metrics, dict):
        raise TypeError("`metrics` must be a mapping, list, string, or null")
    return metrics
