from __future__ import annotations

from typing import Iterable, Optional


def parse_nq_spec(spec, max_nq: Optional[int], is_tokenizer: bool) -> list[int | None]:
    if spec is None or spec == "" or spec == "full":
        return [None]
    if not is_tokenizer:
        raise ValueError(f"RVQ layer spec {spec!r} was provided for a non-tokenizer adapter")

    if isinstance(spec, int):
        values = [spec]
    elif isinstance(spec, (list, tuple)):
        values = []
        for item in spec:
            values.extend(parse_nq_spec(item, max_nq=max_nq, is_tokenizer=True))
        return _validate(values, max_nq)
    elif isinstance(spec, str):
        spec = spec.strip().lower()
        if spec == "all":
            if max_nq is None:
                raise ValueError("`nq: all` requires the adapter to expose max_nq or config to set max_nq")
            values = list(range(1, int(max_nq) + 1))
        elif ".." in spec:
            left, right = spec.split("..", 1)
            start = int(left)
            end = int(right)
            step = 1 if end >= start else -1
            values = list(range(start, end + step, step))
        elif "," in spec:
            values = [int(x.strip()) for x in spec.split(",") if x.strip()]
        else:
            values = [int(spec)]
    else:
        raise TypeError(f"Unsupported nq spec type: {type(spec)!r}")
    return _validate(values, max_nq)


def _validate(values: list[int | None], max_nq: Optional[int]) -> list[int | None]:
    seen = set()
    out: list[int | None] = []
    for value in values:
        if value is None:
            out.append(value)
            continue
        value = int(value)
        if value <= 0:
            raise ValueError(f"nq must be positive, got {value}")
        if max_nq is not None and value > int(max_nq):
            raise ValueError(f"nq={value} exceeds max_nq={max_nq}")
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def nq_tag(nq: int | None, is_tokenizer: bool) -> str:
    if nq is None:
        return "full" if is_tokenizer else "default"
    return f"rvq{int(nq)}"
