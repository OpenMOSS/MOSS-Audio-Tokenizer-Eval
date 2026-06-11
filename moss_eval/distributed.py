from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class DistributedContext:
    rank: int = 0
    world_size: int = 1
    local_rank: int | None = None
    source: str = "none"

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def shard(self, items: Sequence[T]) -> list[T]:
        if self.world_size <= 1:
            return list(items)
        return [item for i, item in enumerate(items) if i % self.world_size == self.rank]

    def shard_indices(self, n_items: int) -> list[int]:
        if self.world_size <= 1:
            return list(range(n_items))
        return [i for i in range(n_items) if i % self.world_size == self.rank]

    def describe(self) -> str:
        if not self.is_distributed:
            return "single-process"
        local = "" if self.local_rank is None else f", local_rank={self.local_rank}"
        return f"rank={self.rank}, world_size={self.world_size}{local}, source={self.source}"

    def resolve_device(self, requested: str | None) -> str | None:
        if requested in {None, "", "auto"}:
            if self.local_rank is not None:
                return f"cuda:{self.local_rank}"
            return requested
        if requested == "cuda" and self.local_rank is not None:
            return f"cuda:{self.local_rank}"
        return requested


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return int(value)


def _validate(rank: int, world_size: int, local_rank: int | None, source: str) -> DistributedContext:
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    return DistributedContext(rank=rank, world_size=world_size, local_rank=local_rank, source=source)


def distributed_context(
    mode: str = "auto",
    *,
    num_shards: int | None = None,
    shard_index: int | None = None,
    local_rank: int | None = None,
) -> DistributedContext:
    """Build a sharding context from explicit args, torchrun env, or Slurm env.

    This intentionally avoids initializing torch.distributed. Reconstruction is an
    embarrassingly parallel workload: each rank writes a disjoint set of files.
    """

    mode = (mode or "auto").lower()
    if mode not in {"auto", "none", "manual", "torchrun", "slurm"}:
        raise ValueError(f"Unsupported distributed mode: {mode}")

    if mode == "none":
        return DistributedContext()

    if num_shards is not None or shard_index is not None:
        if num_shards is None or shard_index is None:
            raise ValueError("--num-shards and --shard-index must be provided together")
        return _validate(int(shard_index), int(num_shards), local_rank, "manual")

    if mode in {"auto", "torchrun"}:
        rank = _env_int("RANK")
        world = _env_int("WORLD_SIZE")
        env_local_rank = _env_int("LOCAL_RANK")
        if rank is not None and world is not None:
            return _validate(rank, world, local_rank if local_rank is not None else env_local_rank, "torchrun")
        if mode == "torchrun":
            raise ValueError("torchrun mode requires RANK and WORLD_SIZE environment variables")

    if mode in {"auto", "slurm"}:
        rank = _env_int("SLURM_PROCID")
        world = _env_int("SLURM_NTASKS")
        env_local_rank = _env_int("SLURM_LOCALID")
        if rank is not None and world is not None:
            return _validate(rank, world, local_rank if local_rank is not None else env_local_rank, "slurm")
        if mode == "slurm":
            raise ValueError("slurm mode requires SLURM_PROCID and SLURM_NTASKS environment variables")

    return DistributedContext()


def log_distributed_context(ctx: DistributedContext) -> None:
    logging.info("Distributed context: %s", ctx.describe())
