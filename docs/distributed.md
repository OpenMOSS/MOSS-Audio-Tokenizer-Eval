# 多机器多卡评测

本仓库使用 deterministic sharding 支持多机器多卡：每个进程只处理
`sample_index % world_size == rank` 的样本。模型参数不需要同步，因此不使用 DDP；
这比把评测推理包进 `DistributedDataParallel` 更简单、更稳定。

## 推荐：torchrun

单机 8 卡：

```bash
torchrun --nproc_per_node=8 -m moss_eval.cli run \
  --config configs/examples/moss_audio_tokenizer.yaml \
  --device cuda \
  --distributed torchrun
```

多机示例：

```bash
torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  -m moss_eval.cli run \
  --config configs/examples/moss_audio_tokenizer.yaml \
  --device cuda \
  --distributed torchrun
```

`--device cuda` 会自动映射成 `cuda:${LOCAL_RANK}`。

## Slurm / srun

如果环境里有 `SLURM_PROCID`、`SLURM_NTASKS`、`SLURM_LOCALID`，可以用：

```bash
srun python -m moss_eval.cli run \
  --config configs/examples/moss_audio_tokenizer.yaml \
  --device cuda \
  --distributed slurm
```

`--distributed auto` 也会自动识别 torchrun 或 Slurm 环境。

## 手动分片

不依赖 torchrun/Slurm 时，可以手动启动多个进程：

```bash
python -m moss_eval.cli reconstruct --config cfg.yaml --num-shards 4 --shard-index 0 --device cuda:0 &
python -m moss_eval.cli reconstruct --config cfg.yaml --num-shards 4 --shard-index 1 --device cuda:1 &
python -m moss_eval.cli reconstruct --config cfg.yaml --num-shards 4 --shard-index 2 --device cuda:2 &
python -m moss_eval.cli reconstruct --config cfg.yaml --num-shards 4 --shard-index 3 --device cuda:3 &
wait
python -m moss_eval.cli metrics --output-dir exp/model/rvq8/dataset
```

## `run` 与 `reconstruct` 的区别

- `run`：所有 rank 并行重建；rank0 默认等待所有 shard 输出齐全后写完整
  `manifest.json` 并运行 metrics；非 rank0 只重建自己的 shard。
- `reconstruct`：只做重建。默认不等待其他 shard。如果希望 rank0 等待并写完整
  manifest，可以加 `--wait-timeout 86400`。

## 输出文件

每个 rank 会写自己的 shard manifest：

```text
manifest.rank00000-of-00008.json
manifest.rank00001-of-00008.json
...
```

当所有 `gt_audios/` 和 `syn_audios/` 文件齐全后，rank0 会写完整的
`manifest.json`。后续重复运行时，框架会用完整 manifest 或 shard manifest 跳过已完成
工作。
