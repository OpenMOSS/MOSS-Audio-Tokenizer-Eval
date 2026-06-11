# 新评测 CLI 快速开始

这套新入口以单 Conda 环境为主，旧的多环境 bash 暂时保留做兼容。推荐新实验都走
`moss-eval`。

## 安装

```bash
conda env create -f environment.yml
conda activate moss-audio-eval
```

如果你已经有合适的 PyTorch 环境，也可以只安装评测依赖：

```bash
pip install -r requirements-eval.txt
```

## 本地 smoke test

仓库内置了一个极小的 identity adapter 测试，不需要下载模型权重：

```bash
moss-eval validate --config configs/examples/identity_smoke.yaml
moss-eval run --config configs/examples/identity_smoke.yaml --force
moss-eval summarize --exp-dir examples/outputs --output-dir examples/csv_results
```

输出结构：

```text
examples/outputs/identity/default/tiny/
  gt_audios/
  syn_audios/
  manifest.json
examples/outputs/identity/default/results.json
```

## 跑 MOSS-Audio-Tokenizer 系列

先把 `configs/examples/moss_audio_tokenizer.yaml` 里的 JSONL 路径改成你的测试集：

```bash
moss-eval run --config configs/examples/moss_audio_tokenizer.yaml --device cuda
```

`nq: all` 会展开为 `rvq1` 到 `rvq32`。如果只想跑部分层，可以写：

```yaml
nq: 1,2,4,8,16,32
```

或者命令行覆盖：

```bash
moss-eval run --config configs/examples/moss_audio_tokenizer.yaml --nq 1..8
```

## 接入新模型

实现一个 adapter 类即可。核心接口是：输入原始 `[C, T]` waveform 和 sample rate，输出
重建音频、用于对齐评测的 reference 音频，以及模型采样率。

详见 [adding_adapters.md](adding_adapters.md)。


## 多机器多卡

```bash
torchrun --nproc_per_node=8 -m moss_eval.cli run \
  --config configs/examples/moss_audio_tokenizer.yaml \
  --device cuda \
  --distributed torchrun
```

更多 Slurm 和手动分片示例见 [distributed.md](distributed.md)。
