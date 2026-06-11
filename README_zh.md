# MOSS-Audio-Tokenizer-Eval 中文说明

MOSS-Audio-Tokenizer-Eval 是一个用于评测 audio tokenizer、neural codec、audio VAE、vocoder 重建质量的客观指标工具包。

项目默认走一个 Conda/Python 环境完成：模型重建、常用客观指标、manifest、结果汇总都通过同一个 CLI 管理。对于有系统依赖或权重分发复杂的指标，可以作为 optional metric 额外接入。

## 核心能力

- 原生支持 MOSS-Audio-Tokenizer 系列模型：
  - `OpenMOSS-Team/MOSS-Audio-Tokenizer`
  - `OpenMOSS-Team/MOSS-Audio-Tokenizer-v2`
  - `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano`
- 支持任意 RVQ 层重建评测：`1`、`1,2,4,8`、`1..32`、`all`。
- 支持多机器多卡重建：基于 deterministic sharding，不需要 DDP。
- 通过 adapter 接口轻松接入其他 tokenizer、codec、VAE、vocoder。
- 稳定输出结构：`gt_audios/`、`syn_audios/`、`manifest.json`、`results.json`。
- 内置常用客观指标：STOI、PESQ-NB、PESQ-WB、mel loss、spectral convergence、SDR、SI-SDR。
- 基于 manifest 的跳过逻辑，避免错误复用旧模型、旧数据或旧 RVQ 层的结果。

## 安装

推荐使用项目提供的单环境配置：

```bash
conda env create -f environment.yml
conda activate moss-audio-eval
```

如果你已经有兼容的 PyTorch 环境，也可以直接安装 pip 依赖：

```bash
pip install -r requirements-eval.txt
```

安装后可以使用命令行入口：

```bash
moss-eval --help
```

如果尚未安装 package entrypoint，也可以直接用：

```bash
python -m moss_eval.cli --help
```

## 快速自测

仓库内置了一个极小的 identity adapter 测试，不需要下载模型权重：

```bash
moss-eval validate --config configs/examples/identity_smoke.yaml
moss-eval run --config configs/examples/identity_smoke.yaml --force
moss-eval summarize --exp-dir examples/outputs --output-dir examples/csv_results
```

对应的 Python module 调用方式：

```bash
python -m moss_eval.cli validate --config configs/examples/identity_smoke.yaml
python -m moss_eval.cli run --config configs/examples/identity_smoke.yaml --force
python -m moss_eval.cli summarize --exp-dir examples/outputs --output-dir examples/csv_results
```

## 评测 MOSS-Audio-Tokenizer

先修改示例配置中的 JSONL 路径：

```text
configs/examples/moss_audio_tokenizer.yaml
```

然后运行：

```bash
moss-eval run --config configs/examples/moss_audio_tokenizer.yaml --device cuda
```

示例配置默认使用 `nq: all`，会展开为所有 RVQ 层。例如 MOSS 系列默认会评测 `rvq1` 到 `rvq32`。

如果只想评测部分层，可以在配置里写：

```yaml
nq: 1,2,4,8,16,32
```

也可以在命令行覆盖：

```bash
moss-eval run --config configs/examples/moss_audio_tokenizer.yaml --nq 1..8
```

## 多机器多卡

本项目使用数据分片支持多机器多卡。每个进程处理满足以下条件的样本：

```text
sample_index % world_size == rank
```

这种方式不需要 DDP，也不需要同步模型参数，适合重建评测这种 embarrassingly parallel 的任务。

单机 8 卡：

```bash
torchrun --nproc_per_node=8 -m moss_eval.cli run \
  --config configs/examples/moss_audio_tokenizer.yaml \
  --device cuda \
  --distributed torchrun
```

多机 torchrun：

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

Slurm / srun：

```bash
srun python -m moss_eval.cli run \
  --config configs/examples/moss_audio_tokenizer.yaml \
  --device cuda \
  --distributed slurm
```

`--distributed auto` 会自动识别 torchrun 或 Slurm 环境。

更多说明见 [docs/distributed.md](docs/distributed.md)。

## 输入 JSONL 格式

每一行必须是一个 JSON object。音频路径字段支持以下任一 key：

- `audio_path`
- `audio_file`
- `audio`
- `wav`
- `path`

示例：

```jsonl
{"audio_path": "/path/to/audio_0001.wav"}
{"audio_file": "relative/path/audio_0002.flac"}
```

相对路径会按 JSONL 文件所在目录解析。空行会被认为是错误，因为空行会破坏可复现的样本索引。

完整输入输出契约见 [docs/io_contract.md](docs/io_contract.md)。

## 输出结构

每个模型、RVQ 层、数据集对应一个输出目录：

```text
exp_root/model_tag/rvqN_or_full/dataset_name/
  gt_audios/
  syn_audios/
  manifest.json
```

结果文件写在该 dataset 目录的父目录：

```text
exp_root/model_tag/rvqN_or_full/results.json
```

`gt_audios/` 和 `syn_audios/` 内的文件名完全一致，方便指标脚本做一一对应。

## 指标

默认支持：

- `stoi`
- `pesq_nb`
- `pesq_wb`
- `mel_loss`
- `spectral_convergence`
- `sdr`
- `sisdr`

可以用命令行指定指标：

```bash
moss-eval metrics \
  --output-dir exp/MOSS-Audio-Tokenizer/rvq8/my_eval_set \
  --metrics stoi,mel_loss,sisdr
```

`PESQ` 依赖 `pesq` 包；`SDR` 优先使用 `mir_eval`，如果环境没有 `mir_eval`，会 fallback 到普通 waveform SDR。

## 接入新的 tokenizer / codec / VAE / vocoder

新增模型只需要实现一个 adapter 类。adapter 接收原始 waveform `[C, T]` 和 sample rate，输出：

- `audio`：模型重建音频
- `reference`：用于公平对齐评测的参考音频
- `sample_rate`：输出音频采样率

最小示例：

```python
from moss_eval.adapters.base import AudioModelAdapter, ReconstructOutput
from moss_eval import audio as audio_utils

class MyCodecAdapter(AudioModelAdapter):
    is_tokenizer = True
    max_nq = 8

    def load(self):
        self.model = ...

    def reconstruct(self, wav, sample_rate, nq=None):
        ref = audio_utils.resample_audio(wav, sample_rate, 24000, device=self.device)
        ref = audio_utils.ensure_channels(ref, 1)
        y = ...  # encode/decode, optionally using nq
        return ReconstructOutput(audio=y, reference=ref, sample_rate=24000)
```

配置中这样引用：

```yaml
models:
  - name: my-codec
    adapter: my_package.my_codec:MyCodecAdapter
    checkpoint: /path/to/ckpt.pt
    nq: 1..8
```

对于 VAE 或 vocoder，设置 `is_tokenizer = False`，并且不要传 `nq`。

更多说明见 [docs/adding_adapters.md](docs/adding_adapters.md)。

## 常用命令

校验数据集和配置：

```bash
moss-eval validate --config configs/examples/moss_audio_tokenizer.yaml
```

只做重建：

```bash
moss-eval reconstruct --config configs/examples/moss_audio_tokenizer.yaml --device cuda
```

只对已有输出跑指标：

```bash
moss-eval metrics --output-dir exp/model/rvq8/dataset
```

完整流程：

```bash
moss-eval run --config configs/examples/moss_audio_tokenizer.yaml --device cuda
```

汇总结果：

```bash
moss-eval summarize --exp-dir exp --output-dir csv_results
```
