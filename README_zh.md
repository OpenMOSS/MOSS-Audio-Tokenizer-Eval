# MOSS-Audio-Tokenizer-Eval 中文说明

MOSS-Audio-Tokenizer-Eval 是一个用于评测 audio tokenizer、neural codec、audio VAE、vocoder 重建质量的客观指标工具包。

项目默认走一个 Conda/Python 环境完成：模型重建、常用客观指标、manifest、结果汇总都通过同一个 CLI 管理。对于需要额外依赖或权重的指标，可以在配置里显式指定路径。

## 核心能力

- 原生支持 MOSS-Audio-Tokenizer 系列模型：
  - `OpenMOSS-Team/MOSS-Audio-Tokenizer`
  - `OpenMOSS-Team/MOSS-Audio-Tokenizer-v2`
  - `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano`
- 支持任意 RVQ 层重建评测：`1`、`1,2,4,8`、`1..32`、`all`。
- 支持多机器多卡重建：基于 deterministic sharding，不需要 DDP。
- 通过 adapter 接口轻松接入其他 tokenizer、codec、VAE、vocoder。
- 稳定输出结构：`gt_audios/`、`syn_audios/`、`manifest.json`、`results.json`。
- 默认支持常用客观指标：STOI、PESQ-NB、PESQ-WB、mel loss、spectral convergence、SDR、SI-SDR、STFT。
- 支持需要额外权重的指标：speaker similarity `sim`、`utmos`。
- 基于 manifest 的跳过逻辑，避免错误复用旧模型、旧数据或旧 RVQ 层的结果。

## 安装

推荐使用项目提供的单环境配置。该文件会安装模型重建、数据准备、默认指标和结果汇总所需的 Python 包：

```bash
conda env create -f environment.yml
conda activate moss-audio-eval
```

如果希望安装过程显示 pip 下载进度，也可以用等价的拆分方式：

```bash
conda create -n moss-audio-eval python=3.10 pip ffmpeg libsndfile -c conda-forge
conda activate moss-audio-eval
pip install -r requirements-eval.txt
```

如果你已经有兼容的 PyTorch 环境，也可以直接安装同一套 pip 依赖：

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

## 指标说明

当前项目支持以下重建质量指标。默认示例配置会启用无需额外模型权重的指标；`sim`、`utmos` 需要先准备对应权重。

| 指标 | 评测设置 | 额外文件或依赖 |
|---|---|---|
| `stoi` | 16 kHz / mono，按同名文件逐条计算 | 不需要 |
| `pesq-nb` | 重采样到 16 kHz 后按 NB 模式计算 | 不需要 |
| `pesq-wb` | 重采样到 16 kHz 后按 WB 模式计算 | 不需要 |
| `mel_loss` | 双尺度 `[2048, 512]`、mel bins `[150, 80]`、sqrt-Hann、幅度 L1 + log L1 | 不需要 |
| `spectral_convergence` | 多分辨率频谱收敛度 | 不需要 |
| `sdr` | `mir_eval.separation.bss_eval_sources` 优先 | 不需要 |
| `sisdr` | 先 L2 norm，再用 projection/distortion 公式 | 不需要 |
| `stft` | 16 kHz / mono，双尺度 `[2048, 512]` sqrt-Hann STFT loss | 不需要 |
| `sim` | 16 kHz / mono speaker similarity | 需要 `wavlm_large_finetune.pth` |
| `utmos` | 调用 UTMOS 外部脚本；评测前临时转 `24 kHz / mono` | 安装 `requirements-optional.txt`；准备 `epoch=3-step=7459.ckpt`、`wav2vec_small.pt` |

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

## 准备 LibriSpeech test-clean 数据集

推荐使用 Hugging Face 上的 `AudioLLMs/librispeech_test_clean` 作为公开可复现的 test-clean 评测集：

```text
https://huggingface.co/datasets/AudioLLMs/librispeech_test_clean
```

该数据集把音频 WAV bytes 存在 parquet 的 `context.bytes` 字段中，不能直接把 parquet 路径写进评测 JSONL。仓库提供了数据准备脚本：

```text
scripts/prepare_hf_librispeech_test_clean.py
```

脚本会完成三件事：下载或读取 parquet、提取音频到 `extracted_wav/`、生成本项目评测可直接读取的 JSONL。

### 在线下载并准备

在有网络的机器上直接运行：

```bash
python scripts/prepare_hf_librispeech_test_clean.py \
  --output-dir data/librispeech_test_clean_hf
```

输出目录会包含下载的 parquet、提取后的 WAV 文件和 JSONL。后续评测使用这个文件：

```text
data/librispeech_test_clean_hf/z-librispeech_test_clean_hf_order.jsonl
```

### 使用已下载 parquet

如果在线下载不稳定，或者需要先在有网络的机器下载后拷贝到离线机器，可以手动下载这两个 parquet 文件：

```text
https://huggingface.co/datasets/AudioLLMs/librispeech_test_clean/resolve/main/data/test-00000-of-00002.parquet
https://huggingface.co/datasets/AudioLLMs/librispeech_test_clean/resolve/main/data/test-00001-of-00002.parquet
```

假设两个 parquet 已经放在 `/path/to/librispeech_test_clean/data/`，运行：

```bash
python scripts/prepare_hf_librispeech_test_clean.py \
  --local-parquet-dir /path/to/librispeech_test_clean/data \
  --output-dir data/librispeech_test_clean_hf
```

这种方式不会访问网络，适合离线评测机器。

### 输出文件

主要输出如下：

- `extracted_wav/`：从 parquet 中提取出的 WAV 音频。
- `z-librispeech_test_clean_hf_order.jsonl`：评测入口 JSONL。
- `prepare_summary.json`、`prepare_report.md`：数据条数和输出路径。

### 接入评测

把生成的 JSONL 写入配置文件，例如：

```yaml
datasets:
  - name: librispeech_test_clean_hf
    jsonl: /abs/path/to/data/librispeech_test_clean_hf/z-librispeech_test_clean_hf_order.jsonl
```

然后检查配置并启动评测：

```bash
moss-eval validate --config configs/examples/moss_audio_tokenizer.yaml
moss-eval run --config configs/examples/moss_audio_tokenizer.yaml --device cuda
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

## Speaker Similarity / `sim` 指标

`sim` 使用 `wavlm_large_finetune.pth` 权重文件。运行 `sim` 前请先下载该文件：

```text
https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view

filename: wavlm_large_finetune.pth
```

推荐放置路径：

```bash
mkdir -p pretrained_models/sim
```

也可以在有网络的机器上手动下载后拷贝到离线机器。

配置中启用 `sim`：

```yaml
metrics:
  enabled: [stoi, pesq_nb, pesq_wb, mel_loss, spectral_convergence, sdr, sisdr, stft, sim]
  options:
    sim:
      model_path: pretrained_models/sim/wavlm_large_finetune.pth
      target_sr: 16000
```

离线机器上需要提前把该权重拷贝过去，并在配置里显式写本地 `model_path`。

## `utmos` 指标

`utmos` 按需加入 `metrics.enabled`。使用前安装可选依赖：

```bash
pip install -r requirements-optional.txt
```

该指标调用 `work_dir` 中的外部脚本：

```bash
python do_reconstruct_evalation_given_dir.py --output_dir <dataset_output_dir>
```

调用外部脚本前，本项目会临时生成 `24 kHz / mono` 的 `gt_audios`、`syn_audios`，原始重建音频不会被改动。CLI 调用 `utmos` 时会自动设置 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`，以兼容 PyTorch 2.6+ 的 checkpoint 加载行为。

`utmos` 需要两个权重文件：

```text
https://huggingface.co/spaces/sarulab-speech/UTMOS-demo/resolve/bc80791f9e8d1bba44bf0319895e9edee078c6e9/epoch%3D3-step%3D7459.ckpt
https://huggingface.co/spaces/sarulab-speech/UTMOS-demo/resolve/bc80791f9e8d1bba44bf0319895e9edee078c6e9/wav2vec_small.pt
```

下载后放到 UTMOS work_dir 根目录。换机器或离线运行时，需要同时准备 UTMOS work_dir 里的评测脚本、上述两个权重文件，以及安装了 `requirements-optional.txt` 的 Python 环境。

配置中启用：

```yaml
metrics:
  enabled: [stoi, pesq_nb, pesq_wb, mel_loss, spectral_convergence, sdr, sisdr, stft, utmos]
  options:
    utmos:
      work_dir: /path/to/UTMOS-demo
      python_bin: /path/to/python
```

## 启动指标评测示例

如果已经完成重建，输出目录形如：

```text
exp/MOSS-Audio-Tokenizer/rvq16/my_eval_set/
  gt_audios/
  syn_audios/
  manifest.json
```

只对这个目录跑默认指标：

```bash
python -m moss_eval.cli metrics \
  --config configs/examples/moss_audio_tokenizer.yaml \
  --output-dir exp/MOSS-Audio-Tokenizer/rvq16/my_eval_set \
  --metrics stoi,pesq_nb,pesq_wb,mel_loss,spectral_convergence,sdr,sisdr,stft \
  --device cuda
```

结果会写到该 dataset 目录的父目录：

```text
exp/MOSS-Audio-Tokenizer/rvq16/results.json
```

对一个实验目录下多个模型/RVQ 输出批量补跑指标：

```bash
for output_dir in exp/*/rvq*/my_eval_set; do
  python -m moss_eval.cli metrics \
    --config configs/examples/moss_audio_tokenizer.yaml \
    --output-dir "${output_dir}" \
    --metrics stoi,pesq_nb,pesq_wb,mel_loss,spectral_convergence,sdr,sisdr,stft \
    --device cuda
done

python -m moss_eval.cli summarize \
  --exp-dir exp \
  --output-dir csv_results
```

如果已经准备好 `sim` 和 `utmos` 权重，可以运行完整指标：

```bash
python -m moss_eval.cli run \
  --config configs/examples/moss_audio_tokenizer.yaml \
  --device cuda \
  --metrics stoi,pesq_nb,pesq_wb,mel_loss,spectral_convergence,sdr,sisdr,stft,sim,utmos
```

## 多机器多卡

本项目使用数据分片支持多机器多卡。每个进程处理满足以下条件的样本：

```text
sample_index % world_size == rank
```

这种方式不需要 DDP，也不需要同步模型参数，适合重建评测这类独立样本任务。

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
- `stft`

可选指标：

- `sim`
- `utmos`

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
- `reference`：用于计算指标的参考音频
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

检查数据集和配置：

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
