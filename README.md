[中文说明](README_zh.md) | English

# MOSS-Audio-Tokenizer-Eval

Objective reconstruction evaluation toolkit for audio tokenizers, neural codecs,
audio VAEs, and vocoders.

The default path is a single Conda/Python environment: model reconstruction,
common objective metrics, result manifests, and CSV summaries all run through one
CLI. Metrics that need extra dependencies or weights can be configured with
explicit paths.

## Features

- Native Hugging Face support for:
  - `OpenMOSS-Team/MOSS-Audio-Tokenizer`
  - `OpenMOSS-Team/MOSS-Audio-Tokenizer-v2`
  - `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano`
- RVQ evaluation with arbitrary layers: `1`, `1,2,4,8`, `1..32`, or `all`.
- Multi-node/multi-GPU reconstruction through deterministic data sharding.
- Extensible adapter interface for other tokenizers, codecs, VAEs, and vocoders.
- Stable output contract: `gt_audios/`, `syn_audios/`, `manifest.json`, `results.json`.
- Default metrics: STOI, PESQ-NB, PESQ-WB, mel loss, spectral convergence, SDR, SI-SDR, STFT.
- Metrics that need extra weights: speaker similarity `sim`, `utmos`.
- Manifest-based skip logic to avoid accidentally reusing stale reconstructions.

## Installation

```bash
conda env create -f environment.yml
conda activate moss-audio-eval
```

This installs model reconstruction, data preparation, default metrics, and
result summarization dependencies.

If you want visible pip download progress during installation, use the
equivalent split commands:

```bash
conda create -n moss-audio-eval python=3.10 pip ffmpeg libsndfile -c conda-forge
conda activate moss-audio-eval
pip install -r requirements-eval.txt
```

If you already have a compatible PyTorch environment, install the same pip
dependencies directly:

```bash
pip install -r requirements-eval.txt
```

## Smoke Test

This test uses an identity adapter and a tiny generated audio file. It does not
download model weights.

```bash
moss-eval validate --config configs/examples/identity_smoke.yaml
moss-eval run --config configs/examples/identity_smoke.yaml --force
moss-eval summarize --exp-dir examples/outputs --output-dir examples/csv_results
```

Equivalent without installing the package entrypoint:

```bash
python -m moss_eval.cli run --config configs/examples/identity_smoke.yaml --force
```

## Prepare LibriSpeech test-clean

Use the Hugging Face dataset below for a public, reproducible test-clean set:

```text
https://huggingface.co/datasets/AudioLLMs/librispeech_test_clean
```

The dataset stores WAV bytes in parquet `context.bytes` fields, so the parquet
files cannot be used directly as evaluation JSONL. This repository provides:

```text
scripts/prepare_hf_librispeech_test_clean.py
```

The script downloads or reads parquet files, extracts WAV files to
`extracted_wav/`, and writes JSONL files accepted by this evaluator.

Download and prepare on a networked machine:

```bash
python scripts/prepare_hf_librispeech_test_clean.py \
  --output-dir data/librispeech_test_clean_hf
```

The JSONL for normal evaluation is:

```text
data/librispeech_test_clean_hf/z-librispeech_test_clean_hf_order.jsonl
```

If automatic download is unstable, download these parquet files manually and
copy them to the evaluation machine:

```text
https://huggingface.co/datasets/AudioLLMs/librispeech_test_clean/resolve/main/data/test-00000-of-00002.parquet
https://huggingface.co/datasets/AudioLLMs/librispeech_test_clean/resolve/main/data/test-00001-of-00002.parquet
```

Then extract audio and JSONL without network access:

```bash
python scripts/prepare_hf_librispeech_test_clean.py \
  --local-parquet-dir /path/to/librispeech_test_clean/data \
  --output-dir data/librispeech_test_clean_hf
```

Important outputs:

- `extracted_wav/`: extracted WAV files.
- `z-librispeech_test_clean_hf_order.jsonl`: evaluation JSONL.
- `prepare_summary.json` and `prepare_report.md`: counts and output paths.

Use the generated JSONL in the config:

```yaml
datasets:
  - name: librispeech_test_clean_hf
    jsonl: /abs/path/to/data/librispeech_test_clean_hf/z-librispeech_test_clean_hf_order.jsonl
```

Then validate and run evaluation:

```bash
moss-eval validate --config configs/examples/moss_audio_tokenizer.yaml
moss-eval run --config configs/examples/moss_audio_tokenizer.yaml --device cuda
```

## Metrics

The toolkit supports the following reconstruction metrics. The example config
enables metrics that do not need extra model weights; `sim` and `utmos` require
additional checkpoints.

| Metric | Evaluation setting | Extra requirement |
|---|---|---|
| `stoi` | 16 kHz / mono, same filenames | None |
| `pesq-nb` | Resample to 16 kHz, PESQ NB mode | None |
| `pesq-wb` | 16 kHz | None |
| `mel_loss` | `[2048, 512]` windows, `[150, 80]` mel bins, sqrt-Hann, magnitude L1 + log L1 | None |
| `spectral_convergence` | Multi-resolution spectral convergence | None |
| `sdr` | `mir_eval.separation.bss_eval_sources` first | None |
| `sisdr` | L2-normalize, then projection/distortion formula | None |
| `stft` | 16 kHz / mono, two-scale `[2048, 512]` sqrt-Hann STFT loss | None |
| `sim` | 16 kHz / mono speaker similarity | `wavlm_large_finetune.pth` |
| `utmos` | External UTMOS script, temporary `24 kHz / mono` inputs | Install `requirements-optional.txt`; prepare `epoch=3-step=7459.ckpt`, `wav2vec_small.pt` |

## Evaluate MOSS-Audio-Tokenizer

Edit `configs/examples/moss_audio_tokenizer.yaml` and replace the JSONL path with
your evaluation dataset.

```bash
moss-eval run --config configs/examples/moss_audio_tokenizer.yaml --device cuda
```

The example uses `nq: all`, which expands to `rvq1` through `rvq32`. To evaluate
specific layers only:

```yaml
nq: 1,2,4,8,16,32
```

or override at runtime:

```bash
moss-eval run --config configs/examples/moss_audio_tokenizer.yaml --nq 1..8
```

## Speaker Similarity (`sim`)

`sim` uses the `wavlm_large_finetune.pth` checkpoint. Download this file
manually before running `sim`:

```text
https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view

filename: wavlm_large_finetune.pth
```

Recommended location:

```bash
mkdir -p pretrained_models/sim
```

You can also download it on a networked machine and copy it to an offline
machine.

Enable `sim` in the config:

```yaml
metrics:
  enabled: [stoi, pesq_nb, pesq_wb, mel_loss, spectral_convergence, sdr, sisdr, stft, sim]
  options:
    sim:
      model_path: pretrained_models/sim/wavlm_large_finetune.pth
      target_sr: 16000
```

For offline evaluation, copy the checkpoint to the offline machine first and set
`model_path` to that local file.

## `utmos`

Add `utmos` to `metrics.enabled` when needed. Install optional dependencies
before running it:

```bash
pip install -r requirements-optional.txt
```

The metric calls the external script in `work_dir`:

```bash
python do_reconstruct_evalation_given_dir.py --output_dir <dataset_output_dir>
```

Before calling the external script, this project prepares temporary
`24 kHz / mono` `gt_audios` and `syn_audios` directories. The original
reconstructed audio files are not modified. The CLI automatically sets
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for PyTorch 2.6+ checkpoint loading.

`utmos` needs two checkpoint files:

```text
https://huggingface.co/spaces/sarulab-speech/UTMOS-demo/resolve/bc80791f9e8d1bba44bf0319895e9edee078c6e9/epoch%3D3-step%3D7459.ckpt
https://huggingface.co/spaces/sarulab-speech/UTMOS-demo/resolve/bc80791f9e8d1bba44bf0319895e9edee078c6e9/wav2vec_small.pt
```

Put both files in the UTMOS work directory root. Another/offline machine must
also have the metric wrapper script and a Python environment with
`requirements-optional.txt` installed.

Enable them in config:

```yaml
metrics:
  enabled: [stoi, pesq_nb, pesq_wb, mel_loss, spectral_convergence, sdr, sisdr, stft, utmos]
  options:
    utmos:
      work_dir: /path/to/UTMOS-demo
      python_bin: /path/to/python
```

## Metric Run Examples

Run the default metrics on an existing reconstruction output:

```bash
python -m moss_eval.cli metrics \
  --config configs/examples/moss_audio_tokenizer.yaml \
  --output-dir exp/MOSS-Audio-Tokenizer/rvq16/my_eval_set \
  --metrics stoi,pesq_nb,pesq_wb,mel_loss,spectral_convergence,sdr,sisdr,stft \
  --device cuda
```

The result is written to:

```text
exp/MOSS-Audio-Tokenizer/rvq16/results.json
```

Batch-run metrics over multiple model/RVQ outputs and summarize:

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

Run reconstruction and all metrics after preparing `sim` and `utmos` weights:

```bash
python -m moss_eval.cli run \
  --config configs/examples/moss_audio_tokenizer.yaml \
  --device cuda \
  --metrics stoi,pesq_nb,pesq_wb,mel_loss,spectral_convergence,sdr,sisdr,stft,sim,utmos
```



## Multi-Node / Multi-GPU

Reconstruction can be sharded across processes. With `torchrun`, each process
automatically receives a rank and uses `cuda:${LOCAL_RANK}` when `--device cuda`
is passed:

```bash
torchrun --nproc_per_node=8 -m moss_eval.cli run \
  --config configs/examples/moss_audio_tokenizer.yaml \
  --device cuda \
  --distributed torchrun
```

For Slurm, use `--distributed slurm` or `--distributed auto`. See
[`docs/distributed.md`](docs/distributed.md) for multi-node and manual sharding
examples.

## Dataset JSONL

Each line must be a JSON object. Audio path keys can be any of:

- `audio_path`
- `audio_file`
- `audio`
- `wav`
- `path`

Relative paths are resolved from the JSONL file directory. See
[`docs/io_contract.md`](docs/io_contract.md) for the full contract.

## Add A New Model

Implement an adapter class and reference it from config as `module:ClassName`.
The adapter receives raw waveform `[C, T]` and sample rate, then returns the
reconstructed audio and the reference audio used for metric alignment.

See [`docs/adding_adapters.md`](docs/adding_adapters.md).
