# MOSS-Audio-Tokenizer-Eval

Objective reconstruction evaluation toolkit for audio tokenizers, neural codecs,
audio VAEs, and vocoders.

The default path is a single Conda/Python environment: model reconstruction,
common objective metrics, result manifests, and CSV summaries all run through one
CLI. Legacy or system-dependent metrics can still be added as optional adapters.

## Features

- Native Hugging Face support for:
  - `OpenMOSS-Team/MOSS-Audio-Tokenizer`
  - `OpenMOSS-Team/MOSS-Audio-Tokenizer-v2`
  - `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano`
- RVQ evaluation with arbitrary layers: `1`, `1,2,4,8`, `1..32`, or `all`.
- Multi-node/multi-GPU reconstruction through deterministic data sharding.
- Extensible adapter interface for other tokenizers, codecs, VAEs, and vocoders.
- Stable output contract: `gt_audios/`, `syn_audios/`, `manifest.json`, `results.json`.
- Built-in metrics: STOI, PESQ-NB, PESQ-WB, mel loss, spectral convergence, SDR, SI-SDR.
- Manifest-based skip logic to avoid accidentally reusing stale reconstructions.

## Installation

```bash
conda env create -f environment.yml
conda activate moss-audio-eval
```

If you already have a compatible PyTorch environment:

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
