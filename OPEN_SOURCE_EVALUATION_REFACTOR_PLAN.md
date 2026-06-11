# MOSS-Audio-Tokenizer 评测代码开源化重构计划

## 背景

当前仓库的重建评测流程是：先用 codec / tokenizer / VAE 的推理脚本生成
`output_dir/{gt_audios,syn_audios}`，再在不同 Python/Conda 环境中运行多套客观
评测指标。

开源目标是把这个仓库整理成一个通用、可复现、易扩展的音频重建客观指标评测工具，
用于评测 audio tokenizer、audio VAE、vocoder 等模型。对于 audio tokenizer，
必须支持 RVQ 任意层数的重建评测。

需要原生支持以下 MOSS-Audio-Tokenizer 系列模型：

- `OpenMOSS-Team/MOSS-Audio-Tokenizer`
- `OpenMOSS-Team/MOSS-Audio-Tokenizer-v2`
- `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano`

根据官方 MOSS-Audio-Tokenizer README，MOSS-Audio-Tokenizer 是 24 kHz 单声道模型；
v2 和 Nano 是 48 kHz 双声道模型。官方示例中 RVQ 子层重建通过
`enc.audio_codes[:nq]` 传给 `decode` 实现。

## 当前问题

| 模块 | 当前状态 | 开源化问题 |
|---|---|---|
| 入口脚本 | `reconstruct_evaluation/scripts/evaluate_speechtokenizer.sh` 把数据集发现、推理、指标执行、日志、路径配置都写在一个 bash 里。 | 难复用、难测试，新增 codec 或 metric 时容易继续膨胀。 |
| 路径 | 脚本里写死了 `/inspire/...` 私有绝对路径和特定用户的 Conda 环境。 | 外部用户无法直接运行。 |
| 环境 | 同一个 bash 进程里反复 `conda activate` 多个互相冲突的环境。 | 依赖冲突难排查，失败原因不清晰。 |
| codec 接入 | 每个 codec 都有一份自己的 shell wrapper 和推理脚本。 | 没有稳定 adapter 接口，不同 codec 的行为不一致。 |
| RVQ 控制 | `nq` 由脚本手动传入。 | 不支持统一的 `--nq all`、范围、多层列表，也无法标准化 quantizer dropout 模型的全层评测。 |
| 输出校验 | 是否跳过推理只看 JSONL 行数和生成音频数量。 | 可能错误复用旧输出，例如换了模型、换了 nq、换了数据但文件数没变。 |
| 结果 | metric 会更新 `results.json`，但 provenance 和 bps 元数据需要手动补。 | 难复现、难审计，后处理容易出错。 |
| 文档 | README 仍偏内部流程说明，并引用私有分支和内部路径。 | 需要面向外部用户的安装、quickstart、配置示例和扩展指南。 |

## 重构计划

| 阶段 | 目标 | 具体工作 | 主要文件 / 产物 | 正确性检查 |
|---|---|---|---|---|
| 1. 固化输入输出契约 | 在兼容现有 metric 的前提下定义公开接口。 | 文档化输入 JSONL schema、支持的音频字段、输出目录结构、文件命名规则、采样率/声道处理策略、结果 JSON schema。 | `docs/io_contract.md`，更新 README。 | validator 能拒绝缺失音频路径、重复输出名、空 JSONL、缺失 `gt_audios` / `syn_audios`、文件数量不匹配等情况。 |
| 2. 用 CLI 替代大型 bash | 用户不需要编辑 shell 脚本即可运行评测。 | 增加 Python CLI，提供 `reconstruct`、`metrics`、`run`、`summarize`、`validate` 子命令。旧 shell 脚本只保留为薄示例。 | 包入口，例如 `moss_eval.cli`；示例命令。 | `--help` 可用；tiny fixture 能跑通 inference-only、metrics-only 和完整 pipeline。 |
| 3. 配置驱动编排 | 移除代码中的私有路径。 | 引入 YAML/TOML 配置，描述数据集、实验根目录、模型 adapter、RVQ 层数、metric runners、Python 可执行文件/Conda prefix、device、batch/chunk 参数和 skip 策略。 | `configs/examples/moss_audio_tokenizer.yaml`，`configs/examples/external_codec.yaml`。 | 示例配置不包含私有绝对路径，并能通过 schema 校验。 |
| 4. 按步骤隔离环境 | 可靠解决依赖冲突。 | 每个 inference 或 metric runner 都作为独立 subprocess 执行，显式指定 `python`、`cwd`、`PYTHONPATH` 和环境变量。不依赖当前 shell 的 `conda activate` 状态。支持 `python`、`conda run -p` 或容器命令模板。 | `runner/subprocess.py`，metric runner 配置。 | 任一 metric 失败时能报告 command、cwd、环境名、日志路径和 exit code，且不破坏其他结果。 |
| 5. 定义统一模型 adapter 接口 | 让 codec / VAE / vocoder 接入方式一致。 | 增加 adapter protocol：`load()`、`prepare_audio()`、`reconstruct(audio, sample_rate, nq=None)`、`metadata()`。tokenizer 暴露 `max_nq`；VAE/vocoder 标记 `nq=None`。 | `adapters/base.py`，`adapters/registry.py`。 | fake adapter 单测覆盖 shape、采样率、确定性文件名和 metadata。 |
| 6. 原生 MOSS adapters | 直接支持三个 MOSS 系列模型。 | 基于 Hugging Face `AutoModel.from_pretrained(..., trust_remote_code=True)` 实现 MOSS、v2、Nano adapters。MOSS 转 24 kHz mono；v2/Nano 转 48 kHz stereo；decode 后裁剪到输入长度。RVQ 子层用 `enc.audio_codes[:nq]`。 | `adapters/moss_audio_tokenizer.py`，MOSS 示例配置。 | 在权重可用时，对每个模型做短音频 smoke test；检查 `nq=1`、`nq=max`、非法 `nq`。 |
| 7. RVQ 层调度 | 把任意 RVQ 层评测作为一等能力。 | 支持 `--nq 1`、`--nq 1,2,4,8`、`--nq 1..32`、`--nq all`。对 quantizer dropout 模型默认评测所有层。输出 tag 和 metadata 必须包含 RVQ 层数。 | CLI parser、config schema、run planner。 | planner 正确展开 RVQ spec；对非 tokenizer 模型传 RVQ spec 时拒绝，除非显式允许。 |
| 8. 增加输出 manifest | 防止错误跳过和旧结果污染。 | 每个 dataset/run 写 `manifest.json`，包含输入 JSONL hash、model id、revision/checkpoint、adapter version、nq、采样率、声道数、音频数量、命令和时间戳。只有 manifest 完全匹配时才允许 skip。 | 每个 `output_dir` 下的 `manifest.json`。 | 改 JSONL、model id、nq 或 adapter 配置都会触发重新推理。 |
| 9. 清理 metric plugin | 让指标可扩展且可复现。 | 把现有 SPT3、UTMOS、DAC metric 包成具名 metric runner，声明输出字段。每个 runner 读取 `output_dir`，写 namespaced results，可启用/禁用。 | `metrics/registry.py`，runner configs，迁移后的 metric entrypoints。 | 同一个 metric 重跑应幂等；部分 metric 失败时保留已有有效结果，并标记失败状态。 |
| 10. 规范结果 schema 和 bps | 不再手动改 `results.json`。 | 自动写入模型元数据和 bitrate 字段：frame rate、codebook size、num quantizers、bps、采样率、声道数、model family、model id。VAE/vocoder 支持手动 override。 | `results.json`，`summary.csv`，更新 `process_results`。 | summary 生成依赖 schema 字段；缺失 bitrate 元数据时给出 warning。 |
| 11. 增加测试和 fixtures | 在开源前保护正确性。 | 加 tiny 生成音频和 JSONL fixtures。单测覆盖 adapter、路径处理、manifest skip、RVQ planner、result merge、metric runner 命令构造。GPU 相关做可选 integration tests。 | `tests/`，`tests/fixtures/`。 | 无私有数据和模型权重时 `pytest` 也能通过；integration tests 只有设置 env vars 时才运行。 |
| 12. 开源发布整理 | 让仓库达到可发布状态。 | 重写 README，增加安装说明、quickstart、模型支持矩阵、扩展指南、license notice、依赖文件、`.gitignore` 和 examples。私有实验脚本移除或隔离。 | README，`pyproject.toml`，`requirements*.txt`，`examples/`。 | fresh clone 后能按文档运行 validation 和 fixture tests，不依赖私有 `/inspire/...` 路径。 |

## 建议的公开 CLI 形态

| 命令 | 用途 | 示例 |
|---|---|---|
| `moss-eval validate` | 运行前检查 dataset JSONL 和配置。 | `moss-eval validate --config configs/examples/moss_audio_tokenizer.yaml` |
| `moss-eval reconstruct` | 只生成 `gt_audios/` 和 `syn_audios/`。 | `moss-eval reconstruct --config cfg.yaml --model moss-audio-tokenizer --nq 1..32` |
| `moss-eval metrics` | 对已有重建目录运行指标。 | `moss-eval metrics --output-dir exp/MOSS/rvq8/libritts` |
| `moss-eval run` | 完整流程：先重建，再评测。 | `moss-eval run --config cfg.yaml` |
| `moss-eval summarize` | 汇总结果 JSON 为 CSV/Markdown。 | `moss-eval summarize --exp-dir exp --output summary.csv` |

## MOSS 模型支持方案

| 模型别名 | 默认 HF Repo | 音频处理策略 | RVQ 策略 | 备注 |
|---|---|---|---|---|
| `moss-audio-tokenizer` | `OpenMOSS-Team/MOSS-Audio-Tokenizer` | 重采样到 24 kHz；转单声道。 | `enc.audio_codes[:nq]` 后 decode；最大层数从 model/config 或 encoded codes 推断。 | 与官方 reconstruction 示例保持一致。 |
| `moss-audio-tokenizer-v2` | `OpenMOSS-Team/MOSS-Audio-Tokenizer-v2` | 重采样到 48 kHz；保证双声道，单声道 repeat，多声道截断。 | `enc.audio_codes[:nq]` 后 decode；默认完整 RVQ。 | 保存前裁剪 decoded waveform 到输入长度。 |
| `moss-audio-tokenizer-nano` | `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano` | 同 v2：48 kHz 双声道。 | `enc.audio_codes[:nq]` 后 decode；默认完整 RVQ。 | 与 v2 共用 adapter 逻辑，仅 repo id 不同。 |

## 里程碑

| 里程碑 | 交付物 | 验收标准 |
|---|---|---|
| M1 | 公开 I/O 契约、配置 schema、CLI 骨架。 | tiny fixture 能校验配置并打印 planned runs。 |
| M2 | 重建 adapters 和 manifest skip 逻辑。 | fake adapter 和 MOSS adapter 能生成合法的 `gt_audios/`、`syn_audios/`、`manifest.json`。 |
| M3 | 环境隔离的 metric runners。 | 现有 SPT3/UTMOS/DAC metrics 能作为独立 subprocess 运行并安全更新结果。 |
| M4 | RVQ sweep 和结果汇总。 | `--nq all` 能创建每个 RVQ 层的输出，summary CSV 包含 bps 和 provenance。 |
| M5 | 开源发布清理。 | README quickstart 和 tests 不依赖私有 `/inspire/...` 路径。 |

## 立即下一步

| 顺序 | 任务 | 原因 |
|---|---|---|
| 1 | 增加 config、JSONL、output dir 的 schema 和 validator。 | 先防住静默错误，保证后续重构不破坏正确性。 |
| 2 | 实现只规划、不跑模型的 CLI planner。 | 可以先验证实验展开、RVQ 调度和路径生成是否正确。 |
| 3 | 把当前 `evaluate_speechtokenizer.sh` 的行为迁移成配置示例。 | 在重构内部实现前保留现有行为。 |
| 4 | 实现 MOSS adapter 和 fake adapter。 | fake adapter 支持 CI；MOSS adapter 覆盖最核心的开源模型族。 |
| 5 | 把现有 metrics 包成隔离 subprocess runners。 | 直接解决当前最主要的环境冲突问题。 |
