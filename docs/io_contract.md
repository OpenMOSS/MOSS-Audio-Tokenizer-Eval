# 输入输出契约

## 输入 JSONL

每一行必须是一个 JSON object。音频路径字段支持以下任一 key：

- `audio_path`
- `audio_file`
- `audio`
- `wav`
- `path`

相对路径会按 JSONL 文件所在目录解析。空行会被视为错误，因为空行会破坏可复现的
样本索引。

## 输出目录

每个模型、RVQ 层、数据集对应一个 `output_dir`：

```text
exp_root/model_tag/rvqN_or_full/dataset_name/
  gt_audios/
  syn_audios/
  manifest.json
```

`gt_audios` 和 `syn_audios` 必须包含完全相同的文件名。当前框架使用
`000000_original_stem.flac` 这样的稳定文件名，避免不同目录下同名音频互相覆盖。

## manifest

`manifest.json` 记录 JSONL hash、模型 metadata、RVQ 层、样本数和输出文件名。
只有 manifest 与当前计划完全匹配，且所有 gt/syn 音频都存在时，才会跳过推理。

## results.json

指标结果写到 `output_dir` 的父目录：

```text
exp_root/model_tag/rvqN_or_full/results.json
```

结构保持兼容旧脚本：

```json
{
  "dataset_name": {
    "stoi": 0.95,
    "pesq-wb": 3.1
  },
  "_meta": {}
}
```
