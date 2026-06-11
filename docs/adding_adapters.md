# 接入新的 codec / tokenizer / VAE

新增模型只需要实现一个 adapter 类，并在 config 中通过 `module:ClassName` 引用。

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
        y = ...  # run encode/decode, optionally using nq
        return ReconstructOutput(audio=y, reference=ref, sample_rate=24000)
```

配置示例：

```yaml
models:
  - name: my-codec
    adapter: my_package.my_codec:MyCodecAdapter
    checkpoint: /path/to/ckpt.pt
    nq: 1..8
```

对于 VAE 或 vocoder，设置 `is_tokenizer = False`，并不要在配置里传 `nq`。
