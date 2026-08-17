# VAD 模型说明

README 中的 VAD 表格保留了模型大小、特征、误差和 RTF 基准；模型文件位于 [Hugging Face VAD 目录](https://huggingface.co/1024plus1/vad-filter-onnx-models/tree/main/vad)。

## 统一数据流

除 TEN-VAD 的音高特征外，特征提取均已导出到 ONNX 图中：

```text
PCM waveform
    |
    +--> framing/window --> STFT/Fbank/MelBank --> neural network --> speech probability
    |                                                               |
    +<------------------------- recurrent cache/state --------------+
```

## FireRedVAD

FireRedVAD 使用 Fbank 特征和带缓存的流式网络。每个 hop 输入一帧，网络输出语音概率以及下一帧使用的 cache；`int8` 版本使用动态量化权重。

```text
waveform -> Fbank -> causal feature stack -> recurrent VAD head -> probability
                                      ^                         |
                                      +------- cache_in/out ----+
```

## FSMN-VAD

FSMN-VAD 提供 8 kHz 和 16 kHz 两套模型，输入为 25 ms Fbank 帧、10 ms hop。时间记忆由 causal FSMN memory block 保存，导出后 cache 作为显式 ONNX 输入输出。

## Silero VAD

Silero VAD 使用 STFT 前端，v4/v5/v6 以及 `op15` 变体均保留各自的采样和窗口配置。网络是带上下文状态的序列分类器，输出概率和状态张量。

## TEN-VAD

TEN-VAD 的 ONNX 图接收 40 维 log-mel 特征；第 41 维是由 `utils/pitch-estimator.{h,cc}` 计算的有状态 F0。该 LPC/Viterbi 音高估计不能可靠地追踪到 ONNX，因此在图外完成。

```text
PCM -> 40-dim HTK log-mel -> ONNX TEN-VAD --+
                                             +--> speech probability
PCM -> LPC/Viterbi pitch (41st feature) -----+
```

## NeMo MarbleNet v2

MarbleNet 是非因果卷积栈，没有 recurrent cache。C++ 流式封装以滑动窗口重复运行整段网络，并保留中间帧，因此延迟高于 FSMN/FireRed 的显式缓存方案。

## 验证方法

FireRed/FSMN 使用固定随机种子、零初始化 cache 的 PyTorch 对照；TEN-VAD 对照上游 TensorFlow/ONNX 图；MarbleNet 对照对应的窗口推理结果。RTF 使用 Intel Xeon Silver 4316，5 次 warmup、20 次测量。
