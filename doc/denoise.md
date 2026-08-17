# Denoise 模型说明

README 中的降噪表格保留了全部 benchmark；模型文件位于 [Hugging Face denoise 目录](https://huggingface.co/1024plus1/vad-filter-onnx-models/tree/main/denoise)。

## 统一数据流

```text
PCM waveform -> STFT/fbank frontend -> enhancement network -> mask/filter -> iSTFT -> waveform
                                      ^                         |
                                      +---- streaming state ----+
```

## GTCRN

GTCRN 使用 512 点 STFT、`sqrt(Hann)` 窗和 overlap-add。DNS3 网络预测频谱掩码；流式图每 hop 维护 cache，并带一个 hop 的内部延迟。

## DPDFNet

DPDFNet 由 2/4/8 个 DPRNN block 组成，提供 8/16/48 kHz 配置。图内包含 Vorbis-window STFT/ISTFT，网络输入输出原始 waveform，状态压成单一向量。

```text
waveform -> Vorbis STFT -> DPRNN blocks x N -> complex mask -> iSTFT -> waveform
                             ^                  |
                             +--- state -------+
```

## FRCRN

FRCRN 是离线非因果双 U-Net，使用 640 点卷积 STFT 和 SE gating。它没有可复用的流式状态，C++ 封装会缓冲完整输入，`input_finished=true` 后一次运行。

## MossFormerGAN

MossFormerGAN 的推理部分是 `SyncANet` generator。导出器将 STFT、功率压缩和 iSTFT 变成固定 DFT 的 Conv1d/ConvTranspose1d，长音频按 10 秒窗口、75% stride 分段并拼接。

## MossFormer2

MossFormer2 使用 180 维 Kaldi log-mel（60 mel、delta、delta-delta）预测掩码，再作用于独立的 1920/384/1920 STFT。导出图用 matmul DFT 替代 `torch.fft`，4 秒窗口用于长音频分段。

## Resemble Enhance Denoiser

该模型是 resemble-enhance 的 Denoiser 子模块：1680 点 centered STFT、四层 2D U-Net 掩码和 atan2 相位处理。模型完全离线、无状态，整段输入在结束时一次推理。

## DeepFilterNet 2/3

DeepFilterNet 是真正的因果帧递归网络：ERB 特征卷积/GRU 预测掩码，deep-filtering head 预测过去频谱帧上的复数滤波 tap。卷积历史、GRU、ERB running mean 和频谱 ring buffer 都打包进 `state_in/state_out`。

```text
waveform -> Vorbis STFT -> ERB features -> Conv/GRU encoder-decoder -> mask
       ^                                      |                         |
       |                                      +--> deep-filter taps ----+
       +<---------------- state (history, GRU, ring buffer) ----- iSTFT
```

## DFSMN-ANS-PSM

DFSMN-ANS 使用 120 维 Fbank 输入、9 个 causal `UniDeepFsmn` 层和 961-bin 频谱掩码。每层保存 19 帧、256 通道的历史（共 43,776 个 float），以 O(1) 每 hop 的方式运行。

## 量化和验证

`int8` 文件主要通过 ONNX Runtime 动态量化 Linear/Conv 权重生成；分组卷积、复杂 DFT 常量和部分 ConvTranspose 不量化。各模型的导出脚本会进行 streaming PyTorch/ONNX 对照，非因果模型使用完整音频离线对照。
