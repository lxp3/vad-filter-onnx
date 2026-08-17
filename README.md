# vad-filter-onnx

## Introduction

This project provides a simple, efficient C++/Python interface for running
audio ONNX models:

- **Simple and efficient inference.** Feature extraction is built into each
  ONNX graph, so callers pass raw waveform samples without an extra Fbank,
  STFT, or mel-processing pipeline.
- **Low-latency streaming.** The API focuses on real-time use, with explicit
  frame-wise state and low-latency streaming support where the model allows it.
- **Broad model coverage.** The repository includes VAD, denoise, speech
  enhancement, and speaker-diarization models, with more model families sharing
  the same interface.
- **Cross-platform.** The C++ implementation supports Linux, Windows, and
  macOS.

The ONNX files are hosted in the [Hugging Face model repository](https://huggingface.co/1024plus1/vad-filter-onnx-models). Browse the [VAD models](https://huggingface.co/1024plus1/vad-filter-onnx-models/tree/main/vad) or [denoise models](https://huggingface.co/1024plus1/vad-filter-onnx-models/tree/main/denoise).

# VAD models

Model architecture and implementation details are documented in [doc/vad.md](doc/vad.md). The table below retains benchmark results; model files are hosted in the [Hugging Face ONNX model repository](https://huggingface.co/1024plus1/vad-filter-onnx-models/tree/main/vad).

<table>
<thead>
<tr>
  <th rowspan="2">Model</th>
  <th rowspan="2">File size<br>(MB)</th>
  <th rowspan="2">Feature</th>
  <th rowspan="2">Sample<br>rate</th>
  <th colspan="2" align="center">Frame</th>
  <th colspan="2" align="center">Max diff</th>
  <th colspan="3" align="center">RTF</th>
</tr>
<tr>
  <th>Length</th>
  <th>Shift</th>
  <th>Logits</th>
  <th>Cache</th>
  <th>Online<br>(5s)</th>
  <th>Offline<br>(5s)</th>
  <th>Offline<br>(120s)</th>
</tr>
</thead>
<tbody>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/vad/firered_vad.onnx"><code>firered_vad.onnx</code></a></td><td align="right">3.30</td><td rowspan="2" valign="middle">Fbank</td><td align="right">16000</td><td align="right">25ms</td><td align="right">10ms</td><td align="right">0.00000417</td><td align="right">0.00030851</td><td align="right">0.011287</td><td align="right">0.011907</td><td align="right">0.011891</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/vad/firered_vad.int8.onnx"><code>firered_vad.int8.onnx</code></a></td><td align="right">1.76</td><td align="right">16000</td><td align="right">25ms</td><td align="right">10ms</td><td align="right">0.05357799</td><td align="right">5.96644974</td><td align="right">0.010993</td><td align="right">0.011226</td><td align="right">0.011194</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/vad/fsmn_vad.16k.onnx"><code>fsmn_vad.16k.onnx</code></a></td><td align="right">2.76</td><td rowspan="2" valign="middle">Fbank</td><td align="right">16000</td><td align="right">25ms</td><td align="right">10ms</td><td align="right">0.00000522</td><td align="right">0.00002837</td><td align="right">0.005762</td><td align="right">0.008597</td><td align="right">0.008684</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/vad/fsmn_vad.16k.int8.onnx"><code>fsmn_vad.16k.int8.onnx</code></a></td><td align="right">1.59</td><td align="right">16000</td><td align="right">25ms</td><td align="right">10ms</td><td align="right">0.07808840</td><td align="right">0.35685480</td><td align="right">0.005494</td><td align="right">0.008536</td><td align="right">0.008140</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/vad/fsmn_vad.8k.onnx"><code>fsmn_vad.8k.onnx</code></a></td><td align="right">1.97</td><td rowspan="2" valign="middle">Fbank</td><td align="right">8000</td><td align="right">25ms</td><td align="right">10ms</td><td align="right">0.00000000</td><td align="right">0.00000127</td><td align="right">0.004503</td><td align="right">0.006307</td><td align="right">0.006321</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/vad/fsmn_vad.8k.int8.onnx"><code>fsmn_vad.8k.int8.onnx</code></a></td><td align="right">0.80</td><td align="right">8000</td><td align="right">25ms</td><td align="right">10ms</td><td align="right">0.00000150</td><td align="right">0.01216167</td><td align="right">0.003612</td><td align="right">0.005645</td><td align="right">0.005662</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/vad/silero_vad.v4.onnx"><code>silero_vad.v4.onnx</code></a></td><td align="right">1.72</td><td rowspan="4" valign="middle">STFT</td><td align="right">16000</td><td align="right">32ms</td><td align="right">32ms</td><td align="right">0</td><td align="right">0</td><td align="right">0.005879</td><td align="right">0.005400</td><td align="right">0.005450</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/vad/silero_vad.v5.onnx"><code>silero_vad.v5.onnx</code></a></td><td align="right">2.21</td><td align="right">16000</td><td align="right">36ms</td><td align="right">32ms</td><td align="right">0</td><td align="right">0</td><td align="right">0.004727</td><td align="right">0.004806</td><td align="right">0.004792</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/vad/silero_vad.v6.onnx"><code>silero_vad.v6.onnx</code></a></td><td align="right">2.22</td><td align="right">16000</td><td align="right">36ms</td><td align="right">32ms</td><td align="right">0</td><td align="right">0</td><td align="right">0.004745</td><td align="right">0.004851</td><td align="right">0.004693</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/vad/silero_vad_16k_op15.v6.onnx"><code>silero_vad_16k_op15.v6.onnx</code></a></td><td align="right">1.23</td><td align="right">16000</td><td align="right">36ms</td><td align="right">32ms</td><td align="right">0</td><td align="right">0</td><td align="right">0.004717</td><td align="right">0.004659</td><td align="right">0.004600</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/vad/ten_vad.onnx"><code>ten_vad.onnx</code></a></td><td align="right">0.38</td><td rowspan="2" valign="middle">MelBank<br>+ pitch</td><td align="right">16000</td><td align="right">48ms</td><td align="right">16ms</td><td align="right">0.00000012</td><td align="right">0.00000083</td><td align="right">0.010696</td><td align="right">0.010725</td><td align="right">0.011800</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/vad/ten_vad.int8.onnx"><code>ten_vad.int8.onnx</code></a></td><td align="right">0.17</td><td align="right">16000</td><td align="right">48ms</td><td align="right">16ms</td><td align="right">0.01035109</td><td align="right">0.15491605</td><td align="right">0.011312</td><td align="right">0.011323</td><td align="right">0.012643</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/vad/nemo_marblenet_v2.onnx"><code>nemo_marblenet_v2.onnx</code></a></td><td align="right">1.50</td><td rowspan="2" valign="middle">Mel</td><td align="right">16000</td><td align="right">25ms</td><td align="right">10ms</td><td align="right">0.00000016</td><td align="right">no cache</td><td align="right">0.007780</td><td align="right">0.001238</td><td align="right">0.001601</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/vad/nemo_marblenet_v2.int8.onnx"><code>nemo_marblenet_v2.int8.onnx</code></a></td><td align="right">1.29</td><td align="right">16000</td><td align="right">25ms</td><td align="right">10ms</td><td align="right">0.02430001</td><td align="right">no cache</td><td align="right">0.007837</td><td align="right">0.001233</td><td align="right">0.001593</td></tr>
</tbody>
</table>

## Denoise models

Architecture and implementation details are documented in [doc/denoise.md](doc/denoise.md). The table below retains benchmark results; model files are hosted in the [Hugging Face ONNX model repository](https://huggingface.co/1024plus1/vad-filter-onnx-models/tree/main/denoise).

<table>
<thead>
<tr>
  <th rowspan="2">Model</th>
  <th rowspan="2">File size<br>(MB)</th>
  <th rowspan="2">Feature</th>
  <th rowspan="2">Sample<br>rate</th>
  <th colspan="2" align="center">Frame</th>
  <th colspan="2" align="center">Max diff</th>
  <th colspan="2" align="center">RTF</th>
</tr>
<tr>
  <th>Length</th>
  <th>Shift</th>
  <th>Waveform</th>
  <th>State</th>
  <th>Online<br>(5s)</th>
  <th>Offline<br>(5s)</th>
</tr>
</thead>
<tbody>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/denoise/gtcrn.onnx"><code>gtcrn.onnx</code></a></td><td align="right">2.43</td><td rowspan="2" valign="middle">STFT</td><td align="right">16000</td><td align="right">32ms</td><td align="right">16ms</td><td align="right">0.00000020</td><td align="right">0.00001526</td><td align="right">0.059596</td><td align="right">0.062197</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/denoise/gtcrn.int8.onnx"><code>gtcrn.int8.onnx</code></a></td><td align="right">1.72</td><td align="right">16000</td><td align="right">32ms</td><td align="right">16ms</td><td align="right">0.01503867</td><td align="right">0.93472493</td><td align="right">0.065453</td><td align="right">0.061722</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/denoise/dpdfnet2.onnx"><code>dpdfnet2.onnx</code></a></td><td align="right">10.50</td><td rowspan="2" valign="middle">STFT</td><td align="right">16000</td><td align="right">20ms</td><td align="right">10ms</td><td align="right">0.00000006</td><td align="right">0.00000859</td><td align="right">0.146970</td><td align="right">0.147022</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/denoise/dpdfnet2.int8.onnx"><code>dpdfnet2.int8.onnx</code></a></td><td align="right">3.79</td><td align="right">16000</td><td align="right">20ms</td><td align="right">10ms</td><td align="right">0.01028965</td><td align="right">1.12154722</td><td align="right">0.135047</td><td align="right">0.139796</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/denoise/dpdfnet4.onnx"><code>dpdfnet4.onnx</code></a></td><td align="right">11.89</td><td rowspan="2" valign="middle">STFT</td><td align="right">16000</td><td align="right">20ms</td><td align="right">10ms</td><td align="right">0.00000008</td><td align="right">0.00000954</td><td align="right">0.224903</td><td align="right">0.218286</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/denoise/dpdfnet4.int8.onnx"><code>dpdfnet4.int8.onnx</code></a></td><td align="right">4.81</td><td align="right">16000</td><td align="right">20ms</td><td align="right">10ms</td><td align="right">0.01495073</td><td align="right">1.12154722</td><td align="right">0.203133</td><td align="right">0.228366</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/denoise/dpdfnet8.onnx"><code>dpdfnet8.onnx</code></a></td><td align="right">14.68</td><td rowspan="2" valign="middle">STFT</td><td align="right">16000</td><td align="right">20ms</td><td align="right">10ms</td><td align="right">0.00000007</td><td align="right">0.00001144</td><td align="right">0.386592</td><td align="right">0.379440</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/denoise/dpdfnet8.int8.onnx"><code>dpdfnet8.int8.onnx</code></a></td><td align="right">6.84</td><td align="right">16000</td><td align="right">20ms</td><td align="right">10ms</td><td align="right">0.01465445</td><td align="right">1.12154722</td><td align="right">0.347815</td><td align="right">0.341847</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/denoise/dpdfnet2_8khz.onnx"><code>dpdfnet2_8khz.onnx</code></a></td><td align="right">9.93</td><td rowspan="2" valign="middle">STFT</td><td align="right">8000</td><td align="right">20ms</td><td align="right">10ms</td><td align="right">0.00000019</td><td align="right">0.00001144</td><td align="right">0.143448</td><td align="right">0.136883</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/denoise/dpdfnet2_8khz.int8.onnx"><code>dpdfnet2_8khz.int8.onnx</code></a></td><td align="right">3.44</td><td align="right">8000</td><td align="right">20ms</td><td align="right">10ms</td><td align="right">0.07346697</td><td align="right">0.37933445</td><td align="right">0.130049</td><td align="right">0.131854</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/denoise/dpdfnet8_8khz.onnx"><code>dpdfnet8_8khz.onnx</code></a></td><td align="right">14.10</td><td rowspan="2" valign="middle">STFT</td><td align="right">8000</td><td align="right">20ms</td><td align="right">10ms</td><td align="right">0.00000015</td><td align="right">0.00002289</td><td align="right">0.348253</td><td align="right">0.358878</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/denoise/dpdfnet8_8khz.int8.onnx"><code>dpdfnet8_8khz.int8.onnx</code></a></td><td align="right">6.48</td><td align="right">8000</td><td align="right">20ms</td><td align="right">10ms</td><td align="right">0.01285687</td><td align="right">0.23293401</td><td align="right">0.318560</td><td align="right">0.318578</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/denoise/dpdfnet2_48khz_hr.onnx"><code>dpdfnet2_48khz_hr.onnx</code></a></td><td align="right">17.06</td><td rowspan="2" valign="middle">STFT</td><td align="right">48000</td><td align="right">20ms</td><td align="right">10ms</td><td align="right">0.00000063</td><td align="right">0.00000477</td><td align="right">0.241080</td><td align="right">0.259222</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/denoise/dpdfnet2_48khz_hr.int8.onnx"><code>dpdfnet2_48khz_hr.int8.onnx</code></a></td><td align="right">7.80</td><td align="right">48000</td><td align="right">20ms</td><td align="right">10ms</td><td align="right">0.10554083</td><td align="right">0.31436515</td><td align="right">0.210734</td><td align="right">0.211461</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/denoise/dpdfnet8_48khz_hr.onnx"><code>dpdfnet8_48khz_hr.onnx</code></a></td><td align="right">21.23</td><td rowspan="2" valign="middle">STFT</td><td align="right">48000</td><td align="right">20ms</td><td align="right">10ms</td><td align="right">0.00000022</td><td align="right">0.00000763</td><td align="right">0.615036</td><td align="right">0.611795</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/denoise/dpdfnet8_48khz_hr.int8.onnx"><code>dpdfnet8_48khz_hr.int8.onnx</code></a></td><td align="right">10.85</td><td align="right">48000</td><td align="right">20ms</td><td align="right">10ms</td><td align="right">0.01225267</td><td align="right">0.14442804</td><td align="right">0.541708</td><td align="right">0.522404</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/denoise/frcrn_se_16k.onnx"><code>frcrn_se_16k.onnx</code></a></td><td align="right">54.46</td><td rowspan="2" valign="middle">STFT</td><td align="right">16000</td><td align="right">40ms</td><td align="right">20ms</td><td align="right">0.00002837</td><td align="right">no cache</td><td align="right">—</td><td align="right">13.894452</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/denoise/frcrn_se_16k.int8.onnx"><code>frcrn_se_16k.int8.onnx</code></a></td><td align="right">45.15</td><td align="right">16000</td><td align="right">40ms</td><td align="right">20ms</td><td align="right">0.10301372</td><td align="right">no cache</td><td align="right">—</td><td align="right">13.269131</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/denoise/mossformergan_se_16k.onnx"><code>mossformergan_se_16k.onnx</code></a></td><td align="right">29.50</td><td rowspan="2" valign="middle">STFT</td><td align="right">16000</td><td align="right">25ms</td><td align="right">6.25ms</td><td align="right">0.00003433</td><td align="right">no cache</td><td align="right">—</td><td align="right">9.651132</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/denoise/mossformergan_se_16k.int8.onnx"><code>mossformergan_se_16k.int8.onnx</code></a></td><td align="right">28.37</td><td align="right">16000</td><td align="right">25ms</td><td align="right">6.25ms</td><td align="right">0.01554050</td><td align="right">no cache</td><td align="right">—</td><td align="right">8.192783</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/denoise/mossformer2_se_48k.onnx"><code>mossformer2_se_48k.onnx</code></a></td><td align="right">276.19</td><td rowspan="2" valign="middle">Mel + STFT</td><td align="right">48000</td><td align="right">40ms</td><td align="right">8ms</td><td align="right">0.00000690</td><td align="right">no cache</td><td align="right">—</td><td align="right">0.418125</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/denoise/mossformer2_se_48k.int8.onnx"><code>mossformer2_se_48k.int8.onnx</code></a></td><td align="right">255.70</td><td align="right">48000</td><td align="right">40ms</td><td align="right">8ms</td><td align="right">0.05649380</td><td align="right">no cache</td><td align="right">—</td><td align="right">0.392090</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/denoise/resemble_enhance_denoiser.onnx"><code>resemble_enhance_denoiser.onnx</code></a></td><td align="right">73.03</td><td rowspan="2" valign="middle">STFT</td><td align="right">44100</td><td align="right">38ms</td><td align="right">9.5ms</td><td align="right">0.00000014</td><td align="right">no cache</td><td align="right">—</td><td align="right">0.544603</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/denoise/resemble_enhance_denoiser.int8.onnx"><code>resemble_enhance_denoiser.int8.onnx</code></a></td><td align="right">42.88</td><td align="right">44100</td><td align="right">38ms</td><td align="right">9.5ms</td><td align="right">0.07081813</td><td align="right">no cache</td><td align="right">—</td><td align="right">0.600248</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/denoise/deepfilternet2.onnx"><code>deepfilternet2.onnx</code></a></td><td align="right">16.02</td><td rowspan="2" valign="middle">STFT</td><td align="right">48000</td><td align="right">20ms</td><td align="right">10ms</td><td align="right">0.00000630</td><td align="right">0.00182045</td><td align="right">0.120311</td><td align="right">0.123132</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/denoise/deepfilternet2.int8.onnx"><code>deepfilternet2.int8.onnx</code></a></td><td align="right">13.32</td><td align="right">48000</td><td align="right">20ms</td><td align="right">10ms</td><td align="right">0.08411509</td><td align="right">2.42691803</td><td align="right">0.116793</td><td align="right">0.117444</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/denoise/deepfilternet3.onnx"><code>deepfilternet3.onnx</code></a></td><td align="right">15.37</td><td rowspan="2" valign="middle">STFT</td><td align="right">48000</td><td align="right">20ms</td><td align="right">10ms</td><td align="right">0.00000155</td><td align="right">0.00048999</td><td align="right">0.112678</td><td align="right">0.111382</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/denoise/deepfilternet3.int8.onnx"><code>deepfilternet3.int8.onnx</code></a></td><td align="right">12.68</td><td align="right">48000</td><td align="right">20ms</td><td align="right">10ms</td><td align="right">0.18774867</td><td align="right">2.42691803</td><td align="right">0.108910</td><td align="right">0.108322</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/denoise/dfsmn_ans_psm_48k.onnx"><code>dfsmn_ans_psm_48k.onnx</code></a></td><td align="right">50.42</td><td rowspan="2" valign="middle">Mel + STFT</td><td align="right">48000</td><td align="right">40ms</td><td align="right">20ms</td><td align="right">0.00000059</td><td align="right">0.00020444</td><td align="right">0.191640</td><td align="right">0.187346</td></tr>
<tr><td><a href="https://huggingface.co/1024plus1/vad-filter-onnx-models/resolve/main/denoise/dfsmn_ans_psm_48k.int8.onnx"><code>dfsmn_ans_psm_48k.int8.onnx</code></a></td><td align="right">23.71</td><td align="right">48000</td><td align="right">40ms</td><td align="right">20ms</td><td align="right">0.01314504</td><td align="right">7.17047120</td><td align="right">0.061175</td><td align="right">0.065487</td></tr>
</tbody>
</table>

```cpp
#include <denoise-filter-onnx-cxx-api.h>

auto handle = VadFilterOnnx::AutoDenoiseModel::create(
    "denoise/gtcrn.onnx");
auto denoise = handle->init(VadFilterOnnx::DenoiseConfig{});

std::vector<float> enhanced = denoise->decode(
    samples.data(), static_cast<int>(samples.size()), true);
```

## C++ CMake integration

The recommended way to use this project from another C++ CMake project is
`FetchContent`. Link the exported target `vad_filter_onnx::vad_filter_onnx`.

```cmake
include(FetchContent)

set(ENABLE_PYTHON OFF CACHE BOOL "" FORCE)
set(VAD_FILTER_ONNX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(VAD_FILTER_ONNX_BUILD_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    vad_filter_onnx
    GIT_REPOSITORY https://github.com/your-org/vad-filter-onnx.git
    GIT_TAG        main
)

FetchContent_MakeAvailable(vad_filter_onnx)

target_link_libraries(your_app PRIVATE vad_filter_onnx::vad_filter_onnx)
```

Minimal C++ usage:

```cpp
#include <vad-filter-onnx-cxx-api.h>

int main() {
    auto handle = VadFilterOnnx::AutoVadModel::create("vad/fsmn_vad.16k.onnx");
    VadFilterOnnx::VadConfig config;
    auto vad = handle->init(config);
    return vad ? 0 : 1;
}
```

ONNX Runtime is downloaded automatically by default. The archive cache is kept
under the build directory at `_deps/onnxruntime-downloads`. Override it with
`-DVAD_FILTER_ONNX_ORT_DOWNLOAD_DIR=/path/to/cache` if needed.

If you prefer a reusable wrapper, copy `cmake/vad_filter_onnx.cmake` into your
project, set `VAD_FILTER_ONNX_GIT_REPOSITORY` and `VAD_FILTER_ONNX_GIT_TAG`, and
include it before linking `vad_filter_onnx::vad_filter_onnx`.
