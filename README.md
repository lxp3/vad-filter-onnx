# vad-filter-onnx

This project packages several VAD (and denoise) models behind one
consistent C++/Python interface:

- **Feature extraction baked into the ONNX graph.** Fbank/STFT/MelBank/Mel
  frontends are traced into the model, so the exported ONNX graph takes
  raw PCM waveform directly — no separate feature step to keep in sync.
- **One bit-packed sliding window for speech/silence detection.**
  `SlidingWindowBit` tracks recent frame decisions as bits and counts them
  with `std::popcount`, shared by every backend's voice start/end logic.
- **One 0~1 threshold, one config, for every backend.** Every model
  outputs a 0~1 speech probability, so a single `VadConfig` works
  unchanged across Silero, FSMN-VAD, TenVAD, FireRedVAD, and
  NeMo-MarbleNet.

# Vad models

The FireRedVAD exports were compared with the source PyTorch Stream-VAD model
using one second of deterministic 16 kHz float audio (`torch.manual_seed(20260723)`)
and zero-initialized caches. The FSMN-VAD exports used the corresponding 8 kHz
or 16 kHz source PyTorch model, the same duration and seed, zero-initialized
caches, and feature padding `[2, 2]`. The table reports the maximum absolute
error over the probability/logits output and all output caches.

RTF benchmarks use deterministic audio, 5 warmup runs, and 20 measured runs on
an Intel Xeon Silver 4316 CPU. Online RTF uses 5 seconds of audio and 100 ms
chunks; offline RTF is reported for both 5-second and 120-second inputs.

TEN-VAD has no PyTorch checkpoint upstream (it ships a TensorFlow-exported
ONNX graph plus C/C++ feature extraction), so `ten_vad.onnx` is compared
against that upstream graph driven with the same features rather than against
a source PyTorch model; the int8 row is compared against the float export.
Its 41st input feature is a pitch (F0) estimate, which is a stateful
LPC/Viterbi algorithm and so cannot be traced into the graph — it is a graph
input fed by `vad-filter-onnx/utils/pitch-estimator.{h,cc}`, while the 40
log-mel features are computed inside the graph. Note the mel filterbank must
match upstream's unnormalized HTK integer-bin triangles: substituting a
librosa Slaney filterbank drops frame accuracy from 0.9001 to 0.8491 on the
30 labeled files in TEN-VAD's own `testset/`.

NeMo-MarbleNet-v2.0 has no recurrent cache/state (a non-causal conv stack with
a receptive field wider than one output frame), so it has no cache max diff
column. Its C++ streaming implementation approximates real-time inference by
repeatedly running the whole model over a sliding window with left/right
context and keeping only the middle frames, matching NeMo's own real-time
demo approach; this adds a fixed extra latency versus the cached FSMN-VAD/
FireRedVAD models.

```bash
./build/test-rtf-online \
  --model-path public/models/fsmn_vad.16k.onnx \
  --chunk-ms 100 \
  --num-warmups 5 \
  --num-runs 20
```

<table>
<thead>
<tr>
  <th rowspan="2">Model</th>
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
<tr><td><a href="public/models/firered_vad.onnx"><code>firered_vad.onnx</code></a></td><td rowspan="2" valign="middle">Fbank</td><td align="right">16000</td><td align="right">25ms</td><td align="right">10ms</td><td align="right">0.00000417</td><td align="right">0.00030851</td><td align="right">0.011287</td><td align="right">0.011907</td><td align="right">0.011891</td></tr>
<tr><td><a href="public/models/firered_vad.int8.onnx"><code>firered_vad.int8.onnx</code></a></td><td align="right">16000</td><td align="right">25ms</td><td align="right">10ms</td><td align="right">0.05357799</td><td align="right">5.96644974</td><td align="right">0.010993</td><td align="right">0.011226</td><td align="right">0.011194</td></tr>
<tr><td><a href="public/models/fsmn_vad.16k.onnx"><code>fsmn_vad.16k.onnx</code></a></td><td rowspan="2" valign="middle">Fbank</td><td align="right">16000</td><td align="right">25ms</td><td align="right">10ms</td><td align="right">0.00000522</td><td align="right">0.00002837</td><td align="right">0.005762</td><td align="right">0.008597</td><td align="right">0.008684</td></tr>
<tr><td><a href="public/models/fsmn_vad.16k.int8.onnx"><code>fsmn_vad.16k.int8.onnx</code></a></td><td align="right">16000</td><td align="right">25ms</td><td align="right">10ms</td><td align="right">0.07808840</td><td align="right">0.35685480</td><td align="right">0.005494</td><td align="right">0.008536</td><td align="right">0.008140</td></tr>
<tr><td><a href="public/models/fsmn_vad.8k.onnx"><code>fsmn_vad.8k.onnx</code></a></td><td rowspan="2" valign="middle">Fbank</td><td align="right">8000</td><td align="right">25ms</td><td align="right">10ms</td><td align="right">0.00000000</td><td align="right">0.00000127</td><td align="right">0.004503</td><td align="right">0.006307</td><td align="right">0.006321</td></tr>
<tr><td><a href="public/models/fsmn_vad.8k.int8.onnx"><code>fsmn_vad.8k.int8.onnx</code></a></td><td align="right">8000</td><td align="right">25ms</td><td align="right">10ms</td><td align="right">0.00000150</td><td align="right">0.01216167</td><td align="right">0.003612</td><td align="right">0.005645</td><td align="right">0.005662</td></tr>
<tr><td><a href="public/models/silero_vad.v4.onnx"><code>silero_vad.v4.onnx</code></a></td><td rowspan="4" valign="middle">STFT</td><td align="right">16000</td><td align="right">32ms</td><td align="right">32ms</td><td align="right">0</td><td align="right">0</td><td align="right">0.005879</td><td align="right">0.005400</td><td align="right">0.005450</td></tr>
<tr><td><a href="public/models/silero_vad.v5.onnx"><code>silero_vad.v5.onnx</code></a></td><td align="right">16000</td><td align="right">36ms</td><td align="right">32ms</td><td align="right">0</td><td align="right">0</td><td align="right">0.004727</td><td align="right">0.004806</td><td align="right">0.004792</td></tr>
<tr><td><a href="public/models/silero_vad.v6.onnx"><code>silero_vad.v6.onnx</code></a></td><td align="right">16000</td><td align="right">36ms</td><td align="right">32ms</td><td align="right">0</td><td align="right">0</td><td align="right">0.004745</td><td align="right">0.004851</td><td align="right">0.004693</td></tr>
<tr><td><a href="public/models/silero_vad_16k_op15.v6.onnx"><code>silero_vad_16k_op15.v6.onnx</code></a></td><td align="right">16000</td><td align="right">36ms</td><td align="right">32ms</td><td align="right">0</td><td align="right">0</td><td align="right">0.004717</td><td align="right">0.004659</td><td align="right">0.004600</td></tr>
<tr><td><a href="public/models/ten_vad.onnx"><code>ten_vad.onnx</code></a></td><td rowspan="2" valign="middle">MelBank<br>+ pitch</td><td align="right">16000</td><td align="right">48ms</td><td align="right">16ms</td><td align="right">0.00000012</td><td align="right">0.00000083</td><td align="right">0.010696</td><td align="right">0.010725</td><td align="right">0.011800</td></tr>
<tr><td><a href="public/models/ten_vad.int8.onnx"><code>ten_vad.int8.onnx</code></a></td><td align="right">16000</td><td align="right">48ms</td><td align="right">16ms</td><td align="right">0.01035109</td><td align="right">0.15491605</td><td align="right">0.011312</td><td align="right">0.011323</td><td align="right">0.012643</td></tr>
<tr><td><a href="public/models/nemo_marblenet_v2.onnx"><code>nemo_marblenet_v2.onnx</code></a></td><td rowspan="2" valign="middle">Mel</td><td align="right">16000</td><td align="right">25ms</td><td align="right">10ms</td><td align="right">0.00000016</td><td align="right">no cache</td><td align="right">0.007780</td><td align="right">0.001238</td><td align="right">0.001601</td></tr>
<tr><td><a href="public/models/nemo_marblenet_v2.int8.onnx"><code>nemo_marblenet_v2.int8.onnx</code></a></td><td align="right">16000</td><td align="right">25ms</td><td align="right">10ms</td><td align="right">0.02430001</td><td align="right">no cache</td><td align="right">0.007837</td><td align="right">0.001233</td><td align="right">0.001593</td></tr>
</tbody>
</table>

## Denoise models

[`gtcrn.onnx`](public/models/gtcrn.onnx) uses the upstream
[GTCRN](https://github.com/Xiaobin-Rong/gtcrn) DNS3 checkpoint. The ONNX graph
accepts and returns normalized mono float waveform samples directly. Its
512-point STFT, inverse STFT, periodic `sqrt(Hann)` window, and overlap-add are
implemented with standard Torch/ONNX tensor operations.

- Sample rate: 16 kHz only
- ONNX input/output hop: 256 samples (16 ms)
- C++ input chunks: arbitrary size
- Stream output: delayed by one hop internally and flushed to the exact input
  length when `input_finished=true`

The exporter compares 20 consecutive streaming frames using deterministic
audio (`torch.manual_seed(20260723)`). The upstream `mix.wav` was also checked
over all 611 frames.

| Input | Enhanced waveform max abs diff | All caches max abs diff |
| :--- | ---: | ---: |
| Deterministic audio, 20 frames | 0.00000019650906 | 0.000015258789 |
| Upstream `mix.wav`, 611 frames | 0.00000062584877 | 0.000061064959 |

```bash
python scripts/export_onnx_gtcrn.py \
  --source-dir debug/gtcrn \
  --checkpoint debug/gtcrn/checkpoints/model_trained_on_dns3.tar \
  --output public/models/gtcrn.onnx
```

```cpp
#include <denoise-filter-onnx-cxx-api.h>

auto handle = VadFilterOnnx::AutoDenoiseModel::create(
    "public/models/gtcrn.onnx");
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
    auto handle = VadFilterOnnx::AutoVadModel::create("public/models/fsmn_vad.16k.onnx");
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
