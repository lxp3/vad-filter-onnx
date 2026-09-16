# VAD 并发压测

日志：`vad.20260915.log`。

## 测试机器

主机 `192.168.1.125`（`ai-test`），VMware 虚拟机，CentOS 7。

| 项目 | 配置 |
| --- | --- |
| CPU | Intel Xeon Gold 5218R @ 2.10 GHz |
| 逻辑 CPU | 32 vCPU（1 thread/core，2 NUMA node × 16） |
| 内存 | 32 GB（无 Swap） |
| 内核 | Linux 3.10.0-1160.el7.x86_64 |

## 测试方式

程序：`test-vad-threads`。按 FreeSWITCH 媒体线程模型模拟：

- 进程内 **1 个 VAD handle**（ONNX Session / WebRTC 库共享），**每路通话 1 个 instance**。
- **每路一条 OS 线程**，对应 FS 的 session/media 线程；VAD **同步 inline** 跑在该线程上，不再开 VAD 线程池。
- 每 20 ms 生成一包随机 PCM int16（8 kHz，160 点），转 float 后按需 resample 到模型采样率，再 `decode()`。
- 用 `steady_clock` 对齐 20 ms 拍：处理完若未到下一拍则 sleep，超时则记 **xrun** 并赶下一拍。
- ORT `num_threads=1`。WebRTC 测 100/500/1000/3000/5000 路；ONNX 测 100/500/1000/1500/2000/2500/3000 路。每路 60 s。

## 指标说明

| 指标 | 含义 |
| --- | --- |
| 墙钟 (s) | 从齐步开跑到全部线程结束的实际经过时间。目标是 60 s；明显大于 60 s 说明已经跟不上 20 ms 实时节奏。墙钟 ≤ 61s 表示 60s 音频基本按 20ms 节拍跑完（允许少量超时赶工），该点才用于外推。 |
| cpu负载 | 进程平均 CPU 相对整机（32 核）。`100%` 表示吃满全部逻辑核。 |
| xrun | 单包处理超过 20 ms 时限的包占比。接近 0 表示媒体线程能在 ptime 内跑完 VAD。 |
| P99 (ms) | 单包处理时延的 99 分位（生成 PCM + resample + `decode`，不含 sleep）。20 ms 预算内才算实时。 |
| 聚合RTF | `CPU时间 / (路数 × 墙钟)`。墙钟被拉长时 RTF 会看起来变小，不能单独用来判断能否实时。 |
| RSS峰值 | 进程峰值物理内存（`VmHWM`）。含模型、每路 instance、线程栈等。 |

## 并发估算

每个模型选用 **墙钟 ≤ 61s** 的最高并发测点，再按 `预计最大并发 = N / cpu负载` 外推到整机 100%。墙钟 ≤ 61s 表示 60s 音频基本按 20ms 节拍跑完（允许少量超时赶工），该点才用于外推；该测点即使 xrun 或 cpu负载 > 90% 仍采用。预估内存按该测点 RSS 峰值随并发线性外推：`预估内存 = RSS峰值 × 预计最大并发 / N`。

<table>
<thead>
<tr>
  <th>模型</th>
  <th>预计最大并发</th>
  <th>预估内存</th>
</tr>
</thead>
<tbody>
<tr>
  <td><code>webrtc</code></td>
  <td align="right">23697</td>
  <td align="right">947 MB</td>
</tr>
<tr>
  <td><code>firered_vad.int8.onnx</code></td>
  <td align="right">1355</td>
  <td align="right">1.15 GB</td>
</tr>
<tr>
  <td><code>firered_vad.onnx</code></td>
  <td align="right">1259</td>
  <td align="right">1.05 GB</td>
</tr>
<tr>
  <td><code>fsmn_vad.16k.int8.onnx</code></td>
  <td align="right">680</td>
  <td align="right">461 MB</td>
</tr>
<tr>
  <td><code>fsmn_vad.16k.onnx</code></td>
  <td align="right">691</td>
  <td align="right">490 MB</td>
</tr>
<tr>
  <td><code>fsmn_vad.8k.int8.onnx</code></td>
  <td align="right">796</td>
  <td align="right">385 MB</td>
</tr>
<tr>
  <td><code>fsmn_vad.8k.onnx</code></td>
  <td align="right">833</td>
  <td align="right">386 MB</td>
</tr>
<tr>
  <td><code>pulsevad_81k.int8.onnx</code></td>
  <td align="right">1596</td>
  <td align="right">550 MB</td>
</tr>
<tr>
  <td><code>pulsevad_81k.onnx</code></td>
  <td align="right">2756</td>
  <td align="right">719 MB</td>
</tr>
<tr>
  <td><code>pulsevad.int8.onnx</code></td>
  <td align="right">1877</td>
  <td align="right">598 MB</td>
</tr>
<tr>
  <td><code>pulsevad.onnx</code></td>
  <td align="right">3282</td>
  <td align="right">697 MB</td>
</tr>
<tr>
  <td><code>silero_vad_16k_op15.v6.onnx</code></td>
  <td align="right">1884</td>
  <td align="right">186 MB</td>
</tr>
<tr>
  <td><code>silero_vad.v4.onnx</code></td>
  <td align="right">1263</td>
  <td align="right">203 MB</td>
</tr>
<tr>
  <td><code>silero_vad.v5.onnx</code></td>
  <td align="right">1953</td>
  <td align="right">194 MB</td>
</tr>
<tr>
  <td><code>silero_vad.v6.onnx</code></td>
  <td align="right">1724</td>
  <td align="right">183 MB</td>
</tr>
<tr>
  <td><code>ten_vad.int8.onnx</code></td>
  <td align="right">412</td>
  <td align="right">131 MB</td>
</tr>
<tr>
  <td><code>ten_vad.onnx</code></td>
  <td align="right">415</td>
  <td align="right">121 MB</td>
</tr>
</tbody>
</table>

## 压测结果

<table>
<thead>
<tr>
  <th>模型</th>
  <th>并发</th>
  <th>墙钟 (s)</th>
  <th>cpu负载</th>
  <th>xrun</th>
  <th>P99 (ms)</th>
  <th>聚合RTF</th>
  <th>RSS峰值</th>
</tr>
</thead>
<tbody>
<tr>
  <td rowspan="5"><code>webrtc</code></td>
  <td align="right">100</td>
  <td align="right">60.00</td>
  <td align="right">0.48%</td>
  <td align="right">0.000%</td>
  <td align="right">0.056</td>
  <td align="right">0.0015</td>
  <td align="right">7.08 MB</td>
</tr>
<tr>
  <td align="right">500</td>
  <td align="right">60.01</td>
  <td align="right">2.26%</td>
  <td align="right">0.000%</td>
  <td align="right">0.055</td>
  <td align="right">0.0014</td>
  <td align="right">23.63 MB</td>
</tr>
<tr>
  <td align="right">1000</td>
  <td align="right">60.02</td>
  <td align="right">4.59%</td>
  <td align="right">0.000%</td>
  <td align="right">0.060</td>
  <td align="right">0.0015</td>
  <td align="right">43.50 MB</td>
</tr>
<tr>
  <td align="right">3000</td>
  <td align="right">60.06</td>
  <td align="right">13.5%</td>
  <td align="right">0.053%</td>
  <td align="right">0.078</td>
  <td align="right">0.0014</td>
  <td align="right">122.3 MB</td>
</tr>
<tr>
  <td align="right">5000</td>
  <td align="right">60.10</td>
  <td align="right">21.1%</td>
  <td align="right">0.060%</td>
  <td align="right">0.091</td>
  <td align="right">0.0014</td>
  <td align="right">199.9 MB</td>
</tr>
<tr>
  <td rowspan="7"><code>firered_vad.int8.onnx</code></td>
  <td align="right">100</td>
  <td align="right">60.00</td>
  <td align="right">8.89%</td>
  <td align="right">0.13%</td>
  <td align="right">6.08</td>
  <td align="right">0.0284</td>
  <td align="right">122.6 MB</td>
</tr>
<tr>
  <td align="right">500</td>
  <td align="right">60.02</td>
  <td align="right">37.9%</td>
  <td align="right">15.6%</td>
  <td align="right">20.14</td>
  <td align="right">0.0243</td>
  <td align="right">450.3 MB</td>
</tr>
<tr>
  <td align="right">1000</td>
  <td align="right">60.03</td>
  <td align="right">73.8%</td>
  <td align="right">40.5%</td>
  <td align="right">50.30</td>
  <td align="right">0.0236</td>
  <td align="right">870.8 MB</td>
</tr>
<tr>
  <td align="right">1500</td>
  <td align="right">67.61</td>
  <td align="right">93.1%</td>
  <td align="right">98.6%</td>
  <td align="right">446.17</td>
  <td align="right">0.0199</td>
  <td align="right">1308.3 MB</td>
</tr>
<tr>
  <td align="right">2000</td>
  <td align="right">89.86</td>
  <td align="right">93.3%</td>
  <td align="right">99.8%</td>
  <td align="right">624.97</td>
  <td align="right">0.0149</td>
  <td align="right">1662.0 MB</td>
</tr>
<tr>
  <td align="right">2500</td>
  <td align="right">113.4</td>
  <td align="right">93.4%</td>
  <td align="right">99.8%</td>
  <td align="right">771.99</td>
  <td align="right">0.0120</td>
  <td align="right">1872.2 MB</td>
</tr>
<tr>
  <td align="right">3000</td>
  <td align="right">134.7</td>
  <td align="right">93.3%</td>
  <td align="right">99.8%</td>
  <td align="right">951.95</td>
  <td align="right">0.0100</td>
  <td align="right">2127.8 MB</td>
</tr>
<tr>
  <td rowspan="7"><code>firered_vad.onnx</code></td>
  <td align="right">100</td>
  <td align="right">60.00</td>
  <td align="right">9.73%</td>
  <td align="right">0.18%</td>
  <td align="right">7.60</td>
  <td align="right">0.0311</td>
  <td align="right">134.3 MB</td>
</tr>
<tr>
  <td align="right">500</td>
  <td align="right">60.02</td>
  <td align="right">41.2%</td>
  <td align="right">18.7%</td>
  <td align="right">24.08</td>
  <td align="right">0.0264</td>
  <td align="right">463.8 MB</td>
</tr>
<tr>
  <td align="right">1000</td>
  <td align="right">60.02</td>
  <td align="right">79.4%</td>
  <td align="right">45.6%</td>
  <td align="right">60.99</td>
  <td align="right">0.0254</td>
  <td align="right">857.5 MB</td>
</tr>
<tr>
  <td align="right">1500</td>
  <td align="right">75.18</td>
  <td align="right">94.0%</td>
  <td align="right">99.6%</td>
  <td align="right">484.87</td>
  <td align="right">0.0200</td>
  <td align="right">1281.4 MB</td>
</tr>
<tr>
  <td align="right">2000</td>
  <td align="right">99.40</td>
  <td align="right">93.3%</td>
  <td align="right">99.8%</td>
  <td align="right">668.90</td>
  <td align="right">0.0149</td>
  <td align="right">1620.5 MB</td>
</tr>
<tr>
  <td align="right">2500</td>
  <td align="right">122.5</td>
  <td align="right">94.2%</td>
  <td align="right">99.7%</td>
  <td align="right">819.70</td>
  <td align="right">0.0121</td>
  <td align="right">1960.0 MB</td>
</tr>
<tr>
  <td align="right">3000</td>
  <td align="right">148.4</td>
  <td align="right">94.4%</td>
  <td align="right">99.6%</td>
  <td align="right">984.28</td>
  <td align="right">0.0101</td>
  <td align="right">2407.6 MB</td>
</tr>
<tr>
  <td rowspan="7"><code>fsmn_vad.16k.int8.onnx</code></td>
  <td align="right">100</td>
  <td align="right">60.00</td>
  <td align="right">15.8%</td>
  <td align="right">0.12%</td>
  <td align="right">8.03</td>
  <td align="right">0.0505</td>
  <td align="right">91.48 MB</td>
</tr>
<tr>
  <td align="right">500</td>
  <td align="right">60.04</td>
  <td align="right">73.5%</td>
  <td align="right">36.9%</td>
  <td align="right">34.76</td>
  <td align="right">0.0470</td>
  <td align="right">338.6 MB</td>
</tr>
<tr>
  <td align="right">1000</td>
  <td align="right">93.40</td>
  <td align="right">94.7%</td>
  <td align="right">99.8%</td>
  <td align="right">407.71</td>
  <td align="right">0.0303</td>
  <td align="right">705.7 MB</td>
</tr>
<tr>
  <td align="right">1500</td>
  <td align="right">137.0</td>
  <td align="right">95.1%</td>
  <td align="right">99.9%</td>
  <td align="right">604.39</td>
  <td align="right">0.0203</td>
  <td align="right">989.3 MB</td>
</tr>
<tr>
  <td align="right">2000</td>
  <td align="right">187.1</td>
  <td align="right">95.1%</td>
  <td align="right">99.9%</td>
  <td align="right">864.61</td>
  <td align="right">0.0152</td>
  <td align="right">1255.8 MB</td>
</tr>
<tr>
  <td align="right">2500</td>
  <td align="right">227.4</td>
  <td align="right">95.2%</td>
  <td align="right">100.0%</td>
  <td align="right">1039.84</td>
  <td align="right">0.0122</td>
  <td align="right">1466.8 MB</td>
</tr>
<tr>
  <td align="right">3000</td>
  <td align="right">275.7</td>
  <td align="right">95.3%</td>
  <td align="right">100.0%</td>
  <td align="right">1214.50</td>
  <td align="right">0.0102</td>
  <td align="right">1630.4 MB</td>
</tr>
<tr>
  <td rowspan="7"><code>fsmn_vad.16k.onnx</code></td>
  <td align="right">100</td>
  <td align="right">60.00</td>
  <td align="right">15.8%</td>
  <td align="right">0.093%</td>
  <td align="right">8.72</td>
  <td align="right">0.0507</td>
  <td align="right">91.85 MB</td>
</tr>
<tr>
  <td align="right">500</td>
  <td align="right">60.03</td>
  <td align="right">72.4%</td>
  <td align="right">34.6%</td>
  <td align="right">33.34</td>
  <td align="right">0.0464</td>
  <td align="right">354.4 MB</td>
</tr>
<tr>
  <td align="right">1000</td>
  <td align="right">91.41</td>
  <td align="right">94.3%</td>
  <td align="right">99.7%</td>
  <td align="right">392.96</td>
  <td align="right">0.0302</td>
  <td align="right">699.9 MB</td>
</tr>
<tr>
  <td align="right">1500</td>
  <td align="right">137.7</td>
  <td align="right">95.1%</td>
  <td align="right">99.9%</td>
  <td align="right">601.46</td>
  <td align="right">0.0203</td>
  <td align="right">939.3 MB</td>
</tr>
<tr>
  <td align="right">2000</td>
  <td align="right">183.4</td>
  <td align="right">95.9%</td>
  <td align="right">100.0%</td>
  <td align="right">820.37</td>
  <td align="right">0.0153</td>
  <td align="right">1212.6 MB</td>
</tr>
<tr>
  <td align="right">2500</td>
  <td align="right">230.1</td>
  <td align="right">95.6%</td>
  <td align="right">100.0%</td>
  <td align="right">1021.74</td>
  <td align="right">0.0122</td>
  <td align="right">1441.3 MB</td>
</tr>
<tr>
  <td align="right">3000</td>
  <td align="right">274.8</td>
  <td align="right">95.6%</td>
  <td align="right">99.9%</td>
  <td align="right">1185.12</td>
  <td align="right">0.0102</td>
  <td align="right">1711.6 MB</td>
</tr>
<tr>
  <td rowspan="7"><code>fsmn_vad.8k.int8.onnx</code></td>
  <td align="right">100</td>
  <td align="right">60.00</td>
  <td align="right">13.9%</td>
  <td align="right">0.084%</td>
  <td align="right">7.78</td>
  <td align="right">0.0445</td>
  <td align="right">66.67 MB</td>
</tr>
<tr>
  <td align="right">500</td>
  <td align="right">60.03</td>
  <td align="right">62.8%</td>
  <td align="right">28.2%</td>
  <td align="right">31.00</td>
  <td align="right">0.0402</td>
  <td align="right">241.9 MB</td>
</tr>
<tr>
  <td align="right">1000</td>
  <td align="right">80.49</td>
  <td align="right">94.7%</td>
  <td align="right">99.3%</td>
  <td align="right">362.15</td>
  <td align="right">0.0303</td>
  <td align="right">471.7 MB</td>
</tr>
<tr>
  <td align="right">1500</td>
  <td align="right">132.7</td>
  <td align="right">95.5%</td>
  <td align="right">99.7%</td>
  <td align="right">587.64</td>
  <td align="right">0.0204</td>
  <td align="right">660.2 MB</td>
</tr>
<tr>
  <td align="right">2000</td>
  <td align="right">161.4</td>
  <td align="right">95.1%</td>
  <td align="right">99.8%</td>
  <td align="right">779.63</td>
  <td align="right">0.0152</td>
  <td align="right">827.2 MB</td>
</tr>
<tr>
  <td align="right">2500</td>
  <td align="right">202.8</td>
  <td align="right">95.2%</td>
  <td align="right">99.8%</td>
  <td align="right">944.68</td>
  <td align="right">0.0122</td>
  <td align="right">961.3 MB</td>
</tr>
<tr>
  <td align="right">3000</td>
  <td align="right">239.2</td>
  <td align="right">95.3%</td>
  <td align="right">99.9%</td>
  <td align="right">1120.17</td>
  <td align="right">0.0102</td>
  <td align="right">1080.1 MB</td>
</tr>
<tr>
  <td rowspan="7"><code>fsmn_vad.8k.onnx</code></td>
  <td align="right">100</td>
  <td align="right">60.00</td>
  <td align="right">11.7%</td>
  <td align="right">0.051%</td>
  <td align="right">6.12</td>
  <td align="right">0.0375</td>
  <td align="right">66.56 MB</td>
</tr>
<tr>
  <td align="right">500</td>
  <td align="right">60.03</td>
  <td align="right">60.0%</td>
  <td align="right">24.9%</td>
  <td align="right">28.06</td>
  <td align="right">0.0384</td>
  <td align="right">231.8 MB</td>
</tr>
<tr>
  <td align="right">1000</td>
  <td align="right">76.33</td>
  <td align="right">94.8%</td>
  <td align="right">99.6%</td>
  <td align="right">363.96</td>
  <td align="right">0.0303</td>
  <td align="right">454.5 MB</td>
</tr>
<tr>
  <td align="right">1500</td>
  <td align="right">116.1</td>
  <td align="right">95.4%</td>
  <td align="right">99.7%</td>
  <td align="right">559.49</td>
  <td align="right">0.0204</td>
  <td align="right">616.6 MB</td>
</tr>
<tr>
  <td align="right">2000</td>
  <td align="right">159.0</td>
  <td align="right">94.9%</td>
  <td align="right">99.8%</td>
  <td align="right">750.04</td>
  <td align="right">0.0152</td>
  <td align="right">781.0 MB</td>
</tr>
<tr>
  <td align="right">2500</td>
  <td align="right">198.6</td>
  <td align="right">95.4%</td>
  <td align="right">99.8%</td>
  <td align="right">937.75</td>
  <td align="right">0.0122</td>
  <td align="right">934.4 MB</td>
</tr>
<tr>
  <td align="right">3000</td>
  <td align="right">231.6</td>
  <td align="right">95.4%</td>
  <td align="right">99.8%</td>
  <td align="right">1122.42</td>
  <td align="right">0.0102</td>
  <td align="right">1081.7 MB</td>
</tr>
<tr>
  <td rowspan="7"><code>pulsevad_81k.int8.onnx</code></td>
  <td align="right">100</td>
  <td align="right">60.01</td>
  <td align="right">7.15%</td>
  <td align="right">0.075%</td>
  <td align="right">4.84</td>
  <td align="right">0.0229</td>
  <td align="right">46.27 MB</td>
</tr>
<tr>
  <td align="right">500</td>
  <td align="right">60.04</td>
  <td align="right">30.8%</td>
  <td align="right">10.6%</td>
  <td align="right">14.94</td>
  <td align="right">0.0197</td>
  <td align="right">181.7 MB</td>
</tr>
<tr>
  <td align="right">1000</td>
  <td align="right">60.09</td>
  <td align="right">61.5%</td>
  <td align="right">31.2%</td>
  <td align="right">35.02</td>
  <td align="right">0.0197</td>
  <td align="right">322.1 MB</td>
</tr>
<tr>
  <td align="right">1500</td>
  <td align="right">60.12</td>
  <td align="right">94.0%</td>
  <td align="right">70.5%</td>
  <td align="right">247.80</td>
  <td align="right">0.0201</td>
  <td align="right">516.5 MB</td>
</tr>
<tr>
  <td align="right">2000</td>
  <td align="right">77.29</td>
  <td align="right">94.9%</td>
  <td align="right">99.4%</td>
  <td align="right">553.34</td>
  <td align="right">0.0152</td>
  <td align="right">645.5 MB</td>
</tr>
<tr>
  <td align="right">2500</td>
  <td align="right">100.5</td>
  <td align="right">95.0%</td>
  <td align="right">99.7%</td>
  <td align="right">713.60</td>
  <td align="right">0.0122</td>
  <td align="right">718.0 MB</td>
</tr>
<tr>
  <td align="right">3000</td>
  <td align="right">122.1</td>
  <td align="right">95.2%</td>
  <td align="right">99.7%</td>
  <td align="right">868.09</td>
  <td align="right">0.0102</td>
  <td align="right">913.0 MB</td>
</tr>
<tr>
  <td rowspan="7"><code>pulsevad_81k.onnx</code></td>
  <td align="right">100</td>
  <td align="right">60.01</td>
  <td align="right">5.52%</td>
  <td align="right">0.055%</td>
  <td align="right">4.12</td>
  <td align="right">0.0177</td>
  <td align="right">42.88 MB</td>
</tr>
<tr>
  <td align="right">500</td>
  <td align="right">60.04</td>
  <td align="right">22.2%</td>
  <td align="right">3.77%</td>
  <td align="right">9.20</td>
  <td align="right">0.0142</td>
  <td align="right">153.8 MB</td>
</tr>
<tr>
  <td align="right">1000</td>
  <td align="right">60.06</td>
  <td align="right">40.9%</td>
  <td align="right">14.8%</td>
  <td align="right">14.32</td>
  <td align="right">0.0131</td>
  <td align="right">263.6 MB</td>
</tr>
<tr>
  <td align="right">1500</td>
  <td align="right">60.10</td>
  <td align="right">59.4%</td>
  <td align="right">26.8%</td>
  <td align="right">24.80</td>
  <td align="right">0.0127</td>
  <td align="right">401.2 MB</td>
</tr>
<tr>
  <td align="right">2000</td>
  <td align="right">60.12</td>
  <td align="right">75.8%</td>
  <td align="right">37.8%</td>
  <td align="right">42.24</td>
  <td align="right">0.0121</td>
  <td align="right">509.2 MB</td>
</tr>
<tr>
  <td align="right">2500</td>
  <td align="right">60.14</td>
  <td align="right">90.7%</td>
  <td align="right">61.7%</td>
  <td align="right">197.66</td>
  <td align="right">0.0116</td>
  <td align="right">652.0 MB</td>
</tr>
<tr>
  <td align="right">3000</td>
  <td align="right">67.89</td>
  <td align="right">92.8%</td>
  <td align="right">98.1%</td>
  <td align="right">575.78</td>
  <td align="right">0.0099</td>
  <td align="right">777.6 MB</td>
</tr>
<tr>
  <td rowspan="7"><code>pulsevad.int8.onnx</code></td>
  <td align="right">100</td>
  <td align="right">60.01</td>
  <td align="right">6.52%</td>
  <td align="right">0.067%</td>
  <td align="right">4.65</td>
  <td align="right">0.0209</td>
  <td align="right">46.13 MB</td>
</tr>
<tr>
  <td align="right">500</td>
  <td align="right">60.04</td>
  <td align="right">27.2%</td>
  <td align="right">8.84%</td>
  <td align="right">14.27</td>
  <td align="right">0.0174</td>
  <td align="right">175.0 MB</td>
</tr>
<tr>
  <td align="right">1000</td>
  <td align="right">60.07</td>
  <td align="right">51.0%</td>
  <td align="right">23.4%</td>
  <td align="right">27.56</td>
  <td align="right">0.0163</td>
  <td align="right">315.1 MB</td>
</tr>
<tr>
  <td align="right">1500</td>
  <td align="right">60.11</td>
  <td align="right">79.9%</td>
  <td align="right">42.7%</td>
  <td align="right">52.06</td>
  <td align="right">0.0171</td>
  <td align="right">477.7 MB</td>
</tr>
<tr>
  <td align="right">2000</td>
  <td align="right">64.63</td>
  <td align="right">94.1%</td>
  <td align="right">97.7%</td>
  <td align="right">467.04</td>
  <td align="right">0.0151</td>
  <td align="right">631.2 MB</td>
</tr>
<tr>
  <td align="right">2500</td>
  <td align="right">81.97</td>
  <td align="right">94.5%</td>
  <td align="right">99.6%</td>
  <td align="right">599.61</td>
  <td align="right">0.0121</td>
  <td align="right">699.2 MB</td>
</tr>
<tr>
  <td align="right">3000</td>
  <td align="right">96.96</td>
  <td align="right">94.7%</td>
  <td align="right">99.6%</td>
  <td align="right">707.08</td>
  <td align="right">0.0101</td>
  <td align="right">804.7 MB</td>
</tr>
<tr>
  <td rowspan="7"><code>pulsevad.onnx</code></td>
  <td align="right">100</td>
  <td align="right">60.01</td>
  <td align="right">4.81%</td>
  <td align="right">0.033%</td>
  <td align="right">3.47</td>
  <td align="right">0.0154</td>
  <td align="right">38.84 MB</td>
</tr>
<tr>
  <td align="right">500</td>
  <td align="right">60.03</td>
  <td align="right">19.3%</td>
  <td align="right">1.06%</td>
  <td align="right">8.43</td>
  <td align="right">0.0124</td>
  <td align="right">142.4 MB</td>
</tr>
<tr>
  <td align="right">1000</td>
  <td align="right">60.06</td>
  <td align="right">35.0%</td>
  <td align="right">11.1%</td>
  <td align="right">11.28</td>
  <td align="right">0.0112</td>
  <td align="right">260.3 MB</td>
</tr>
<tr>
  <td align="right">1500</td>
  <td align="right">60.08</td>
  <td align="right">50.8%</td>
  <td align="right">21.1%</td>
  <td align="right">18.17</td>
  <td align="right">0.0108</td>
  <td align="right">368.0 MB</td>
</tr>
<tr>
  <td align="right">2000</td>
  <td align="right">60.11</td>
  <td align="right">65.4%</td>
  <td align="right">30.7%</td>
  <td align="right">29.16</td>
  <td align="right">0.0105</td>
  <td align="right">435.8 MB</td>
</tr>
<tr>
  <td align="right">2500</td>
  <td align="right">60.14</td>
  <td align="right">80.7%</td>
  <td align="right">42.2%</td>
  <td align="right">56.95</td>
  <td align="right">0.0103</td>
  <td align="right">608.8 MB</td>
</tr>
<tr>
  <td align="right">3000</td>
  <td align="right">60.20</td>
  <td align="right">91.4%</td>
  <td align="right">76.6%</td>
  <td align="right">294.96</td>
  <td align="right">0.0097</td>
  <td align="right">637.1 MB</td>
</tr>
<tr>
  <td rowspan="7"><code>silero_vad_16k_op15.v6.onnx</code></td>
  <td align="right">100</td>
  <td align="right">60.01</td>
  <td align="right">6.96%</td>
  <td align="right">0.000%</td>
  <td align="right">2.62</td>
  <td align="right">0.0223</td>
  <td align="right">32.86 MB</td>
</tr>
<tr>
  <td align="right">500</td>
  <td align="right">60.01</td>
  <td align="right">30.8%</td>
  <td align="right">0.018%</td>
  <td align="right">8.11</td>
  <td align="right">0.0197</td>
  <td align="right">63.20 MB</td>
</tr>
<tr>
  <td align="right">1000</td>
  <td align="right">60.03</td>
  <td align="right">57.1%</td>
  <td align="right">7.72%</td>
  <td align="right">26.96</td>
  <td align="right">0.0183</td>
  <td align="right">100.5 MB</td>
</tr>
<tr>
  <td align="right">1500</td>
  <td align="right">60.04</td>
  <td align="right">79.6%</td>
  <td align="right">41.0%</td>
  <td align="right">81.43</td>
  <td align="right">0.0170</td>
  <td align="right">148.2 MB</td>
</tr>
<tr>
  <td align="right">2000</td>
  <td align="right">70.82</td>
  <td align="right">86.0%</td>
  <td align="right">100.0%</td>
  <td align="right">275.45</td>
  <td align="right">0.0138</td>
  <td align="right">171.4 MB</td>
</tr>
<tr>
  <td align="right">2500</td>
  <td align="right">86.58</td>
  <td align="right">84.4%</td>
  <td align="right">100.0%</td>
  <td align="right">332.00</td>
  <td align="right">0.0108</td>
  <td align="right">223.9 MB</td>
</tr>
<tr>
  <td align="right">3000</td>
  <td align="right">104.8</td>
  <td align="right">85.2%</td>
  <td align="right">100.0%</td>
  <td align="right">402.68</td>
  <td align="right">0.0091</td>
  <td align="right">264.7 MB</td>
</tr>
<tr>
  <td rowspan="7"><code>silero_vad.v4.onnx</code></td>
  <td align="right">100</td>
  <td align="right">60.01</td>
  <td align="right">9.77%</td>
  <td align="right">0.019%</td>
  <td align="right">3.52</td>
  <td align="right">0.0313</td>
  <td align="right">41.42 MB</td>
</tr>
<tr>
  <td align="right">500</td>
  <td align="right">60.02</td>
  <td align="right">42.3%</td>
  <td align="right">0.56%</td>
  <td align="right">10.15</td>
  <td align="right">0.0270</td>
  <td align="right">108.3 MB</td>
</tr>
<tr>
  <td align="right">1000</td>
  <td align="right">60.05</td>
  <td align="right">79.2%</td>
  <td align="right">28.2%</td>
  <td align="right">42.43</td>
  <td align="right">0.0253</td>
  <td align="right">160.9 MB</td>
</tr>
<tr>
  <td align="right">1500</td>
  <td align="right">73.02</td>
  <td align="right">91.7%</td>
  <td align="right">100.0%</td>
  <td align="right">293.47</td>
  <td align="right">0.0196</td>
  <td align="right">211.0 MB</td>
</tr>
<tr>
  <td align="right">2000</td>
  <td align="right">96.00</td>
  <td align="right">90.8%</td>
  <td align="right">100.0%</td>
  <td align="right">374.12</td>
  <td align="right">0.0145</td>
  <td align="right">266.3 MB</td>
</tr>
<tr>
  <td align="right">2500</td>
  <td align="right">120.7</td>
  <td align="right">91.1%</td>
  <td align="right">100.0%</td>
  <td align="right">466.45</td>
  <td align="right">0.0117</td>
  <td align="right">312.7 MB</td>
</tr>
<tr>
  <td align="right">3000</td>
  <td align="right">146.2</td>
  <td align="right">91.2%</td>
  <td align="right">100.0%</td>
  <td align="right">565.25</td>
  <td align="right">0.0097</td>
  <td align="right">377.4 MB</td>
</tr>
<tr>
  <td rowspan="7"><code>silero_vad.v5.onnx</code></td>
  <td align="right">100</td>
  <td align="right">60.00</td>
  <td align="right">6.62%</td>
  <td align="right">0.000%</td>
  <td align="right">2.54</td>
  <td align="right">0.0212</td>
  <td align="right">36.85 MB</td>
</tr>
<tr>
  <td align="right">500</td>
  <td align="right">60.01</td>
  <td align="right">28.4%</td>
  <td align="right">0.015%</td>
  <td align="right">7.43</td>
  <td align="right">0.0182</td>
  <td align="right">72.88 MB</td>
</tr>
<tr>
  <td align="right">1000</td>
  <td align="right">60.03</td>
  <td align="right">53.5%</td>
  <td align="right">5.57%</td>
  <td align="right">23.55</td>
  <td align="right">0.0171</td>
  <td align="right">111.6 MB</td>
</tr>
<tr>
  <td align="right">1500</td>
  <td align="right">60.03</td>
  <td align="right">76.8%</td>
  <td align="right">41.4%</td>
  <td align="right">81.83</td>
  <td align="right">0.0164</td>
  <td align="right">148.9 MB</td>
</tr>
<tr>
  <td align="right">2000</td>
  <td align="right">66.51</td>
  <td align="right">81.4%</td>
  <td align="right">99.8%</td>
  <td align="right">245.97</td>
  <td align="right">0.0130</td>
  <td align="right">183.4 MB</td>
</tr>
<tr>
  <td align="right">2500</td>
  <td align="right">83.49</td>
  <td align="right">82.6%</td>
  <td align="right">100.0%</td>
  <td align="right">317.49</td>
  <td align="right">0.0106</td>
  <td align="right">234.9 MB</td>
</tr>
<tr>
  <td align="right">3000</td>
  <td align="right">99.31</td>
  <td align="right">82.0%</td>
  <td align="right">100.0%</td>
  <td align="right">375.83</td>
  <td align="right">0.0087</td>
  <td align="right">278.0 MB</td>
</tr>
<tr>
  <td rowspan="7"><code>silero_vad.v6.onnx</code></td>
  <td align="right">100</td>
  <td align="right">60.00</td>
  <td align="right">7.49%</td>
  <td align="right">0.000%</td>
  <td align="right">2.80</td>
  <td align="right">0.0240</td>
  <td align="right">43.16 MB</td>
</tr>
<tr>
  <td align="right">500</td>
  <td align="right">60.01</td>
  <td align="right">31.2%</td>
  <td align="right">0.004%</td>
  <td align="right">8.17</td>
  <td align="right">0.0200</td>
  <td align="right">73.20 MB</td>
</tr>
<tr>
  <td align="right">1000</td>
  <td align="right">60.02</td>
  <td align="right">60.9%</td>
  <td align="right">13.9%</td>
  <td align="right">38.10</td>
  <td align="right">0.0195</td>
  <td align="right">117.6 MB</td>
</tr>
<tr>
  <td align="right">1500</td>
  <td align="right">60.04</td>
  <td align="right">87.0%</td>
  <td align="right">55.4%</td>
  <td align="right">112.33</td>
  <td align="right">0.0186</td>
  <td align="right">159.5 MB</td>
</tr>
<tr>
  <td align="right">2000</td>
  <td align="right">75.35</td>
  <td align="right">88.2%</td>
  <td align="right">100.0%</td>
  <td align="right">293.72</td>
  <td align="right">0.0141</td>
  <td align="right">185.6 MB</td>
</tr>
<tr>
  <td align="right">2500</td>
  <td align="right">91.72</td>
  <td align="right">87.8%</td>
  <td align="right">100.0%</td>
  <td align="right">355.92</td>
  <td align="right">0.0112</td>
  <td align="right">237.7 MB</td>
</tr>
<tr>
  <td align="right">3000</td>
  <td align="right">112.5</td>
  <td align="right">88.3%</td>
  <td align="right">100.0%</td>
  <td align="right">442.25</td>
  <td align="right">0.0094</td>
  <td align="right">277.3 MB</td>
</tr>
<tr>
  <td rowspan="7"><code>ten_vad.int8.onnx</code></td>
  <td align="right">100</td>
  <td align="right">60.01</td>
  <td align="right">24.3%</td>
  <td align="right">0.000%</td>
  <td align="right">8.46</td>
  <td align="right">0.0777</td>
  <td align="right">31.79 MB</td>
</tr>
<tr>
  <td align="right">500</td>
  <td align="right">73.96</td>
  <td align="right">95.1%</td>
  <td align="right">99.8%</td>
  <td align="right">164.00</td>
  <td align="right">0.0609</td>
  <td align="right">96.63 MB</td>
</tr>
<tr>
  <td align="right">1000</td>
  <td align="right">148.3</td>
  <td align="right">96.1%</td>
  <td align="right">99.9%</td>
  <td align="right">347.72</td>
  <td align="right">0.0307</td>
  <td align="right">168.4 MB</td>
</tr>
<tr>
  <td align="right">1500</td>
  <td align="right">225.7</td>
  <td align="right">95.8%</td>
  <td align="right">100.0%</td>
  <td align="right">538.42</td>
  <td align="right">0.0204</td>
  <td align="right">226.6 MB</td>
</tr>
<tr>
  <td align="right">2000</td>
  <td align="right">301.7</td>
  <td align="right">95.6%</td>
  <td align="right">100.0%</td>
  <td align="right">723.58</td>
  <td align="right">0.0153</td>
  <td align="right">290.4 MB</td>
</tr>
<tr>
  <td align="right">2500</td>
  <td align="right">381.4</td>
  <td align="right">95.7%</td>
  <td align="right">100.0%</td>
  <td align="right">927.10</td>
  <td align="right">0.0122</td>
  <td align="right">355.7 MB</td>
</tr>
<tr>
  <td align="right">3000</td>
  <td align="right">503.1</td>
  <td align="right">96.4%</td>
  <td align="right">100.0%</td>
  <td align="right">1192.82</td>
  <td align="right">0.0103</td>
  <td align="right">389.3 MB</td>
</tr>
<tr>
  <td rowspan="7"><code>ten_vad.onnx</code></td>
  <td align="right">100</td>
  <td align="right">60.01</td>
  <td align="right">24.1%</td>
  <td align="right">0.000%</td>
  <td align="right">8.25</td>
  <td align="right">0.0772</td>
  <td align="right">29.04 MB</td>
</tr>
<tr>
  <td align="right">500</td>
  <td align="right">73.06</td>
  <td align="right">94.9%</td>
  <td align="right">99.7%</td>
  <td align="right">165.28</td>
  <td align="right">0.0608</td>
  <td align="right">82.59 MB</td>
</tr>
<tr>
  <td align="right">1000</td>
  <td align="right">141.4</td>
  <td align="right">95.6%</td>
  <td align="right">100.0%</td>
  <td align="right">329.42</td>
  <td align="right">0.0306</td>
  <td align="right">139.4 MB</td>
</tr>
<tr>
  <td align="right">1500</td>
  <td align="right">214.7</td>
  <td align="right">95.8%</td>
  <td align="right">100.0%</td>
  <td align="right">506.97</td>
  <td align="right">0.0204</td>
  <td align="right">194.7 MB</td>
</tr>
<tr>
  <td align="right">2000</td>
  <td align="right">290.4</td>
  <td align="right">96.1%</td>
  <td align="right">100.0%</td>
  <td align="right">694.55</td>
  <td align="right">0.0154</td>
  <td align="right">255.1 MB</td>
</tr>
<tr>
  <td align="right">2500</td>
  <td align="right">368.0</td>
  <td align="right">96.1%</td>
  <td align="right">100.0%</td>
  <td align="right">880.69</td>
  <td align="right">0.0123</td>
  <td align="right">314.0 MB</td>
</tr>
<tr>
  <td align="right">3000</td>
  <td align="right">416.3</td>
  <td align="right">96.2%</td>
  <td align="right">100.0%</td>
  <td align="right">1016.14</td>
  <td align="right">0.0103</td>
  <td align="right">371.7 MB</td>
</tr>
</tbody>
</table>
