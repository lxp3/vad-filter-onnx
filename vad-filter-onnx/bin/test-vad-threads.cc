#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "resample.h"
#include "vad-config.h"
#include "vad-filter-onnx-cxx-api.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <psapi.h>
#include <windows.h>
#else
#include <sys/resource.h>
#endif

namespace {

struct Options {
    std::string model_path;
    int input_sample_rate = 8000;
    int vad_sample_rate = -1;
    int packet_ms = 20;
    int seconds = 30;
    int concurrency = 1;
    VadFilterOnnx::VadConfig config;
};

struct CpuSnapshot {
    double user_seconds = 0.0;
    double sys_seconds = 0.0;
};

struct MemorySnapshot {
    long rss_kb = 0;
    long rss_peak_kb = 0;
    long vmsize_kb = 0;
    long rss_anon_kb = 0;
    long rss_file_kb = 0;
};

struct WorkerResult {
    uint64_t packets = 0;
    uint64_t xruns = 0;
    std::vector<double> latencies_ms;
    std::string error;
};

class CountdownLatch {
  public:
    explicit CountdownLatch(int count) : count_(count) {}

    void count_down() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (count_ <= 0) {
            return;
        }
        --count_;
        if (count_ == 0) {
            cv_.notify_all();
        }
    }

    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return count_ <= 0; });
    }

  private:
    std::mutex mutex_;
    std::condition_variable cv_;
    int count_;
};

void PrintUsage(const char *program) {
    std::cerr << "Usage: " << program << " --model-path PATH [options]\n\n"
              << "Simulate FreeSWITCH-style streaming VAD: one OS thread per call,\n"
              << "one shared AutoVadModel handle, and one instance per thread.\n"
              << "Each thread feeds random PCM int16 at --packet-ms in real time.\n\n"
              << "Options:\n"
              << "  --model-path PATH           ONNX model path, or webrtc (required)\n"
              << "  --input-sample-rate N       input PCM sample rate (default: 8000)\n"
              << "  --vad-sample-rate N         VAD sample rate (default: 8000 for webrtc,\n"
              << "                              16000 otherwise)\n"
              << "  --packet-ms N               packet duration in milliseconds (default: 20)\n"
              << "  --seconds N                 simulated audio seconds per call (default: 30)\n"
              << "  --concurrency N             number of call threads (default: 1)\n"
              << "  --threshold N               VAD threshold (default: 0.4)\n"
              << "  --webrtc-vad-mode N         WebRTC VAD mode 0-3 (default: 3)\n"
              << "  --webrtc-frame-ms N         WebRTC VAD frame ms 10/20/30 (default: 30)\n";
}

int ParseInteger(const std::string &value, const std::string &option) {
    std::size_t parsed = 0;
    int result = 0;
    try {
        result = std::stoi(value, &parsed);
    } catch (const std::exception &) {
        throw std::invalid_argument(option + " requires an integer");
    }
    if (parsed != value.size()) {
        throw std::invalid_argument(option + " requires an integer");
    }
    return result;
}

float ParseFloat(const std::string &value, const std::string &option) {
    std::size_t parsed = 0;
    float result = 0.0F;
    try {
        result = std::stof(value, &parsed);
    } catch (const std::exception &) {
        throw std::invalid_argument(option + " requires a number");
    }
    if (parsed != value.size()) {
        throw std::invalid_argument(option + " requires a number");
    }
    return result;
}

Options ParseArgs(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            PrintUsage(argv[0]);
            std::exit(0);
        }
        if (++i >= argc) {
            throw std::invalid_argument("missing value for " + arg);
        }
        if (arg == "--model-path") {
            options.model_path = argv[i];
        } else if (arg == "--input-sample-rate") {
            options.input_sample_rate = ParseInteger(argv[i], arg);
        } else if (arg == "--vad-sample-rate") {
            options.vad_sample_rate = ParseInteger(argv[i], arg);
        } else if (arg == "--packet-ms") {
            options.packet_ms = ParseInteger(argv[i], arg);
        } else if (arg == "--seconds") {
            options.seconds = ParseInteger(argv[i], arg);
        } else if (arg == "--concurrency") {
            options.concurrency = ParseInteger(argv[i], arg);
        } else if (arg == "--threshold") {
            options.config.threshold = ParseFloat(argv[i], arg);
        } else if (arg == "--webrtc-vad-mode") {
            options.config.webrtc_vad_mode = ParseInteger(argv[i], arg);
        } else if (arg == "--webrtc-frame-ms") {
            options.config.webrtc_frame_ms = ParseInteger(argv[i], arg);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    if (options.model_path.empty()) {
        throw std::invalid_argument("--model-path is required");
    }
    if (options.input_sample_rate <= 0) {
        throw std::invalid_argument("--input-sample-rate must be greater than 0");
    }
    if (options.vad_sample_rate == -1) {
        options.vad_sample_rate = options.model_path == "webrtc" ? 8000 : 16000;
    }
    if (options.vad_sample_rate <= 0) {
        throw std::invalid_argument("--vad-sample-rate must be greater than 0");
    }
    if (options.packet_ms <= 0) {
        throw std::invalid_argument("--packet-ms must be greater than 0");
    }
    if (options.seconds <= 0) {
        throw std::invalid_argument("--seconds must be greater than 0");
    }
    if (options.concurrency <= 0) {
        throw std::invalid_argument("--concurrency must be greater than 0");
    }
    if (static_cast<int64_t>(options.input_sample_rate) * options.packet_ms % 1000 != 0) {
        throw std::invalid_argument("--packet-ms is not aligned with --input-sample-rate");
    }
    options.config.sample_rate = options.vad_sample_rate;
    return options;
}

CpuSnapshot ReadCpu() {
    CpuSnapshot snapshot;
#if defined(_WIN32)
    FILETIME creation_time{};
    FILETIME exit_time{};
    FILETIME kernel_time{};
    FILETIME user_time{};
    if (!GetProcessTimes(GetCurrentProcess(), &creation_time, &exit_time, &kernel_time,
                         &user_time)) {
        throw std::runtime_error("GetProcessTimes failed");
    }
    auto to_seconds = [](FILETIME time) {
        ULARGE_INTEGER value;
        value.LowPart = time.dwLowDateTime;
        value.HighPart = time.dwHighDateTime;
        return static_cast<double>(value.QuadPart) / 10000000.0;
    };
    snapshot.user_seconds = to_seconds(user_time);
    snapshot.sys_seconds = to_seconds(kernel_time);
#else
    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        throw std::runtime_error("getrusage failed");
    }
    snapshot.user_seconds = usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1e6;
    snapshot.sys_seconds = usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1e6;
#endif
    return snapshot;
}

MemorySnapshot ReadMemory() {
    MemorySnapshot snapshot;
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (!GetProcessMemoryInfo(GetCurrentProcess(),
                              reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&counters),
                              sizeof(counters))) {
        throw std::runtime_error("GetProcessMemoryInfo failed");
    }
    snapshot.rss_kb = static_cast<long>(counters.WorkingSetSize / 1024);
    snapshot.rss_peak_kb = static_cast<long>(counters.PeakWorkingSetSize / 1024);
    snapshot.vmsize_kb = static_cast<long>(counters.PeakPagefileUsage / 1024);
    snapshot.rss_anon_kb = static_cast<long>(counters.PrivateUsage / 1024);
#elif defined(__linux__)
    std::ifstream status("/proc/self/status");
    if (!status) {
        throw std::runtime_error("failed to read /proc/self/status");
    }
    auto parse_kb = [](const std::string &line, const char *key, long *value) {
        const std::size_t key_len = std::strlen(key);
        if (line.compare(0, key_len, key) != 0) {
            return;
        }
        std::size_t pos = key_len;
        while (pos < line.size() && (line[pos] == ':' || line[pos] == ' ' || line[pos] == '\t')) {
            ++pos;
        }
        *value = std::strtol(line.c_str() + static_cast<std::ptrdiff_t>(pos), nullptr, 10);
    };
    std::string line;
    while (std::getline(status, line)) {
        parse_kb(line, "VmRSS", &snapshot.rss_kb);
        parse_kb(line, "VmHWM", &snapshot.rss_peak_kb);
        parse_kb(line, "VmSize", &snapshot.vmsize_kb);
        parse_kb(line, "RssAnon", &snapshot.rss_anon_kb);
        parse_kb(line, "RssFile", &snapshot.rss_file_kb);
    }
#else
    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        throw std::runtime_error("getrusage failed");
    }
#if defined(__APPLE__)
    snapshot.rss_peak_kb = static_cast<long>(usage.ru_maxrss / 1024);
#else
    snapshot.rss_peak_kb = static_cast<long>(usage.ru_maxrss);
#endif
    snapshot.rss_kb = snapshot.rss_peak_kb;
#endif
    return snapshot;
}

std::string FormatMemKb(long kb) {
    std::ostringstream out;
    out << kb << " KB";
    if (kb >= 1024) {
        const double mb = static_cast<double>(kb) / 1024.0;
        out << std::fixed << std::setprecision(mb >= 100.0 ? 1 : 2) << " / " << mb << " MB";
    }
    return out.str();
}

std::unique_ptr<sherpa_onnx::LinearResample> MakeResampler(int input_sample_rate,
                                                           int output_sample_rate) {
    if (input_sample_rate == output_sample_rate) {
        return nullptr;
    }
    const float min_freq = static_cast<float>(std::min(input_sample_rate, output_sample_rate));
    const float lowpass_cutoff = 0.99F * 0.5F * min_freq;
    return std::make_unique<sherpa_onnx::LinearResample>(input_sample_rate, output_sample_rate,
                                                         lowpass_cutoff, 6);
}

void FillRandomPcm(std::mt19937 *rng, std::uniform_int_distribution<int> *distribution,
                   std::vector<int16_t> *pcm, std::vector<float> *samples) {
    constexpr float kScale = 1.0F / 32768.0F;
    for (std::size_t i = 0; i < pcm->size(); ++i) {
        const int16_t value = static_cast<int16_t>((*distribution)(*rng));
        (*pcm)[i] = value;
        (*samples)[i] = static_cast<float>(value) * kScale;
    }
}

void RunWorker(int worker_id, VadFilterOnnx::AutoVadModel *handle, const Options &options,
               int num_packets, CountdownLatch *ready, CountdownLatch *start_latch,
               const std::chrono::steady_clock::time_point *start_tp,
               const std::atomic<bool> *abort, WorkerResult *result) {
    bool signaled_ready = false;
    try {
        auto instance = handle->init(options.config);
        if (!instance) {
            throw std::runtime_error("failed to initialize VAD instance");
        }

        auto resampler = MakeResampler(options.input_sample_rate, options.vad_sample_rate);
        const int packet_samples = options.input_sample_rate * options.packet_ms / 1000;
        std::vector<int16_t> pcm(static_cast<std::size_t>(packet_samples));
        std::vector<float> samples(static_cast<std::size_t>(packet_samples));
        std::vector<float> resampled;
        std::mt19937 rng(20260915U + static_cast<unsigned>(worker_id) * 9973U);
        std::uniform_int_distribution<int> distribution(-32768, 32767);
        result->latencies_ms.reserve(static_cast<std::size_t>(num_packets));

        signaled_ready = true;
        ready->count_down();
        start_latch->wait();
        if (abort->load(std::memory_order_acquire)) {
            return;
        }

        const auto period = std::chrono::milliseconds(options.packet_ms);
        auto next_deadline = *start_tp;
        for (int packet = 0; packet < num_packets; ++packet) {
            next_deadline += period;
            const auto packet_start = std::chrono::steady_clock::now();

            FillRandomPcm(&rng, &distribution, &pcm, &samples);
            float *decode_data = samples.data();
            int decode_n = packet_samples;
            if (resampler) {
                resampler->Resample(samples.data(), packet_samples, false, &resampled);
                decode_data = resampled.data();
                decode_n = static_cast<int>(resampled.size());
            }
            if (decode_n > 0) {
                instance->decode(decode_data, decode_n, false);
            }

            const auto packet_end = std::chrono::steady_clock::now();
            result->latencies_ms.push_back(
                std::chrono::duration<double, std::milli>(packet_end - packet_start).count());
            ++result->packets;
            if (packet_end > next_deadline) {
                ++result->xruns;
            } else {
                std::this_thread::sleep_until(next_deadline);
            }
        }

        if (resampler) {
            resampler->Resample(samples.data(), 0, true, &resampled);
            if (!resampled.empty()) {
                instance->decode(resampled.data(), static_cast<int>(resampled.size()), false);
            }
        }
        instance->flush();
    } catch (const std::exception &error) {
        result->error = error.what();
        if (!signaled_ready) {
            ready->count_down();
        }
    } catch (...) {
        result->error = "unknown worker error";
        if (!signaled_ready) {
            ready->count_down();
        }
    }
}

std::string FormatPercent(double percent) {
    std::ostringstream out;
    out << std::fixed;
    const double magnitude = std::fabs(percent);
    if (magnitude >= 10.0) {
        out << std::setprecision(1);
    } else if (magnitude >= 1.0) {
        out << std::setprecision(2);
    } else if (magnitude >= 0.1) {
        out << std::setprecision(2);
    } else {
        out << std::setprecision(3);
    }
    out << percent << '%';
    return out.str();
}

double Percentile(std::vector<double> values, double q) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double index = q * static_cast<double>(values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(index));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(index));
    if (lower == upper) {
        return values[lower];
    }
    const double weight = index - static_cast<double>(lower);
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Options options = ParseArgs(argc, argv);
        const int num_packets = options.seconds * 1000 / options.packet_ms;
        if (num_packets <= 0) {
            throw std::invalid_argument("--seconds is too small for --packet-ms");
        }

        std::unique_ptr<VadFilterOnnx::AutoVadModel> handle;
        if (options.model_path == "webrtc") {
            handle = VadFilterOnnx::AutoVadModel::create_webrtc();
        } else {
            handle = VadFilterOnnx::AutoVadModel::create(options.model_path, 1);
        }
        if (!handle) {
            throw std::runtime_error("failed to load model: " + options.model_path);
        }
        const MemorySnapshot mem_after_load = ReadMemory();

        CountdownLatch ready(options.concurrency);
        CountdownLatch start_latch(1);
        std::chrono::steady_clock::time_point start_tp;
        std::atomic<bool> abort{ false };
        std::vector<WorkerResult> results(static_cast<std::size_t>(options.concurrency));
        std::vector<std::thread> workers;
        workers.reserve(static_cast<std::size_t>(options.concurrency));

        for (int i = 0; i < options.concurrency; ++i) {
            workers.emplace_back([&, i] {
                RunWorker(i, handle.get(), options, num_packets, &ready, &start_latch, &start_tp,
                          &abort, &results[static_cast<std::size_t>(i)]);
            });
        }

        ready.wait();
        std::string init_error;
        for (const auto &result : results) {
            if (!result.error.empty()) {
                init_error = result.error;
                break;
            }
        }
        if (!init_error.empty()) {
            abort.store(true, std::memory_order_release);
            start_latch.count_down();
            for (auto &worker : workers) {
                worker.join();
            }
            throw std::runtime_error(init_error);
        }

        const MemorySnapshot mem_after_init = ReadMemory();
        const CpuSnapshot cpu_before = ReadCpu();
        start_tp = std::chrono::steady_clock::now();
        start_latch.count_down();
        for (auto &worker : workers) {
            worker.join();
        }
        const auto wall_end = std::chrono::steady_clock::now();
        const CpuSnapshot cpu_after = ReadCpu();
        const MemorySnapshot mem_after_run = ReadMemory();

        for (const auto &result : results) {
            if (!result.error.empty()) {
                throw std::runtime_error(result.error);
            }
        }

        uint64_t packets = 0;
        uint64_t xruns = 0;
        std::vector<double> latencies_ms;
        latencies_ms.reserve(static_cast<std::size_t>(options.concurrency) *
                             static_cast<std::size_t>(num_packets));
        for (const auto &result : results) {
            packets += result.packets;
            xruns += result.xruns;
            latencies_ms.insert(latencies_ms.end(), result.latencies_ms.begin(),
                                result.latencies_ms.end());
        }

        const double wall_seconds = std::chrono::duration<double>(wall_end - start_tp).count();
        const double cpu_user = std::max(0.0, cpu_after.user_seconds - cpu_before.user_seconds);
        const double cpu_sys = std::max(0.0, cpu_after.sys_seconds - cpu_before.sys_seconds);
        const double cpu_seconds = cpu_user + cpu_sys;
        const unsigned cpu_cores = std::max(1U, std::thread::hardware_concurrency());
        const double cpu_avg = wall_seconds > 0.0 ? cpu_seconds / wall_seconds * 100.0 : 0.0;
        const double cpu_avg_machine = cpu_avg / static_cast<double>(cpu_cores);
        const double cpu_avg_per_call =
            options.concurrency > 0 ? cpu_avg / static_cast<double>(options.concurrency) : 0.0;
        const double latency_avg =
            latencies_ms.empty() ? 0.0
                                 : std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0) /
                                       static_cast<double>(latencies_ms.size());
        const double latency_p99 = Percentile(latencies_ms, 0.99);
        const double audio_seconds = static_cast<double>(options.concurrency) * wall_seconds;
        const double aggregate_rtf = audio_seconds > 0.0 ? cpu_seconds / audio_seconds : 0.0;
        const double xrun_percent =
            packets > 0 ? static_cast<double>(xruns) / static_cast<double>(packets) * 100.0 : 0.0;
        const double rss_per_call_kb =
            options.concurrency > 0
                ? std::max(0.0, static_cast<double>(mem_after_init.rss_kb - mem_after_load.rss_kb) /
                                    static_cast<double>(options.concurrency))
                : 0.0;

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "model: " << options.model_path << " (VAD模型)\n";
        std::cout << "concurrency: " << options.concurrency << " (并发通话路数)\n";
        std::cout << "input_sample_rate: " << options.input_sample_rate << " (输入PCM采样率, Hz)\n";
        std::cout << "vad_sample_rate: " << options.vad_sample_rate << " (VAD工作采样率, Hz)\n";
        std::cout << "packet_ms: " << options.packet_ms << " (每包时长, 毫秒)\n";
        std::cout << "seconds: " << options.seconds << " (每路模拟音频时长, 秒)\n";
        std::cout << "ort_num_threads: 1 (ONNX Runtime线程数, 固定为1)\n";
        std::cout << "cpu_cores: " << cpu_cores << " (本机逻辑CPU核数)\n";
        std::cout << "wall_seconds: " << wall_seconds << " (实际墙钟时间, 秒)\n";
        std::cout << "cpu_user_seconds: " << cpu_user << " (进程用户态CPU时间, 秒)\n";
        std::cout << "cpu_sys_seconds: " << cpu_sys << " (进程内核态CPU时间, 秒)\n";
        std::cout << "cpu_seconds: " << cpu_seconds << " (进程CPU总时间=user+sys, 秒)\n";
        std::cout << "cpu_avg: " << FormatPercent(cpu_avg) << " (进程平均CPU利用率, 相对1核)\n";
        std::cout << "cpu_avg_machine: " << FormatPercent(cpu_avg_machine)
                  << " (进程平均CPU利用率, 相对整机)\n";
        std::cout << "cpu_avg_per_call: " << FormatPercent(cpu_avg_per_call)
                  << " (平均每路CPU利用率, 相对1核)\n";
        std::cout << "rss: " << FormatMemKb(mem_after_run.rss_kb) << " (当前物理内存RSS)\n";
        std::cout << "rss_peak: " << FormatMemKb(mem_after_run.rss_peak_kb) << " (峰值物理内存)\n";
        std::cout << "vmsize: " << FormatMemKb(mem_after_run.vmsize_kb) << " (虚拟内存VmSize)\n";
        std::cout << "rss_anon: " << FormatMemKb(mem_after_run.rss_anon_kb)
                  << " (匿名页, 堆/栈/每路状态)\n";
        std::cout << "rss_file: " << FormatMemKb(mem_after_run.rss_file_kb)
                  << " (文件页, 共享库/模型mmap)\n";
        std::cout << "rss_after_load: " << FormatMemKb(mem_after_load.rss_kb)
                  << " (加载模型后RSS)\n";
        std::cout << "rss_after_init: " << FormatMemKb(mem_after_init.rss_kb)
                  << " (全部instance初始化后RSS)\n";
        std::cout << "rss_per_call: " << std::setprecision(1) << rss_per_call_kb
                  << " KB (平均每路增量RSS, 相对加载模型后)\n";
        std::cout << std::setprecision(6);
        std::cout << "packets: " << packets << " (处理的音频包总数)\n";
        std::cout << "xruns: " << xruns << " (超时未在时限内处理完的包数)\n";
        std::cout << "xrun_percent: " << FormatPercent(xrun_percent) << " (超时包占比)\n";
        std::cout << "packet_latency_avg_ms: " << latency_avg << " (单包处理时延均值, 毫秒)\n";
        std::cout << "packet_latency_p99_ms: " << latency_p99 << " (单包处理时延P99, 毫秒)\n";
        std::cout << "aggregate_rtf: " << aggregate_rtf << " (聚合实时率=CPU时间/(路数x墙钟))\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        PrintUsage(argv[0]);
        return 1;
    }
}
