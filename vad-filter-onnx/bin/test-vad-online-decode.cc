#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <algorithm>
#include <memory>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include "vad-filter-onnx-cxx-api.h"
#include "vad-config.h"
#include "resample.h"


constexpr std::size_t WAV_HEADER_SIZE = 44;
using namespace VadFilterOnnx;

static void print_usage(char **argv) {
    fprintf(stderr, "Usage: %s [options]\n\n", argv[0]);
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            print this help message and exit\n");
    fprintf(stderr, "  --model-path PATH     path to ONNX model (required)\n");
    fprintf(stderr, "  --wav-path PATH       path to input WAV file (required)\n");
    fprintf(stderr, "  --sample-rate RATE    input WAV sample rate (default: 16000)\n");
    fprintf(stderr, "  --threshold THR       VAD threshold (default: 0.4)\n");
    fprintf(stderr, "  --chunk-size-ms MS    chunk size in milliseconds (default: 100)\n");
    fprintf(stderr, "  --speech-win-size-ms MS   speech detection window size (default: 300)\n");
    fprintf(stderr, "  --speech-win-thr-ms MS    speech detection threshold (default: 250)\n");
    fprintf(stderr, "  --silence-win-size-ms MS  silence detection window size (default: 600)\n");
    fprintf(stderr, "  --silence-win-thr-ms MS   silence detection threshold (default: 500)\n");
    fprintf(stderr, "  --max-speech-ms MS    max speech duration in milliseconds (default: 10000)\n");
    fprintf(stderr, "  --left-padding-ms MS  left padding in milliseconds (default: 100)\n");
    fprintf(stderr, "  --right-padding-ms MS right padding in milliseconds (default: 100)\n");
}

static void parse_args(int argc, char **argv, std::string &model_path, std::string &wav_path,
                       VadConfig &config, int &input_sample_rate, int &chunk_size_ms) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv);
            exit(0);
        } else if (arg == "--model-path" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--wav-path" && i + 1 < argc) {
            wav_path = argv[++i];
        } else if (arg == "--sample-rate" && i + 1 < argc) {
            input_sample_rate = std::stoi(argv[++i]);
        } else if (arg == "--threshold" && i + 1 < argc) {
            config.threshold = std::stof(argv[++i]);
        } else if (arg == "--chunk-size-ms" && i + 1 < argc) {
            chunk_size_ms = std::stoi(argv[++i]);
        } else if (arg == "--speech-win-size-ms" && i + 1 < argc) {
            config.speech_window_size_ms = std::stoi(argv[++i]);
        } else if (arg == "--speech-win-thr-ms" && i + 1 < argc) {
            config.speech_window_threshold_ms = std::stoi(argv[++i]);
        } else if (arg == "--silence-win-size-ms" && i + 1 < argc) {
            config.silence_window_size_ms = std::stoi(argv[++i]);
        } else if (arg == "--silence-win-thr-ms" && i + 1 < argc) {
            config.silence_window_threshold_ms = std::stoi(argv[++i]);
        } else if (arg == "--max-speech-ms" && i + 1 < argc) {
            config.max_speech_ms = std::stoi(argv[++i]);
        } else if (arg == "--left-padding-ms" && i + 1 < argc) {
            config.left_padding_ms = std::stoi(argv[++i]);
        } else if (arg == "--right-padding-ms" && i + 1 < argc) {
            config.right_padding_ms = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv);
            exit(1);
        }
    }

    if (wav_path.empty()) {
        std::cerr << "Error: --wav-path is required." << std::endl;
        print_usage(argv);
        exit(1);
    }

    if (model_path.empty()) {
        std::cerr << "Error: --model-path is required." << std::endl;
        print_usage(argv);
        exit(1);
    }
}

std::vector<float> ResampleSamples(const std::vector<float> &samples, int32_t input_sample_rate,
                                   int32_t output_sample_rate) {
    if (input_sample_rate == output_sample_rate) {
        return samples;
    }

    float min_freq = static_cast<float>(std::min(input_sample_rate, output_sample_rate));
    float lowpass_cutoff = 0.99f * 0.5f * min_freq;
    int32_t lowpass_filter_width = 6;
    sherpa_onnx::LinearResample resampler(input_sample_rate, output_sample_rate, lowpass_cutoff,
                                          lowpass_filter_width);

    std::vector<float> output;
    resampler.Resample(samples.data(), static_cast<int32_t>(samples.size()), true, &output); // 70
    return output;
}



std::vector<float> load_wav(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file: " << path << '\n';
        return {};
    }

    const std::uintmax_t file_size = std::filesystem::file_size(path);
    if (file_size <= WAV_HEADER_SIZE) {
        std::cerr << "Invalid WAV file: " << path << '\n';
        return {};
    }

    file.seekg(WAV_HEADER_SIZE, std::ios::beg);  // 跳过 WAV header

    const std::size_t sample_count =
        (file_size - WAV_HEADER_SIZE) / sizeof(int16_t);

    std::vector<int16_t> raw(sample_count);
    file.read(reinterpret_cast<char*>(raw.data()),
              sample_count * sizeof(int16_t));

    if (!file) {
        std::cerr << "Read error or truncated file: " << path << '\n';
        return {};
    }

    constexpr float scale = 1.0f / 32768.0f;
    std::vector<float> data(sample_count);

    std::transform(raw.begin(), raw.end(), data.begin(),
                   [](int16_t v) {
                       return static_cast<float>(v) * scale;
                   });

    std::cout << "Loaded " << path
              << ": " << data.size() << " samples\n";

    return data;
}

int main(int argc, char *argv[]) {
    std::string model_path;
    std::string wav_path;
    VadConfig config;
    int input_sample_rate = 16000;
    int chunk_size_ms = 100;

    parse_args(argc, argv, model_path, wav_path, config, input_sample_rate, chunk_size_ms);

    // Print available ONNX Runtime providers
    std::cout << "Available ONNX Runtime Providers:" << std::endl;
    auto providers = get_ort_available_providers();
    for (const auto &p : providers) {
        std::cout << "  - " << p << std::endl;
    }

    std::vector<float> samples = load_wav(wav_path);
    if (samples.empty())
        return 1;

    if (input_sample_rate != config.sample_rate) {
        std::cout << "Resampling audio from " << input_sample_rate
                  << " Hz to " << config.sample_rate << " Hz" << std::endl;
        samples = ResampleSamples(samples, input_sample_rate, config.sample_rate);
        if (samples.empty()) {
            std::cerr << "Failed to resample audio" << std::endl;
            return 1;
        }
        std::cout << "Resampled samples: " << samples.size() << std::endl;
    }

    // 1. Create model handle (shared resources) using AutoVadModel API
    std::unique_ptr<AutoVadModel> handle = AutoVadModel::create(model_path);
    if (!handle) {
        std::cerr << "Failed to create VAD model handle" << std::endl;
        return 1;
    }

    // 2. Init an instance for inference
    std::unique_ptr<AutoVadModel> model = handle->init(config);
    if (!model) {
        std::cerr << "Failed to init VAD model instance" << std::endl;
        return 1;
    }

    int chunk_size = (config.sample_rate * chunk_size_ms) / 1000;
    std::cout << "chunk_time_ms " << chunk_size_ms << std::endl;
    std::cout << "chunk_size " << chunk_size << std::endl;

    int total_samples = samples.size();

    std::cout << "Starting VAD online decoding simulation using AutoVadModel..." << std::endl;
    for (int i = 0; i < total_samples; i += chunk_size) {
        int n = std::min(chunk_size, total_samples - i);
        bool input_finished = (i + n >= total_samples);

        // Simulating online/streaming data input
        // std::cout << "i " << i << " n " << n << " input_finished " << input_finished << std::endl;
        std::vector<VadSegment> segments = model->decode(samples.data() + i, n, input_finished);
        for (const auto &seg : segments) {
            std::stringstream ss;
            if (seg.end < 0) {
                ss << "[VoiceStart] idx " << seg.idx << " | start_ms " << seg.start_ms;
            }
            else if (seg.end > 0) {
                ss << "[VoiceEnd] idx " << seg.idx << " | start_ms " << seg.start_ms << " | end_ms " << seg.end_ms;
                auto duration = seg.end_ms - seg.start_ms;
                ss << " | duration " << duration << " ms";
            }
            std::cout << ss.str() << std::endl;
        }
    }

    return 0;
}
