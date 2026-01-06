#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstdint>
#include <algorithm>
#include <memory>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <format>
#include "vad-filter-onnx-cxx-api.h"
#include "vad-config.h"

using namespace VadFilterOnnx;

static void print_usage(char **argv) {
    fprintf(stderr, "Usage: %s [options]\n\n", argv[0]);
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            print this help message and exit\n");
    fprintf(stderr, "  --model-path PATH     path to ONNX model (required)\n");
    fprintf(stderr, "  --wav-path PATH       path to input WAV file (required)\n");
    fprintf(stderr, "  --sample-rate RATE    target sample rate (default: 16000)\n");
    fprintf(stderr, "  --threshold THR       VAD threshold (default: 0.4)\n");
    fprintf(stderr, "  --chunk-size-ms MS    chunk size in milliseconds (default: 100)\n");
    fprintf(stderr, "  --max-silence-ms MS   max silence duration in milliseconds (default: 600)\n");
    fprintf(stderr, "  --window-size-ms MS   window size in milliseconds (default: 300)\n");
    fprintf(stderr, "  --min-speech-ms MS    min speech duration in milliseconds (default: 250)\n");
    fprintf(stderr, "  --max-speech-ms MS    max speech duration in milliseconds (default: 10000)\n");
    fprintf(stderr, "  --left-padding-ms MS  left padding in milliseconds (default: 100)\n");
    fprintf(stderr, "  --right-padding-ms MS right padding in milliseconds (default: 100)\n");
}

static void parse_args(int argc, char **argv, std::string &model_path, std::string &wav_path,
                       VadConfig &config, int &chunk_size_ms) {
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
            config.sample_rate = std::stoi(argv[++i]);
        } else if (arg == "--threshold" && i + 1 < argc) {
            config.threshold = std::stof(argv[++i]);
        } else if (arg == "--chunk-size-ms" && i + 1 < argc) {
            chunk_size_ms = std::stoi(argv[++i]);
        } else if (arg == "--max-silence-ms" && i + 1 < argc) {
            config.max_silence_ms = std::stoi(argv[++i]);
        } else if (arg == "--window-size-ms" && i + 1 < argc) {
            config.window_size_ms = std::stoi(argv[++i]);
        } else if (arg == "--min-speech-ms" && i + 1 < argc) {
            config.min_speech_ms = std::stoi(argv[++i]);
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

std::vector<float> load_wav(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return {};
    }

    // Skip standard 44-byte WAV header
    file.seekg(44, std::ios::beg);

    std::vector<float> data;
    int16_t value;
    while (file.read(reinterpret_cast<char *>(&value), sizeof(int16_t))) {
        data.push_back(static_cast<float>(value) / 32768.0f);
    }

    std::cout << "Loaded " << path << " with " << data.size() << " samples" << std::endl;
    return data;
}


int main(int argc, char *argv[]) {
    std::string model_path;
    std::string wav_path;
    VadConfig config;
    int chunk_size_ms = 100;

    parse_args(argc, argv, model_path, wav_path, config, chunk_size_ms);

    std::vector<float> samples = load_wav(wav_path);
    if (samples.empty())
        return 1;

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
    int total_samples = samples.size();

    std::cout << "Starting VAD online decoding simulation using AutoVadModel..." << std::endl;
    for (int i = 0; i < total_samples; i += chunk_size) {
        int n = std::min(chunk_size, total_samples - i);
        bool last = (i + n >= total_samples);
        // printf("Processing chunk %d of %d samples, last: %d\n", i, total_samples, last);

        // Simulating online/streaming data input
        std::vector<VadSegment> segments = model->decode(samples.data() + i, n, last);
        for (const auto &seg : segments) {
            std::string msg = std::format("[VadSegment] idx {} | start_ms {} | end_ms {}", seg.idx,
                                          seg.start_ms, seg.end_ms);
            if (seg.end > 0) {
                auto duration = seg.end_ms - seg.start_ms;
                msg += std::format(" | duration {} ms", duration);
            }
            std::cout << msg << std::endl;
        }
    }

    return 0;
}
