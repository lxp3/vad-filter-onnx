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
#include "vad/vad-model.h"
#include "vad-config.h"

struct Args {
    std::string model_path;
    std::string wav_path;
    int sample_rate = 16000;
    float threshold = 0.4f;
    int chunk_size_ms = 100;
};

static void print_usage(char **argv) {
    fprintf(stderr, "Usage: %s [options]\n\n", argv[0]);
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            print this help message and exit\n");
    fprintf(stderr, "  --model-path PATH     path to ONNX model (required)\n");
    fprintf(stderr, "  --wav-path PATH       path to input WAV file (required)\n");
    fprintf(stderr, "  --sample-rate RATE    target sample rate (default: 16000)\n");
    fprintf(stderr, "  --threshold THR       VAD threshold (default: 0.4)\n");
    fprintf(stderr, "  --chunk-size-ms MS    chunk size in milliseconds (default: 100)\n");
}

static void parse_args(int argc, char **argv, Args &args) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv);
            exit(0);
        } else if (arg == "--model-path" && i + 1 < argc) {
            args.model_path = argv[++i];
        } else if (arg == "--wav-path" && i + 1 < argc) {
            args.wav_path = argv[++i];
        } else if (arg == "--sample-rate" && i + 1 < argc) {
            args.sample_rate = std::stoi(argv[++i]);
        } else if (arg == "--threshold" && i + 1 < argc) {
            args.threshold = std::stof(argv[++i]);
        } else if (arg == "--chunk-size-ms" && i + 1 < argc) {
            args.chunk_size_ms = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv);
            exit(1);
        }
    }

    if (args.wav_path.empty()) {
        std::cerr << "Error: --wav-path is required." << std::endl;
        print_usage(argv);
        exit(1);
    }

    if (args.model_path.empty()) {
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


int main(int argc, char* argv[]) {
    Args args;
    parse_args(argc, argv, args);

    std::vector<float> samples = load_wav(args.wav_path);
    if (samples.empty()) return 1;

    VadConfig config;
    config.sample_rate = args.sample_rate;
    config.threshold = args.threshold;

    // 1. Create model handle (shared resources)
    std::unique_ptr<VadModel> handle = VadModel::create(args.model_path);
    if (!handle) {
        std::cerr << "Failed to create VAD model handle" << std::endl;
        return 1;
    }

    // 2. Init an instance for inference
    std::unique_ptr<VadModel> model = handle->init(config);
    if (!model) {
        std::cerr << "Failed to init VAD model instance" << std::endl;
        return 1;
    }

    int chunk_size = (args.sample_rate * args.chunk_size_ms) / 1000;
    int total_samples = samples.size();
    
    std::cout << "Starting VAD online decoding simulation..." << std::endl;
    for (int i = 0; i < total_samples; i += chunk_size) {
        int n = std::min(chunk_size, total_samples - i);
        bool last = (i + n >= total_samples);
        printf("Processing chunk %d of %d samples, last: %d\n", i, total_samples, last);
        
        // Simulating online/streaming data input
        std::vector<VadSegment> segments = model->decode(samples.data() + i, n, last);
        for (const auto& seg : segments) {
            printf("[VadSegment] idx %d | start %d | end %d | start_ms %d | end_ms %d\n",
                   seg.idx, seg.start, seg.end, seg.start_ms, seg.end_ms);
        }
    }

    return 0;
}
