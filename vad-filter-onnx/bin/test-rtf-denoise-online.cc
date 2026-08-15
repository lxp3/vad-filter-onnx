#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "denoise-filter-onnx-cxx-api.h"

namespace {

constexpr int kAudioSeconds = 5;

struct Options {
    std::string model_path;
    int num_warmups = 5;
    int num_runs = 20;
    int chunk_ms = 100;
    VadFilterOnnx::DenoiseConfig config;
};

void PrintUsage(const char *program) {
    std::cerr
        << "Usage: " << program << " --model-path PATH [options]\n\n"
        << "Options:\n"
        << "  --num-warmups N             warmup runs (default: 5)\n"
        << "  --num-runs N                measured runs (default: 20)\n"
        << "  --chunk-ms N                streaming chunk size (default: 100)\n"
        << "  --sample-rate N             sample rate (default: 16000)\n";
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
        } else if (arg == "--num-warmups") {
            options.num_warmups = ParseInteger(argv[i], arg);
        } else if (arg == "--num-runs") {
            options.num_runs = ParseInteger(argv[i], arg);
        } else if (arg == "--chunk-ms") {
            options.chunk_ms = ParseInteger(argv[i], arg);
        } else if (arg == "--sample-rate") {
            options.config.sample_rate = ParseInteger(argv[i], arg);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    if (options.model_path.empty()) {
        throw std::invalid_argument("--model-path is required");
    }
    if (options.num_warmups < 0) {
        throw std::invalid_argument("--num-warmups must be at least 0");
    }
    if (options.num_runs <= 0) {
        throw std::invalid_argument("--num-runs must be greater than 0");
    }
    if (options.chunk_ms <= 0) {
        throw std::invalid_argument("--chunk-ms must be greater than 0");
    }
    if (options.config.sample_rate <= 0) {
        throw std::invalid_argument("--sample-rate must be greater than 0");
    }
    return options;
}

double DecodeOnce(VadFilterOnnx::AutoDenoiseModel *model, std::vector<float> *samples,
                  int sample_rate, int chunk_ms) {
    const int chunk_size = sample_rate * chunk_ms / 1000;
    if (chunk_size <= 0) {
        throw std::invalid_argument("--chunk-ms is too small for the selected sample rate");
    }
    model->reset();

    const auto start = std::chrono::steady_clock::now();
    for (std::size_t offset = 0; offset < samples->size(); offset += chunk_size) {
        const auto count = std::min<std::size_t>(chunk_size, samples->size() - offset);
        const bool input_finished = offset + count == samples->size();
        model->decode(samples->data() + offset, static_cast<int>(count), input_finished);
    }
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Options options = ParseArgs(argc, argv);

        std::mt19937 generator(20260723);
        std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
        std::vector<float> samples(options.config.sample_rate * kAudioSeconds);
        std::generate(samples.begin(), samples.end(), [&] { return distribution(generator); });

        auto handle = VadFilterOnnx::AutoDenoiseModel::create(options.model_path);
        if (!handle) {
            throw std::runtime_error("failed to load model: " + options.model_path);
        }

        auto model = handle->init(options.config);
        if (!model) {
            throw std::runtime_error("failed to initialize model instance");
        }

        for (int i = 0; i < options.num_warmups; ++i) {
            DecodeOnce(model.get(), &samples, options.config.sample_rate, options.chunk_ms);
        }

        double total_seconds = 0.0;
        std::cout << std::fixed << std::setprecision(6);
        for (int i = 0; i < options.num_runs; ++i) {
            const double elapsed_seconds =
                DecodeOnce(model.get(), &samples, options.config.sample_rate, options.chunk_ms);
            total_seconds += elapsed_seconds;
            std::cout << "Run " << i + 1 << ": " << elapsed_seconds
                      << " seconds, RTF = " << elapsed_seconds / kAudioSeconds << '\n';
        }

        const double average_seconds = total_seconds / options.num_runs;
        std::cout << "Average: " << average_seconds
                  << " seconds, average RTF = " << average_seconds / kAudioSeconds << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        PrintUsage(argv[0]);
        return 1;
    }
}
