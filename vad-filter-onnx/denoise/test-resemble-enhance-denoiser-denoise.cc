#include "denoise-filter-onnx-cxx-api.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using VadFilterOnnx::AutoDenoiseModel;
using VadFilterOnnx::DenoiseConfig;

int main(int argc, char **argv) {
    try {
        const std::string model_path = argc > 1 ? argv[1] : "public/models/resemble_enhance_denoiser.onnx";
        const int sample_rate = argc > 2 ? std::stoi(argv[2]) : 44100;
        auto handle = AutoDenoiseModel::create(model_path);
        if (!handle) {
            throw std::runtime_error("Failed to create resemble-enhance Denoiser model");
        }
        DenoiseConfig config;
        config.sample_rate = sample_rate;
        auto model = handle->init(config);
        if (!model) {
            throw std::runtime_error("Failed to initialize resemble-enhance Denoiser model");
        }

        std::mt19937 generator(20260815);
        std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
        std::vector<float> samples(sample_rate + 37);
        std::generate(samples.begin(), samples.end(), [&] { return distribution(generator); });

        // (a) A single decode(..., input_finished=true) over the whole
        // buffer returns the enhanced waveform in one call.
        model->reset();
        auto whole = model->decode(samples.data(), static_cast<int>(samples.size()), true);
        if (whole.empty()) {
            throw std::runtime_error("resemble-enhance Denoiser produced no output for a finished stream");
        }

        // (b) Feeding input incrementally is fine, but no output is ever
        // returned before the final input_finished=true call.
        model->reset();
        std::vector<float> chunked;
        constexpr std::size_t kChunkSize = 400;
        for (std::size_t offset = 0; offset < samples.size(); offset += kChunkSize) {
            const auto count = std::min<std::size_t>(kChunkSize, samples.size() - offset);
            const bool finished = offset + count == samples.size();
            auto output =
                model->decode(samples.data() + offset, static_cast<int>(count), finished);
            if (!finished && !output.empty()) {
                throw std::runtime_error("resemble-enhance Denoiser returned output before input_finished");
            }
            chunked.insert(chunked.end(), output.begin(), output.end());
        }
        if (chunked.size() != whole.size()) {
            throw std::runtime_error("Chunked-input resemble-enhance Denoiser output length mismatch");
        }
        float max_diff = 0.0F;
        for (std::size_t index = 0; index < whole.size(); ++index) {
            max_diff = std::max(max_diff, std::abs(whole[index] - chunked[index]));
        }
        if (max_diff > 1e-6F) {
            throw std::runtime_error("Chunked input feeding changed resemble-enhance Denoiser output");
        }

        // (c) decode() after a finished stream throws (until reset()).
        bool threw = false;
        try {
            model->decode(samples.data(), 1, false);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        if (!threw) {
            throw std::runtime_error("resemble-enhance Denoiser accepted decode() after stream finished");
        }

        // Empty input with input_finished immediately is allowed and
        // produces no output.
        model->reset();
        if (!model->decode(nullptr, 0, true).empty()) {
            throw std::runtime_error("Empty finished resemble-enhance Denoiser stream produced output");
        }

        DenoiseConfig invalid_config;
        invalid_config.sample_rate = sample_rate + 1;
        try {
            handle->init(invalid_config);
            throw std::runtime_error("resemble-enhance Denoiser accepted an unsupported sample rate");
        } catch (const std::invalid_argument &) {
        }

        std::cout << "resemble-enhance Denoiser denoise offline checks passed; samples=" << whole.size() << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
