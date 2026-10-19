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

namespace {

std::vector<float> Decode(AutoDenoiseModel *model, const std::vector<float> &samples,
                          int chunk_size) {
    model->reset();
    std::vector<float> result;
    if (samples.empty()) {
        return model->decode(nullptr, 0, true);
    }
    for (std::size_t offset = 0; offset < samples.size(); offset += chunk_size) {
        const auto count = std::min<std::size_t>(chunk_size, samples.size() - offset);
        const bool finished = offset + count == samples.size();
        auto output = model->decode(samples.data() + offset, static_cast<int>(count), finished);
        result.insert(result.end(), output.begin(), output.end());
    }
    return result;
}

void CheckEqual(const std::vector<float> &expected, const std::vector<float> &actual) {
    if (expected.size() != actual.size()) {
        throw std::runtime_error("Denoise output length mismatch");
    }
    float max_diff = 0.0F;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        max_diff = std::max(max_diff, std::abs(expected[index] - actual[index]));
    }
    if (max_diff > 1e-6F) {
        throw std::runtime_error("Denoise chunking changed output values");
    }
}

} // namespace

int main(int argc, char **argv) {
    try {
        const std::string model_path = argc > 1 ? argv[1] : "public/models/dpdfnet2.onnx";
        const int sample_rate = argc > 2 ? std::stoi(argv[2]) : 16000;
        auto handle = AutoDenoiseModel::create(model_path);
        if (!handle) {
            throw std::runtime_error("Failed to create DPDFNet model");
        }
        DenoiseConfig config;
        config.sample_rate = sample_rate;
        auto model = handle->init(config);
        if (!model) {
            throw std::runtime_error("Failed to initialize DPDFNet model");
        }

        std::mt19937 generator(20260815);
        std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
        std::vector<float> samples(sample_rate + 37);
        std::generate(samples.begin(), samples.end(), [&] { return distribution(generator); });

        const auto expected = Decode(model.get(), samples, static_cast<int>(samples.size()));
        if (expected.size() != samples.size()) {
            throw std::runtime_error("DPDFNet output is not input-length aligned");
        }
        for (int chunk_size : { 1, 100, 160, 500 }) {
            CheckEqual(expected, Decode(model.get(), samples, chunk_size));
        }

        for (int length : { 0, 1, 159, 160, 161 }) {
            std::vector<float> short_input(samples.begin(), samples.begin() + length);
            const auto output = Decode(model.get(), short_input, 100);
            if (output.size() != short_input.size()) {
                throw std::runtime_error("Short DPDFNet output length mismatch");
            }
        }

        CheckEqual(expected, Decode(model.get(), samples, 160));
        if (!model->decode(nullptr, 0, true).empty()) {
            throw std::runtime_error("Repeated finish produced extra samples");
        }

        DenoiseConfig invalid_config;
        invalid_config.sample_rate = sample_rate + 1;
        try {
            handle->init(invalid_config);
            throw std::runtime_error("DPDFNet accepted an unsupported sample rate");
        } catch (const std::invalid_argument &) {
        }
        std::cout << "DPDFNet denoise streaming checks passed; samples=" << expected.size()
                  << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
