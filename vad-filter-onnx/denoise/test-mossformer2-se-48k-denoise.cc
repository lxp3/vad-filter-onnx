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

std::vector<float> RunWhole(AutoDenoiseModel *model, const std::vector<float> &samples) {
    model->reset();
    auto output = model->decode(samples.data(), static_cast<int>(samples.size()), true);
    if (output.size() != samples.size()) {
        throw std::runtime_error("MossFormer2 output length mismatch");
    }
    return output;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const std::string model_path =
            argc > 1 ? argv[1] : "public/models/mossformer2_se_48k.onnx";
        const int sample_rate = argc > 2 ? std::stoi(argv[2]) : 48000;
        auto handle = AutoDenoiseModel::create(model_path);
        if (!handle) {
            throw std::runtime_error("Failed to create MossFormer2 model");
        }
        DenoiseConfig config;
        config.sample_rate = sample_rate;
        auto model = handle->init(config);
        if (!model) {
            throw std::runtime_error("Failed to initialize MossFormer2 model");
        }

        std::mt19937 generator(20260815);
        std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);

        // (a) Below one_time_decode_length (20 s): single whole-buffer
        // forward pass, no segmenting.
        std::vector<float> short_samples(sample_rate + 37);
        std::generate(short_samples.begin(), short_samples.end(),
                      [&] { return distribution(generator); });
        auto short_output = RunWhole(model.get(), short_samples);

        // (b) Feeding input incrementally is fine, but no output is ever
        // returned before the final input_finished=true call.
        model->reset();
        std::vector<float> chunked;
        constexpr std::size_t kChunkSize = 400;
        for (std::size_t offset = 0; offset < short_samples.size(); offset += kChunkSize) {
            const auto count = std::min<std::size_t>(kChunkSize, short_samples.size() - offset);
            const bool finished = offset + count == short_samples.size();
            auto output =
                model->decode(short_samples.data() + offset, static_cast<int>(count), finished);
            if (!finished && !output.empty()) {
                throw std::runtime_error("MossFormer2 returned output before input_finished");
            }
            chunked.insert(chunked.end(), output.begin(), output.end());
        }
        if (chunked.size() != short_output.size()) {
            throw std::runtime_error("Chunked-input MossFormer2 output length mismatch");
        }

        // (c) Above one_time_decode_length (20 s): exercises the
        // segmented, overlap-stitched decode path (4 s window, 75%
        // stride).
        std::vector<float> long_samples(sample_rate * 21 + 37);
        std::generate(long_samples.begin(), long_samples.end(),
                      [&] { return distribution(generator); });
        auto long_output = RunWhole(model.get(), long_samples);
        float max_abs = 0.0F;
        for (float value : long_output) {
            max_abs = std::max(max_abs, std::abs(value));
        }
        if (max_abs == 0.0F) {
            throw std::runtime_error("Segmented MossFormer2 decode produced silence");
        }

        // (d) decode() after a finished stream throws (until reset()).
        bool threw = false;
        try {
            model->decode(short_samples.data(), 1, false);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        if (!threw) {
            throw std::runtime_error("MossFormer2 accepted decode() after stream finished");
        }

        // Empty input with input_finished immediately is allowed and
        // produces no output.
        model->reset();
        if (!model->decode(nullptr, 0, true).empty()) {
            throw std::runtime_error("Empty finished MossFormer2 stream produced output");
        }

        DenoiseConfig invalid_config;
        invalid_config.sample_rate = sample_rate + 1;
        try {
            handle->init(invalid_config);
            throw std::runtime_error("MossFormer2 accepted an unsupported sample rate");
        } catch (const std::invalid_argument &) {
        }

        std::cout << "MossFormer2 denoise offline checks passed; short=" << short_output.size()
                  << " long(segmented)=" << long_output.size() << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
