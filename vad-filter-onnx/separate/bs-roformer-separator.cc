#include "separate/bs-roformer-separator.h"

#include "utils/onnx-common.h"
#include "utils/resample.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace VadFilterOnnx {
namespace {

constexpr int kSampleRate = 44100;
constexpr int kChannels = 2;
constexpr int64_t kChunk = 352800;
constexpr int kOverlap = 4;
constexpr int64_t kStep = kChunk / kOverlap;
constexpr float kPeak = 0.9F;
constexpr float kWeightFloor = 1e-10F;

std::string MetadataValue(Ort::Session &session, const char *key) {
    Ort::AllocatorWithDefaultOptions allocator;
    auto value = session.GetModelMetadata().LookupCustomMetadataMapAllocated(key, allocator);
    return value ? std::string(value.get()) : std::string();
}

void RequireMetadata(Ort::Session &session, const char *key, const std::string &expected) {
    const std::string actual = MetadataValue(session, key);
    if (actual != expected) {
        throw std::runtime_error(std::string("unexpected ") + key + " metadata: " + actual);
    }
}

std::vector<float> SymmetricHamming(int size) {
    std::vector<float> window(static_cast<std::size_t>(size));
    if (size == 1) {
        window[0] = 1.0F;
        return window;
    }
    for (int index = 0; index < size; ++index) {
        const double phase = (2.0 * std::numbers::pi * index) / static_cast<double>(size - 1);
        window[static_cast<std::size_t>(index)] =
            static_cast<float>(0.54 - 0.46 * std::cos(phase));
    }
    return window;
}

std::vector<int64_t> ChunkStarts(int64_t audio_length) {
    std::vector<int64_t> starts;
    if (audio_length <= 0) {
        throw std::invalid_argument("audio length must be positive");
    }
    if (audio_length <= kChunk) {
        starts.push_back(0);
        return starts;
    }
    for (int64_t offset = 0; offset < audio_length; offset += kStep) {
        if (offset + kChunk >= audio_length) {
            const int64_t tail = std::max<int64_t>(audio_length - kChunk, 0);
            if (starts.empty() || starts.back() != tail) {
                starts.push_back(tail);
            }
            break;
        }
        starts.push_back(offset);
    }
    return starts;
}

float Peak(const std::vector<float> &wave) {
    float peak = 0.0F;
    for (float value : wave) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("audio contains a non-finite sample");
        }
        peak = std::max(peak, std::abs(value));
    }
    return peak;
}

void LimitPeak(std::vector<float> &wave) {
    const float peak = Peak(wave);
    if (peak > kPeak) {
        const float scale = kPeak / peak;
        for (float &value : wave) {
            value *= scale;
        }
    }
}

std::vector<float> ResampleChannel(const std::vector<float> &samples, int input_rate) {
    if (input_rate == kSampleRate) {
        return samples;
    }
    if (samples.size() > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::invalid_argument("audio is too long to resample");
    }
    const float cutoff = 0.99F * 0.5F * static_cast<float>(std::min(input_rate, kSampleRate));
    sherpa_onnx::LinearResample resampler(input_rate, kSampleRate, cutoff, 6);
    std::vector<float> output;
    resampler.Resample(samples.data(), static_cast<int32_t>(samples.size()), true, &output);
    return output;
}

std::vector<float> ToModelRate(const float *samples, int64_t frames, int channels, int sample_rate) {
    if (samples == nullptr || frames <= 0 || channels < 1 || sample_rate <= 0) {
        throw std::invalid_argument("invalid audio for separation");
    }
    if (frames > std::numeric_limits<int64_t>::max() / channels) {
        throw std::invalid_argument("audio is too long");
    }
    std::vector<float> first(static_cast<std::size_t>(frames));
    std::vector<float> second(static_cast<std::size_t>(frames));
    for (int64_t frame = 0; frame < frames; ++frame) {
        const float left = samples[frame * channels];
        const float right = channels == 1 ? left : samples[frame * channels + 1];
        if (!std::isfinite(left) || !std::isfinite(right)) {
            throw std::invalid_argument("audio contains a non-finite sample");
        }
        first[static_cast<std::size_t>(frame)] = left;
        second[static_cast<std::size_t>(frame)] = right;
    }
    first = ResampleChannel(first, sample_rate);
    second = ResampleChannel(second, sample_rate);
    if (first.size() != second.size() || first.empty()) {
        throw std::runtime_error("resampled channel lengths do not match");
    }
    std::vector<float> packed(first.size() * 2);
    std::copy(first.begin(), first.end(), packed.begin());
    std::copy(second.begin(), second.end(), packed.begin() + static_cast<std::ptrdiff_t>(first.size()));
    return packed;
}

std::vector<float> Interleave(const std::vector<float> &packed) {
    const std::size_t frames = packed.size() / 2;
    std::vector<float> interleaved(packed.size());
    for (std::size_t frame = 0; frame < frames; ++frame) {
        interleaved[frame * 2] = packed[frame];
        interleaved[frame * 2 + 1] = packed[frames + frame];
    }
    return interleaved;
}

} // namespace

struct BsRoformerSeparator::Impl {
    std::shared_ptr<Ort::Session> session;
    std::string input_name;
    std::string output_name;
    std::vector<float> window;

    explicit Impl(const std::string &model_path, int num_threads) {
        if (model_path.empty() || num_threads <= 0) {
            throw std::invalid_argument("model path and thread count are required");
        }
        Ort::SessionOptions options = GetSessionOptions(num_threads, -1);
        try {
#ifdef _WIN32
            const std::wstring wide_path(model_path.begin(), model_path.end());
            session = std::make_shared<Ort::Session>(GetOrtEnv(), wide_path.c_str(), options);
#else
            session = std::make_shared<Ort::Session>(GetOrtEnv(), model_path.c_str(), options);
#endif
        } catch (const std::exception &error) {
            throw std::runtime_error(std::string("failed to load separator model: ") + error.what());
        }
        if (session->GetInputCount() != 1 || session->GetOutputCount() != 1) {
            throw std::runtime_error("separator model must have one input and one output");
        }
        RequireMetadata(*session, "model_type", "bs_roformer_separate");
        RequireMetadata(*session, "sample_rate", "44100");
        RequireMetadata(*session, "channels", "2");
        RequireMetadata(*session, "chunk_size", "352800");
        RequireMetadata(*session, "overlap", "4");
        RequireMetadata(*session, "streaming", "0");

        Ort::AllocatorWithDefaultOptions allocator;
        input_name = session->GetInputNameAllocated(0, allocator).get();
        output_name = session->GetOutputNameAllocated(0, allocator).get();
        const auto input_shape = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        const auto output_shape = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        const std::vector<int64_t> expected = { 1, kChannels, kChunk };
        if (input_shape != expected || output_shape != expected) {
            throw std::runtime_error("separator model shape must be [1, 2, 352800]");
        }
        if (session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType() !=
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType() !=
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error("separator model must be float32");
        }
        window = SymmetricHamming(static_cast<int>(kChunk));
    }

    std::vector<float> RunChunk(const std::vector<float> &packed, int64_t start, int64_t frames) {
        std::vector<float> chunk(static_cast<std::size_t>(kChannels * kChunk), 0.0F);
        const int64_t valid = std::min<int64_t>(kChunk, frames - start);
        for (int channel = 0; channel < kChannels; ++channel) {
            const float *source = packed.data() + channel * frames + start;
            std::copy(source, source + valid, chunk.data() + channel * kChunk);
        }
        const std::array<int64_t, 3> shape = { 1, kChannels, kChunk };
        const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input = Ort::Value::CreateTensor<float>(
            memory, chunk.data(), chunk.size(), shape.data(), shape.size());
        const char *input_names[] = { input_name.c_str() };
        const char *output_names[] = { output_name.c_str() };
        auto outputs = session->Run(Ort::RunOptions{ nullptr }, input_names, &input, 1, output_names, 1);
        if (outputs.size() != 1 || !outputs[0].IsTensor()) {
            throw std::runtime_error("separator model returned an unexpected output");
        }
        const auto count = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
        if (count != static_cast<std::size_t>(kChannels * kChunk)) {
            throw std::runtime_error("separator chunk output has the wrong size");
        }
        const float *data = outputs[0].GetTensorData<float>();
        return std::vector<float>(data, data + count);
    }
};

BsRoformerSeparator::BsRoformerSeparator(const std::string &model_path, int num_threads)
    : impl_(std::make_unique<Impl>(model_path, num_threads)) {}

BsRoformerSeparator::~BsRoformerSeparator() = default;

Separation BsRoformerSeparator::Separate(const float *samples, int64_t frames, int channels,
                                         int sample_rate) {
    std::vector<float> mix = ToModelRate(samples, frames, channels, sample_rate);
    LimitPeak(mix);
    const int64_t length = static_cast<int64_t>(mix.size() / 2);
    std::vector<float> vocals(mix.size(), 0.0F);
    std::vector<float> weight(static_cast<std::size_t>(length), 0.0F);
    for (int64_t start : ChunkStarts(length)) {
        const std::vector<float> chunk = impl_->RunChunk(mix, start, length);
        const int64_t valid = std::min<int64_t>(kChunk, length - start);
        for (int channel = 0; channel < kChannels; ++channel) {
            float *target = vocals.data() + channel * length + start;
            const float *source = chunk.data() + channel * kChunk;
            for (int64_t index = 0; index < valid; ++index) {
                target[index] += source[index] * impl_->window[static_cast<std::size_t>(index)];
            }
        }
        for (int64_t index = 0; index < valid; ++index) {
            weight[static_cast<std::size_t>(start + index)] +=
                impl_->window[static_cast<std::size_t>(index)];
        }
    }
    for (int64_t index = 0; index < length; ++index) {
        const float scale = 1.0F / std::max(weight[static_cast<std::size_t>(index)], kWeightFloor);
        vocals[static_cast<std::size_t>(index)] *= scale;
        vocals[static_cast<std::size_t>(length + index)] *= scale;
    }
    std::vector<float> instrumental(mix.size());
    for (std::size_t index = 0; index < mix.size(); ++index) {
        instrumental[index] = mix[index] - vocals[index];
    }
    LimitPeak(vocals);
    LimitPeak(instrumental);
    Separation result;
    result.sample_rate = kSampleRate;
    result.vocals = Interleave(vocals);
    result.instrumental = Interleave(instrumental);
    return result;
}

} // namespace VadFilterOnnx
