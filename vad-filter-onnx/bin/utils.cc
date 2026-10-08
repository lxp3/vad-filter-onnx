#include "utils.h"
#include "../utils/resample.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <set>
#include <stdexcept>

namespace fs = std::filesystem;
namespace VadBin {

int ParseInt(const std::string &value, const std::string &option) {
    std::size_t pos = 0;
    int result;
    try {
        result = std::stoi(value, &pos);
    } catch (...) {
        throw std::invalid_argument(option + " requires an integer");
    }
    if (pos != value.size())
        throw std::invalid_argument(option + " requires an integer");
    return result;
}

float ParseFloat(const std::string &value, const std::string &option) {
    std::size_t pos = 0;
    float result;
    try {
        result = std::stof(value, &pos);
    } catch (...) {
        throw std::invalid_argument(option + " requires a number");
    }
    if (pos != value.size() || !std::isfinite(result))
        throw std::invalid_argument(option + " requires a finite number");
    return result;
}

static uint16_t U16(const char *data) {
    return static_cast<uint8_t>(data[0]) |
           (static_cast<uint16_t>(static_cast<uint8_t>(data[1])) << 8);
}
static uint32_t U32(const char *data) {
    return U16(data) | (static_cast<uint32_t>(U16(data + 2)) << 16);
}

Audio LoadWav(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open " + path.string());
    std::vector<char> bytes((std::istreambuf_iterator<char>(in)), {});
    if (bytes.size() < 12 || std::string(bytes.data(), 4) != "RIFF" ||
        std::string(bytes.data() + 8, 4) != "WAVE")
        throw std::runtime_error("invalid WAV: " + path.string());
    int rate = 0, channels = 0, format = 0, bits = 0;
    std::size_t pos = 12, data_pos = 0, data_size = 0;
    while (pos + 8 <= bytes.size()) {
        std::string id(bytes.data() + pos, 4);
        uint32_t size = U32(bytes.data() + pos + 4);
        pos += 8;
        if (pos + size > bytes.size())
            throw std::runtime_error("truncated WAV: " + path.string());
        if (id == "fmt " && size >= 16) {
            format = U16(bytes.data() + pos);
            channels = U16(bytes.data() + pos + 2);
            rate = static_cast<int>(U32(bytes.data() + pos + 4));
            bits = U16(bytes.data() + pos + 14);
        }
        if (id == "data") {
            data_pos = pos;
            data_size = size;
            break;
        }
        pos += size + (size & 1);
    }
    if (!data_pos || rate <= 0 || channels < 1 ||
        !((format == 1 && bits == 16) || (format == 3 && bits == 32)))
        throw std::runtime_error("WAV must be PCM16 or float32: " + path.string());
    const std::size_t bytes_per_sample = static_cast<std::size_t>(bits / 8);
    if (!data_size || data_size % (bytes_per_sample * channels))
        throw std::runtime_error("empty or incomplete WAV: " + path.string());
    const std::size_t frames = data_size / (bytes_per_sample * channels);
    std::vector<float> samples(frames);
    for (std::size_t frame = 0; frame < frames; ++frame) {
        double sum = 0;
        for (int channel = 0; channel < channels; ++channel) {
            const char *data =
                bytes.data() + data_pos + (frame * channels + channel) * bytes_per_sample;
            if (bits == 16)
                sum += static_cast<int16_t>(U16(data)) / 32768.0;
            else {
                uint32_t raw = U32(data);
                float value;
                std::memcpy(&value, &raw, sizeof(value));
                sum += value;
            }
        }
        samples[frame] = static_cast<float>(sum / channels);
        if (!std::isfinite(samples[frame]))
            throw std::runtime_error("non-finite WAV samples: " + path.string());
    }
    return { rate, std::move(samples) };
}

void SaveWav(const fs::path &path, const std::vector<float> &samples, int rate) {
    if (rate <= 0 || samples.size() > (std::numeric_limits<uint32_t>::max() - 36) / 2)
        throw std::runtime_error("invalid WAV output size or sample rate");
    std::ofstream out(path, std::ios::binary);
    if (!out)
        throw std::runtime_error("cannot write " + path.string());
    const uint32_t size = static_cast<uint32_t>(samples.size() * 2);
    auto w16 = [&out](uint16_t value) {
        out.put(static_cast<char>(value));
        out.put(static_cast<char>(value >> 8));
    };
    auto w32 = [&w16](uint32_t value) {
        w16(static_cast<uint16_t>(value));
        w16(static_cast<uint16_t>(value >> 16));
    };
    out.write("RIFF", 4);
    w32(36 + size);
    out.write("WAVEfmt ", 8);
    w32(16);
    w16(1);
    w16(1);
    w32(rate);
    w32(rate * 2);
    w16(2);
    w16(16);
    out.write("data", 4);
    w32(size);
    for (float sample : samples) {
        if (!std::isfinite(sample))
            throw std::runtime_error("non-finite WAV output");
        int value =
            std::clamp(static_cast<int>(std::clamp(sample, -1.0f, 1.0f) * 32768.0f), -32768, 32767);
        w16(static_cast<uint16_t>(static_cast<int16_t>(value)));
    }
    out.flush();
    if (!out)
        throw std::runtime_error("failed to write " + path.string());
}

std::vector<float> Resample(const std::vector<float> &samples, int input_rate, int output_rate) {
    if (input_rate <= 0 || output_rate <= 0 || samples.size() > std::numeric_limits<int32_t>::max())
        throw std::invalid_argument("invalid resampling rate or audio too long");
    if (input_rate == output_rate)
        return samples;
    sherpa_onnx::LinearResample resampler(input_rate, output_rate,
                                          0.99f * 0.5f * std::min(input_rate, output_rate), 6);
    std::vector<float> output;
    resampler.Resample(samples.data(), static_cast<int32_t>(samples.size()), true, &output);
    return output;
}

std::vector<std::pair<std::string, fs::path> > CollectAudio(const fs::path &path) {
    std::vector<std::pair<std::string, fs::path> > result;
    auto is_wav = [](const fs::path &file) {
        std::string extension = file.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char character) { return std::tolower(character); });
        return extension == ".wav";
    };
    auto add = [&](const fs::path &file, const std::string &id) {
        std::string key = id.empty() ? file.stem().string() : id;
        std::replace_if(
            key.begin(), key.end(), [](unsigned char character) { return std::isspace(character); },
            '_');
        result.emplace_back(key, file);
    };
    if (fs::is_regular_file(path) && is_wav(path)) {
        add(path, path.stem().string());
    } else if (fs::is_directory(path)) {
        for (const auto &entry : fs::recursive_directory_iterator(path)) {
            if (entry.is_regular_file() && is_wav(entry.path()))
                add(entry.path(),
                    fs::relative(entry.path(), path).replace_extension().generic_string());
        }
        std::sort(result.begin(), result.end());
    } else {
        std::ifstream list(path);
        if (!list)
            throw std::runtime_error("cannot open --path " + path.string());
        std::string line;
        while (std::getline(list, line)) {
            const auto begin = line.find_first_not_of(" \t\r");
            if (begin == std::string::npos)
                continue;
            line = line.substr(begin, line.find_last_not_of(" \t\r") - begin + 1);
            fs::path file = line;
            if (file.is_relative())
                file = path.parent_path() / file;
            std::string id;
            if (!fs::is_regular_file(file)) {
                const auto split = line.find_first_of(" \t");
                if (split != std::string::npos) {
                    id = line.substr(0, split);
                    const auto value = line.find_first_not_of(" \t", split);
                    if (value == std::string::npos)
                        throw std::runtime_error("missing path in list");
                    file = line.substr(value);
                    if (file.is_relative())
                        file = path.parent_path() / file;
                }
            }
            add(file, id);
        }
    }
    if (result.empty())
        throw std::runtime_error("no audio files found in " + path.string());
    std::set<std::string> ids;
    for (const auto &[id, file] : result) {
        fs::path relative_id(id);
        if (relative_id.is_absolute() || id.empty())
            throw std::runtime_error("invalid audio ID: " + id);
        for (const auto &component : relative_id) {
            if (component == ".." || component == ".")
                throw std::runtime_error("invalid audio ID: " + id);
        }
        if (id.find_first_of(" \t\r\n") != std::string::npos)
            throw std::runtime_error("audio ID cannot contain whitespace: " + id);
        if (!ids.insert(id).second)
            throw std::runtime_error("duplicate audio ID: " + id);
    }
    return result;
}
} // namespace VadBin
