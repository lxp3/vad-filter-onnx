#include "separate/bs-roformer-separator.h"
#include "utils.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iterator>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Options {
    std::string model_path;
    std::string path;
    std::string output_dir;
};

struct StereoWav {
    int sample_rate = 0;
    int channels = 0;
    std::vector<float> samples;
};

void Usage(const char *program) {
    std::cout << "Usage: " << program
              << " --model-path PATH --path WAV|DIR --output-dir DIR\n";
}

Options Parse(int argc, char **argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "-h" || argument == "--help") {
            Usage(argv[0]);
            std::exit(0);
        }
        if (++index >= argc) {
            throw std::invalid_argument("missing value for " + argument);
        }
        const std::string value = argv[index];
        if (argument == "--model-path") {
            options.model_path = value;
        } else if (argument == "--path") {
            options.path = value;
        } else if (argument == "--output-dir") {
            options.output_dir = value;
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }
    if (options.model_path.empty() || options.path.empty() || options.output_dir.empty()) {
        throw std::invalid_argument("--model-path, --path, and --output-dir are required");
    }
    return options;
}

uint16_t ReadU16(const char *data) {
    return static_cast<uint8_t>(data[0]) |
           (static_cast<uint16_t>(static_cast<uint8_t>(data[1])) << 8);
}

uint32_t ReadU32(const char *data) {
    return ReadU16(data) | (static_cast<uint32_t>(ReadU16(data + 2)) << 16);
}

bool IsWav(const fs::path &path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char character) { return std::tolower(character); });
    return extension == ".wav";
}

StereoWav LoadStereoWav(const fs::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open " + path.string());
    }
    std::vector<char> bytes((std::istreambuf_iterator<char>(input)), {});
    if (bytes.size() < 12 || std::string(bytes.data(), 4) != "RIFF" ||
        std::string(bytes.data() + 8, 4) != "WAVE") {
        throw std::runtime_error("invalid WAV: " + path.string());
    }
    int rate = 0, channels = 0, format = 0, bits = 0;
    std::size_t pos = 12, data_pos = 0, data_size = 0;
    while (pos + 8 <= bytes.size()) {
        const std::string id(bytes.data() + pos, 4);
        const uint32_t size = ReadU32(bytes.data() + pos + 4);
        pos += 8;
        if (pos + size > bytes.size()) {
            throw std::runtime_error("truncated WAV: " + path.string());
        }
        if (id == "fmt " && size >= 16) {
            format = ReadU16(bytes.data() + pos);
            channels = ReadU16(bytes.data() + pos + 2);
            rate = static_cast<int>(ReadU32(bytes.data() + pos + 4));
            bits = ReadU16(bytes.data() + pos + 14);
        }
        if (id == "data") {
            data_pos = pos;
            data_size = size;
            break;
        }
        pos += size + (size & 1);
    }
    if (!data_pos || rate <= 0 || channels < 1 ||
        !((format == 1 && bits == 16) || (format == 3 && bits == 32))) {
        throw std::runtime_error("WAV must be PCM16 or float32: " + path.string());
    }
    const std::size_t bytes_per_sample = static_cast<std::size_t>(bits / 8);
    if (!data_size || data_size % (bytes_per_sample * static_cast<std::size_t>(channels))) {
        throw std::runtime_error("empty or incomplete WAV: " + path.string());
    }
    const std::size_t frames = data_size / (bytes_per_sample * static_cast<std::size_t>(channels));
    std::vector<float> samples(frames * static_cast<std::size_t>(channels));
    for (std::size_t frame = 0; frame < frames; ++frame) {
        for (int channel = 0; channel < channels; ++channel) {
            const char *data = bytes.data() + data_pos +
                               (frame * static_cast<std::size_t>(channels) + channel) * bytes_per_sample;
            float value = 0.0F;
            if (bits == 16) {
                value = static_cast<float>(static_cast<int16_t>(ReadU16(data)) / 32768.0);
            } else {
                uint32_t raw = ReadU32(data);
                std::memcpy(&value, &raw, sizeof(value));
            }
            if (!std::isfinite(value)) {
                throw std::runtime_error("non-finite WAV samples: " + path.string());
            }
            samples[frame * static_cast<std::size_t>(channels) + channel] = value;
        }
    }
    return { rate, channels, std::move(samples) };
}

void WriteFloatStereoWav(const fs::path &path, const std::vector<float> &interleaved, int rate) {
    if (rate <= 0 || interleaved.size() % 2 != 0) {
        throw std::runtime_error("invalid stereo WAV output");
    }
    const uint64_t data_bytes = interleaved.size() * sizeof(float);
    if (data_bytes > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max() - 36)) {
        throw std::runtime_error("WAV output is too large");
    }
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("cannot write " + path.string());
    }
    const uint32_t chunk = static_cast<uint32_t>(data_bytes);
    auto write16 = [&output](uint16_t value) {
        output.put(static_cast<char>(value));
        output.put(static_cast<char>(value >> 8));
    };
    auto write32 = [&write16](uint32_t value) {
        write16(static_cast<uint16_t>(value));
        write16(static_cast<uint16_t>(value >> 16));
    };
    output.write("RIFF", 4);
    write32(36 + chunk);
    output.write("WAVEfmt ", 8);
    write32(16);
    write16(3);
    write16(2);
    write32(static_cast<uint32_t>(rate));
    write32(static_cast<uint32_t>(rate) * 8);
    write16(8);
    write16(32);
    output.write("data", 4);
    write32(chunk);
    for (float sample : interleaved) {
        if (!std::isfinite(sample)) {
            throw std::runtime_error("non-finite WAV output");
        }
        uint32_t raw = 0;
        std::memcpy(&raw, &sample, sizeof(raw));
        write32(raw);
    }
    output.flush();
    if (!output) {
        throw std::runtime_error("failed to write " + path.string());
    }
}

std::vector<fs::path> CollectWavs(const fs::path &path) {
    std::vector<fs::path> files;
    if (fs::is_regular_file(path) && IsWav(path)) {
        files.push_back(path);
    } else if (fs::is_directory(path)) {
        for (const auto &entry : fs::recursive_directory_iterator(path)) {
            if (entry.is_regular_file() && IsWav(entry.path())) {
                files.push_back(entry.path());
            }
        }
        std::sort(files.begin(), files.end());
    }
    if (files.empty()) {
        throw std::runtime_error("no WAV files found in " + path.string());
    }
    return files;
}

std::string OutputStem(const fs::path &file, const fs::path &root) {
    std::string key;
    if (fs::is_directory(root)) {
        key = fs::relative(file, root).generic_string();
    } else {
        key = file.filename().string();
    }
    const auto extension = key.rfind('.');
    if (extension != std::string::npos) {
        key.erase(extension);
    }
    std::replace(key.begin(), key.end(), '/', '_');
    if (key.empty()) {
        throw std::runtime_error("invalid output name for " + file.string());
    }
    return key;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Options options = Parse(argc, argv);
        VadFilterOnnx::BsRoformerSeparator separator(options.model_path, 1);
        const fs::path root(options.path);
        const auto files = CollectWavs(root);
        fs::create_directories(options.output_dir);
        std::size_t failed = 0;
        for (const auto &file : files) {
            try {
                const StereoWav audio = LoadStereoWav(file);
                auto separated = separator.Separate(audio.samples.data(),
                                                    static_cast<int64_t>(audio.samples.size() /
                                                                         static_cast<std::size_t>(audio.channels)),
                                                    audio.channels, audio.sample_rate);
                const std::string stem = OutputStem(file, root);
                WriteFloatStereoWav(fs::path(options.output_dir) / (stem + "_vocals.wav"),
                                    separated.vocals, separated.sample_rate);
                WriteFloatStereoWav(fs::path(options.output_dir) / (stem + "_instrumental.wav"),
                                    separated.instrumental, separated.sample_rate);
            } catch (const std::exception &error) {
                ++failed;
                std::cerr << file << ": " << error.what() << '\n';
            }
        }
        std::cerr << "Processed: " << files.size() - failed << ", failed: " << failed << '\n';
        return failed ? 1 : 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        Usage(argv[0]);
        return 1;
    }
}
