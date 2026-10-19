#include "separate/bs-roformer-separator.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string model_path;
    std::string input_raw;
    std::string output_vocals;
    std::string output_instrumental;
    int64_t frames = 0;
    int channels = 0;
    int sample_rate = 0;
    int num_threads = 1;
};

void Usage(const char *program) {
    std::cout << "Usage: " << program
              << " --model-path PATH --input-raw PATH --frames N --channels N --sample-rate N"
                 " --output-vocals PATH --output-instrumental PATH [--num-threads N]\n";
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
        } else if (argument == "--input-raw") {
            options.input_raw = value;
        } else if (argument == "--output-vocals") {
            options.output_vocals = value;
        } else if (argument == "--output-instrumental") {
            options.output_instrumental = value;
        } else if (argument == "--frames") {
            options.frames = std::stoll(value);
        } else if (argument == "--channels") {
            options.channels = std::stoi(value);
        } else if (argument == "--sample-rate") {
            options.sample_rate = std::stoi(value);
        } else if (argument == "--num-threads") {
            options.num_threads = std::stoi(value);
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }
    if (options.model_path.empty() || options.input_raw.empty() || options.output_vocals.empty() ||
        options.output_instrumental.empty() || options.frames <= 0 || options.channels <= 0 ||
        options.sample_rate <= 0 || options.num_threads <= 0) {
        throw std::invalid_argument("missing or invalid separation test arguments");
    }
    return options;
}

std::vector<float> ReadRaw(const std::string &path, std::size_t count) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open " + path);
    }
    std::vector<float> samples(count);
    input.read(reinterpret_cast<char *>(samples.data()),
               static_cast<std::streamsize>(count * sizeof(float)));
    if (static_cast<std::size_t>(input.gcount()) != count * sizeof(float)) {
        throw std::runtime_error("truncated raw audio: " + path);
    }
    return samples;
}

void WriteRaw(const std::string &path, const std::vector<float> &samples) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("cannot write " + path);
    }
    output.write(reinterpret_cast<const char *>(samples.data()),
                 static_cast<std::streamsize>(samples.size() * sizeof(float)));
    if (!output) {
        throw std::runtime_error("failed to write " + path);
    }
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Options options = Parse(argc, argv);
        const auto samples = ReadRaw(options.input_raw,
                                     static_cast<std::size_t>(options.frames) *
                                         static_cast<std::size_t>(options.channels));
        VadFilterOnnx::BsRoformerSeparator separator(options.model_path, options.num_threads);
        auto result = separator.Separate(samples.data(), options.frames, options.channels,
                                         options.sample_rate);
        WriteRaw(options.output_vocals, result.vocals);
        WriteRaw(options.output_instrumental, result.instrumental);
        std::cout << "frames=" << result.vocals.size() / 2 << " sample_rate=" << result.sample_rate
                  << "\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        Usage(argv[0]);
        return 1;
    }
}
