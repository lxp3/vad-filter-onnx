#include "utils.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void PrintUsage(const char *program) {
    std::cout << "Usage: " << program
              << " --input-wav-path PATH --output-wav-path PATH --sample-rate RATE\n";
}

int ParseArguments(int argc, char **argv, std::string &input_path, std::string &output_path,
                   int &sample_rate) {
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            PrintUsage(argv[0]);
            return 1;
        }
        if (index + 1 >= argc)
            throw std::invalid_argument("missing value for " + argument);
        const std::string value = argv[++index];
        if (argument == "--input-wav-path")
            input_path = value;
        else if (argument == "--output-wav-path")
            output_path = value;
        else if (argument == "--sample-rate")
            sample_rate = VadBin::ParseInt(value, argument);
        else
            throw std::invalid_argument("unknown argument: " + argument);
    }
    if (input_path.empty() || output_path.empty())
        throw std::invalid_argument("--input-wav-path and --output-wav-path are required");
    if (sample_rate <= 0)
        throw std::invalid_argument("--sample-rate must be greater than zero");
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    try {
        std::string input_path;
        std::string output_path;
        int sample_rate = 16000;
        if (ParseArguments(argc, argv, input_path, output_path, sample_rate) != 0)
            return 0;
        const auto input = VadBin::LoadWav(input_path);
        const auto output = VadBin::Resample(input.samples, input.sample_rate, sample_rate);
        VadBin::SaveWav(output_path, output, sample_rate);
        std::cout << "Resampled " << input_path << " (" << input.sample_rate << " Hz) to "
                  << output_path << " (" << sample_rate << " Hz)\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        PrintUsage(argv[0]);
        return 1;
    }
}
