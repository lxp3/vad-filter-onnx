#include "denoise-filter-onnx-cxx-api.h"
#include "utils.h"

#include <atomic>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

namespace fs = std::filesystem;
using VadFilterOnnx::AutoDenoiseModel;

struct Options {
    std::string model_path, path, output_dir;
    int sample_rate = 16000, num_threads = 4;
};

static void Usage(const char *program) {
    std::cout << "Usage: " << program
              << " --model-path PATH --path WAV|DIR|TXT|SCP --output-dir DIR [--sample-rate N] "
                 "[--num-threads N]\n";
}
static Options Parse(int argc, char **argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "-h" || argument == "--help") {
            Usage(argv[0]);
            std::exit(0);
        }
        if (++index >= argc)
            throw std::invalid_argument("missing value for " + argument);
        const std::string value = argv[index];
        if (argument == "--model-path")
            options.model_path = value;
        else if (argument == "--path")
            options.path = value;
        else if (argument == "--output-dir")
            options.output_dir = value;
        else if (argument == "--sample-rate")
            options.sample_rate = VadBin::ParseInt(value, argument);
        else if (argument == "--num-threads")
            options.num_threads = VadBin::ParseInt(value, argument);
        else
            throw std::invalid_argument("unknown argument: " + argument);
    }
    if (options.model_path.empty() || options.path.empty() || options.output_dir.empty())
        throw std::invalid_argument("--model-path, --path, and --output-dir are required");
    if (options.sample_rate <= 0 || options.num_threads <= 0)
        throw std::invalid_argument("invalid sample rate or thread count");
    return options;
}

int main(int argc, char **argv) {
    try {
        const Options options = Parse(argc, argv);
        auto handle = AutoDenoiseModel::create(options.model_path, 1);
        if (!handle)
            throw std::runtime_error("failed to load model");
        const auto files = VadBin::CollectAudio(options.path);
        std::atomic<std::size_t> next{ 0 };
        std::vector<std::string> errors(files.size());
        const auto workers = std::min<std::size_t>(options.num_threads, files.size());
        std::vector<std::thread> threads;
        for (std::size_t worker = 0; worker < workers; ++worker)
            threads.emplace_back([&, worker] {
                auto model = handle->init({ options.sample_rate });
                if (!model) {
                    errors[worker] = "failed to initialize model";
                    return;
                }
                while (true) {
                    const auto index = next.fetch_add(1);
                    if (index >= files.size())
                        return;
                    try {
                        auto input = VadBin::LoadWav(files[index].second);
                        auto samples =
                            VadBin::Resample(input.samples, input.sample_rate, options.sample_rate);
                        model->reset();
                        auto output =
                            model->decode(samples.data(), static_cast<int>(samples.size()), true);
                        const auto &key = files[index].first;
                        fs::path target = fs::path(options.output_dir) / key;
                        if (target.extension() != ".wav")
                            target += ".wav";
                        fs::create_directories(target.parent_path());
                        VadBin::SaveWav(target, output, options.sample_rate);
                    } catch (const std::exception &error) {
                        errors[index] = error.what();
                    }
                }
            });
        for (auto &thread : threads)
            thread.join();
        std::size_t failed = 0;
        for (std::size_t index = 0; index < files.size(); ++index)
            if (!errors[index].empty()) {
                ++failed;
                std::cerr << files[index].second << ": " << errors[index] << '\n';
            }
        std::cerr << "Processed: " << files.size() - failed << ", failed: " << failed << '\n';
        return failed ? 1 : 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        Usage(argv[0]);
        return 1;
    }
}
