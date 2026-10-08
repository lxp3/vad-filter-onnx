#include "utils.h"
#include "vad-filter-onnx-cxx-api.h"

#include <algorithm>
#include <atomic>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace fs = std::filesystem;
using VadFilterOnnx::AutoVadModel;
using VadFilterOnnx::VadConfig;

struct Options {
    std::string model_path, path, output, save_dir;
    int num_threads = 4, chunk_ms = 100;
    VadConfig config;
};

static void Usage(const char *program) {
    std::cout << "Usage: " << program << " --model-path PATH --path WAV|DIR|TXT|SCP [options]\n"
              << "  --num-threads N --sample-rate N --chunk-ms N --output PATH --save-dir DIR\n"
              << "  --threshold N --max-speech-ms N --left-padding-ms N --right-padding-ms N\n";
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
        else if (argument == "--output")
            options.output = value;
        else if (argument == "--save-dir")
            options.save_dir = value;
        else if (argument == "--num-threads")
            options.num_threads = VadBin::ParseInt(value, argument);
        else if (argument == "--sample-rate")
            options.config.sample_rate = VadBin::ParseInt(value, argument);
        else if (argument == "--chunk-ms")
            options.chunk_ms = VadBin::ParseInt(value, argument);
        else if (argument == "--threshold")
            options.config.threshold = VadBin::ParseFloat(value, argument);
        else if (argument == "--speech-win-size-ms")
            options.config.speech_window_size_ms = VadBin::ParseInt(value, argument);
        else if (argument == "--speech-win-thr-ms")
            options.config.speech_window_threshold_ms = VadBin::ParseInt(value, argument);
        else if (argument == "--silence-win-size-ms")
            options.config.silence_window_size_ms = VadBin::ParseInt(value, argument);
        else if (argument == "--silence-win-thr-ms")
            options.config.silence_window_threshold_ms = VadBin::ParseInt(value, argument);
        else if (argument == "--max-speech-ms")
            options.config.max_speech_ms = VadBin::ParseInt(value, argument);
        else if (argument == "--left-padding-ms")
            options.config.left_padding_ms = VadBin::ParseInt(value, argument);
        else if (argument == "--right-padding-ms")
            options.config.right_padding_ms = VadBin::ParseInt(value, argument);
        else if (argument == "--webrtc-vad-mode")
            options.config.webrtc_vad_mode = VadBin::ParseInt(value, argument);
        else if (argument == "--webrtc-frame-ms")
            options.config.webrtc_frame_ms = VadBin::ParseInt(value, argument);
        else
            throw std::invalid_argument("unknown argument: " + argument);
    }
    if (options.model_path.empty() || options.path.empty())
        throw std::invalid_argument("--model-path and --path are required");
    if (options.num_threads <= 0 || options.config.sample_rate <= 0 || options.chunk_ms <= 0)
        throw std::invalid_argument("invalid thread, sample rate, or chunk size");
    return options;
}

int main(int argc, char **argv) {
    try {
        const Options options = Parse(argc, argv);
        auto handle = options.model_path == "webrtc" ? AutoVadModel::create_webrtc()
                                                     : AutoVadModel::create(options.model_path, 1);
        if (!handle)
            throw std::runtime_error("failed to load model");
        const auto files = VadBin::CollectAudio(options.path);
        std::ofstream result_file;
        if (!options.output.empty()) {
            result_file.open(options.output);
            if (!result_file)
                throw std::runtime_error("cannot write output");
        }
        std::vector<std::string> outputs(files.size()), errors(files.size());
        std::atomic<std::size_t> next{ 0 };
        const auto workers = std::min<std::size_t>(options.num_threads, files.size());
        std::vector<std::thread> threads;
        for (std::size_t worker = 0; worker < workers; ++worker)
            threads.emplace_back([&, worker] {
                auto model = handle->init(options.config);
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
                        auto samples = VadBin::Resample(input.samples, input.sample_rate,
                                                        options.config.sample_rate);
                        model->reset();
                        std::ostringstream text;
                        text << std::fixed << std::setprecision(3);
                        const std::size_t chunk =
                            static_cast<std::size_t>(options.config.sample_rate) *
                            options.chunk_ms / 1000;
                        for (std::size_t offset = 0; offset < samples.size(); offset += chunk) {
                            const auto count = std::min(chunk, samples.size() - offset);
                            for (const auto &segment :
                                 model->decode(samples.data() + offset, static_cast<int>(count),
                                               offset + count == samples.size())) {
                                if (segment.end_ms <= segment.start_ms)
                                    continue;
                                const auto key =
                                    files[index].first + "-" + std::to_string(segment.idx);
                                text << key << ' ' << segment.start_ms * .001 << ' '
                                     << segment.end_ms * .001 << '\n';
                                if (!options.save_dir.empty()) {
                                    const auto begin =
                                        std::clamp<int64_t>(static_cast<int64_t>(segment.start_ms) *
                                                                options.config.sample_rate / 1000,
                                                            0, samples.size());
                                    const auto end =
                                        std::clamp<int64_t>(static_cast<int64_t>(segment.end_ms) *
                                                                options.config.sample_rate / 1000,
                                                            begin, samples.size());
                                    const fs::path target =
                                        fs::path(options.save_dir) / (key + ".wav");
                                    fs::create_directories(target.parent_path());
                                    VadBin::SaveWav(target,
                                                    std::vector<float>(samples.begin() + begin,
                                                                       samples.begin() + end),
                                                    options.config.sample_rate);
                                }
                            }
                        }
                        outputs[index] = text.str();
                    } catch (const std::exception &error) {
                        errors[index] = error.what();
                    }
                }
            });
        for (auto &thread : threads)
            thread.join();
        std::ostream &output = options.output.empty() ? std::cout : result_file;
        std::size_t failed = 0;
        for (std::size_t index = 0; index < files.size(); ++index) {
            if (!errors[index].empty()) {
                ++failed;
                std::cerr << files[index].second << ": " << errors[index] << '\n';
            } else
                output << outputs[index];
        }
        std::cerr << "Processed: " << files.size() - failed << ", failed: " << failed << '\n';
        return failed ? 1 : 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        Usage(argv[0]);
        return 1;
    }
}
