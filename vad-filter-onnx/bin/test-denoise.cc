#include "denoise-filter-onnx-cxx-api.h"
#include "resample.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using VadFilterOnnx::AutoDenoiseModel;
using VadFilterOnnx::DenoiseConfig;

struct Wav {
    int sample_rate;
    std::vector<float> samples;
};

static uint32_t ReadUint32(const char *data) {
    return static_cast<uint8_t>(data[0]) | (static_cast<uint8_t>(data[1]) << 8) |
           (static_cast<uint8_t>(data[2]) << 16) |
           (static_cast<uint8_t>(data[3]) << 24);
}

static uint16_t ReadUint16(const char *data) {
    return static_cast<uint8_t>(data[0]) | (static_cast<uint8_t>(data[1]) << 8);
}

static Wav ReadWav(const fs::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open " + path.string());
    }

    std::vector<char> bytes((std::istreambuf_iterator<char>(input)), {});
    if (bytes.size() < 44 || std::string(bytes.data(), 4) != "RIFF" ||
        std::string(bytes.data() + 8, 4) != "WAVE") {
        throw std::runtime_error("invalid WAV: " + path.string());
    }

    std::size_t position = 12;
    std::size_t data_position = 0;
    std::size_t data_size = 0;
    int sample_rate = 0;
    int channels = 0;
    int bits_per_sample = 0;
    while (position + 8 <= bytes.size()) {
        const std::string chunk_id(bytes.data() + position, 4);
        const uint32_t chunk_size = ReadUint32(bytes.data() + position + 4);
        position += 8;
        if (position + chunk_size > bytes.size()) {
            throw std::runtime_error("truncated WAV: " + path.string());
        }
        if (chunk_id == "fmt " && chunk_size >= 16) {
            if (ReadUint16(bytes.data() + position) != 1) {
                throw std::runtime_error("only PCM WAV supported");
            }
            channels = ReadUint16(bytes.data() + position + 2);
            sample_rate = ReadUint32(bytes.data() + position + 4);
            bits_per_sample = ReadUint16(bytes.data() + position + 14);
        } else if (chunk_id == "data") {
            data_position = position;
            data_size = chunk_size;
            break;
        }
        position += chunk_size + (chunk_size & 1);
    }

    if (!data_position || !sample_rate || channels != 1 || bits_per_sample != 16) {
        throw std::runtime_error("WAV must be mono 16-bit PCM: " + path.string());
    }
    data_size = std::min(data_size, bytes.size() - data_position);
    std::vector<float> samples(data_size / sizeof(int16_t));
    for (std::size_t index = 0; index < samples.size(); ++index) {
        const int16_t value = static_cast<int16_t>(
            ReadUint16(bytes.data() + data_position + index * sizeof(int16_t)));
        samples[index] = static_cast<float>(value) / 32768.0F;
    }
    return { sample_rate, std::move(samples) };
}

static void WriteWav(const fs::path &path, const std::vector<float> &samples,
                     int sample_rate) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("cannot write " + path.string());
    }
    const uint32_t data_size = static_cast<uint32_t>(samples.size() * sizeof(int16_t));
    const auto write_uint16 = [&output](uint16_t value) {
        output.put(static_cast<char>(value));
        output.put(static_cast<char>(value >> 8));
    };
    const auto write_uint32 = [&output](uint32_t value) {
        for (int shift = 0; shift < 4; ++shift) {
            output.put(static_cast<char>(value >> (8 * shift)));
        }
    };
    output.write("RIFF", 4);
    write_uint32(36 + data_size);
    output.write("WAVEfmt ", 8);
    write_uint32(16);
    write_uint16(1);
    write_uint16(1);
    write_uint32(sample_rate);
    write_uint32(sample_rate * sizeof(int16_t));
    write_uint16(sizeof(int16_t));
    write_uint16(16);
    output.write("data", 4);
    write_uint32(data_size);
    for (float sample : samples) {
        const int value = std::clamp<int>(sample * 32768.0F, -32768, 32767);
        write_uint16(static_cast<uint16_t>(static_cast<int16_t>(value)));
    }
}

static void PrintUsage(const char *program) {
    std::cerr << "Usage: " << program
              << " --model-path PATH --input-wav-dir DIR --output-wav-dir DIR"
                 " [--num-threads N] [--sample-rate N]\n";
}

int main(int argc, char **argv) {
    std::string model_path;
    std::string input_wav_dir;
    std::string output_wav_dir;
    int num_threads = 20;
    int sample_rate = 16000;
    try {
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "-h" || argument == "--help") {
                PrintUsage(argv[0]);
                return 0;
            }
            if (index + 1 >= argc) {
                throw std::invalid_argument("missing value for " + argument);
            }
            if (argument == "--model-path") {
                model_path = argv[++index];
            } else if (argument == "--input-wav-dir") {
                input_wav_dir = argv[++index];
            } else if (argument == "--output-wav-dir") {
                output_wav_dir = argv[++index];
            } else if (argument == "--num-threads") {
                num_threads = std::stoi(argv[++index]);
            } else if (argument == "--sample-rate") {
                sample_rate = std::stoi(argv[++index]);
            } else {
                throw std::invalid_argument("unknown argument: " + argument);
            }
        }
        if (model_path.empty() || input_wav_dir.empty() || output_wav_dir.empty() ||
            num_threads < 1 || sample_rate < 1) {
            throw std::invalid_argument("required arguments missing or invalid");
        }

        fs::create_directories(output_wav_dir);
        std::vector<fs::path> wav_files;
        for (const auto &entry : fs::directory_iterator(input_wav_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".wav") {
                wav_files.push_back(entry.path());
            }
        }

        auto handle = AutoDenoiseModel::create(model_path, 1);
        if (!handle) {
            throw std::runtime_error("failed to load model");
        }

        DenoiseConfig config;
        config.sample_rate = sample_rate;
        const std::size_t worker_count =
            std::min<std::size_t>(num_threads, std::max<std::size_t>(1, wav_files.size()));
        std::vector<std::unique_ptr<AutoDenoiseModel>> models;
        models.reserve(worker_count);
        for (std::size_t index = 0; index < worker_count; ++index) {
            auto model = handle->init(config);
            if (!model) {
                throw std::runtime_error("failed to initialize model instance");
            }
            models.push_back(std::move(model));
        }

        std::atomic<std::size_t> next_file = 0;
        std::atomic<bool> failed = false;
        std::exception_ptr worker_error;
        std::mutex error_mutex;
        std::mutex output_mutex;
        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        for (std::size_t worker_index = 0; worker_index < worker_count; ++worker_index) {
            workers.emplace_back([&, worker_index] {
                try {
                    AutoDenoiseModel *model = models[worker_index].get();
                    while (!failed.load()) {
                        const std::size_t file_index = next_file.fetch_add(1);
                        if (file_index >= wav_files.size()) {
                            break;
                        }

                        const fs::path &wav_file = wav_files[file_index];
                        Wav wav = ReadWav(wav_file);
                        std::vector<float> samples = std::move(wav.samples);
                        if (wav.sample_rate != sample_rate) {
                            const float cutoff =
                                0.99F * 0.5F * static_cast<float>(
                                                   std::min(wav.sample_rate, sample_rate));
                            sherpa_onnx::LinearResample resampler(
                                wav.sample_rate, sample_rate, cutoff, 6);
                            std::vector<float> resampled;
                            resampler.Resample(samples.data(),
                                               static_cast<int32_t>(samples.size()), true,
                                               &resampled);
                            samples = std::move(resampled);
                        }

                        model->reset();
                        const std::vector<float> denoised = model->decode(
                            samples.data(), static_cast<int>(samples.size()), true);
                        WriteWav(fs::path(output_wav_dir) / wav_file.filename(), denoised,
                                 sample_rate);
                        std::lock_guard<std::mutex> lock(output_mutex);
                        std::cout << "Processed " << wav_file << '\n';
                    }
                } catch (...) {
                    failed.store(true);
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (!worker_error) {
                        worker_error = std::current_exception();
                    }
                }
            });
        }
        for (auto &worker : workers) {
            worker.join();
        }
        if (worker_error) {
            std::rethrow_exception(worker_error);
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        PrintUsage(argv[0]);
        return 1;
    }
}
