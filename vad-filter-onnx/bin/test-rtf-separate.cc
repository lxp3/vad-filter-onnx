#include "separate/bs-roformer-separator.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

struct Options {
    std::string model;
    int rate = 44100;
    int seconds = 5;
    int channels = 2;
    int warmups = 5;
    int runs = 20;
};

static void Usage(const char *p) {
    std::cout << "Usage: " << p
              << " --model-path PATH [--sample-rate N] [--audio-seconds N] [--channels 1|2] "
                 "[--num-warmups N] [--num-runs N]\nOffline only. Uses deterministic random float "
                 "PCM and times BsRoformerSeparator::Separate, excluding model load.\n";
}

static int Number(const std::string &v, const char *name) {
    std::size_t p = 0;
    int n;
    try {
        n = std::stoi(v, &p);
    } catch (...) {
        throw std::invalid_argument(std::string(name) + " requires an integer");
    }
    if (p != v.size())
        throw std::invalid_argument(std::string(name) + " requires an integer");
    return n;
}

static Options Parse(int argc, char **argv) {
    Options o;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") {
            Usage(argv[0]);
            std::exit(0);
        }
        if (++i >= argc)
            throw std::invalid_argument("missing value for " + a);
        std::string v = argv[i];
        if (a == "--model-path")
            o.model = v;
        else if (a == "--sample-rate")
            o.rate = Number(v, "--sample-rate");
        else if (a == "--audio-seconds")
            o.seconds = Number(v, "--audio-seconds");
        else if (a == "--channels")
            o.channels = Number(v, "--channels");
        else if (a == "--num-warmups")
            o.warmups = Number(v, "--num-warmups");
        else if (a == "--num-runs")
            o.runs = Number(v, "--num-runs");
        else
            throw std::invalid_argument("unknown argument: " + a);
    }

    if (o.model.empty() || o.rate <= 0 || o.seconds <= 0 || (o.channels != 1 && o.channels != 2) ||
        o.warmups < 0 || o.runs <= 0)
        throw std::invalid_argument("invalid benchmark options");
    return o;
}

static int ChunkCount(int64_t frames) {
    constexpr int64_t kChunk = 352800;
    constexpr int64_t kStep = 88200;
    if (frames <= kChunk)
        return 1;
    int count = 0;
    for (int64_t offset = 0; offset < frames; offset += kStep) {
        ++count;
        if (offset + kChunk >= frames)
            break;
    }
    return count;
}

static double Separate(VadFilterOnnx::BsRoformerSeparator &separator, const Options &o,
                       const std::vector<float> &samples, int64_t frames) {
    auto begin = std::chrono::steady_clock::now();
    auto result = separator.Separate(samples.data(), frames, o.channels, o.rate);
    if (result.vocals.size() != static_cast<std::size_t>(frames * 2) ||
        result.instrumental.size() != result.vocals.size())
        throw std::runtime_error("separator output has the wrong size");
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
}

int main(int argc, char **argv) {
    try {
        auto o = Parse(argc, argv);

        int64_t frames = static_cast<int64_t>(o.rate) * o.seconds;
        if (frames <= 0 || frames > std::numeric_limits<int64_t>::max() / o.channels)
            throw std::invalid_argument("random audio is too long");
        std::vector<float> samples(static_cast<std::size_t>(frames * o.channels));
        std::mt19937 g(20260723);
        std::uniform_real_distribution<float> d(-1.F, 1.F);
        std::generate(samples.begin(), samples.end(), [&] { return d(g); });

        std::cout << std::unitbuf << std::fixed << std::setprecision(6);
        std::cout << "loading model\n";
        VadFilterOnnx::BsRoformerSeparator separator(o.model, 1);
        std::cout << "mode: offline\n";
        std::cout << "frames: " << frames << ", channels: " << o.channels
                  << ", sample_rate: " << o.rate << ", chunks: " << ChunkCount(frames) << '\n';
        for (int i = 0; i < o.warmups; ++i) {
            double t = Separate(separator, o, samples, frames);
            std::cout << "Warmup " << i + 1 << ": " << t << " seconds\n";
        }

        double sum = 0;
        for (int i = 0; i < o.runs; ++i) {
            double t = Separate(separator, o, samples, frames);
            sum += t;
            std::cout << "Run " << i + 1 << ": " << t << " seconds, RTF = " << t / o.seconds
                      << '\n';
        }

        std::cout << "Average: " << sum / o.runs
                  << " seconds, average RTF = " << sum / o.runs / o.seconds << '\n';
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << '\n';
        Usage(argv[0]);
        return 1;
    }
}
