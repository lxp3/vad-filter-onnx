#include "vad-filter-onnx-cxx-api.h"

#include <algorithm>
#include <chrono>
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
    std::string mode = "online";
    int rate = 16000;
    int seconds = 5;
    int chunk_ms = 100;
    int warmups = 5;
    int runs = 20;
};

static void Usage(const char *p) {
    std::cout << "Usage: " << p
              << " --model-path PATH [--mode online|offline] [--sample-rate N] [--audio-seconds N] "
                 "[--chunk-ms N] [--num-warmups N] [--num-runs N]\nUses deterministic random float "
                 "PCM only; no audio files are read.\n";
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
        else if (a == "--mode")
            o.mode = v;
        else if (a == "--sample-rate")
            o.rate = Number(v, "--sample-rate");
        else if (a == "--audio-seconds")
            o.seconds = Number(v, "--audio-seconds");
        else if (a == "--chunk-ms")
            o.chunk_ms = Number(v, "--chunk-ms");
        else if (a == "--num-warmups")
            o.warmups = Number(v, "--num-warmups");
        else if (a == "--num-runs")
            o.runs = Number(v, "--num-runs");
        else
            throw std::invalid_argument("unknown argument: " + a);
    }

    if (o.model.empty() || (o.mode != "online" && o.mode != "offline") || o.rate <= 0 ||
        o.seconds <= 0 || o.chunk_ms <= 0 || o.warmups < 0 || o.runs <= 0)
        throw std::invalid_argument("invalid benchmark options");
    return o;
}

static double Decode(VadFilterOnnx::AutoVadModel *m, const Options &o, std::vector<float> &s) {
    m->reset();

    auto begin = std::chrono::steady_clock::now();
    if (o.mode == "offline") {
        m->decode(s.data(), static_cast<int>(s.size()), true);
    } else {
        std::size_t chunk = static_cast<std::size_t>(o.rate) * o.chunk_ms / 1000;
        for (std::size_t i = 0; i < s.size(); i += chunk) {
            auto n = std::min(chunk, s.size() - i);
            m->decode(s.data() + i, static_cast<int>(n), i + n == s.size());
        }
    }

    return std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
}

int main(int argc, char **argv) {
    try {
        auto o = Parse(argc, argv);

        int64_t total = static_cast<int64_t>(o.rate) * o.seconds;
        if (total > std::numeric_limits<int>::max())
            throw std::invalid_argument("random audio is too long");
        std::vector<float> samples(static_cast<std::size_t>(total));
        std::mt19937 g(20260723);
        std::uniform_real_distribution<float> d(-1, 1);
        std::generate(samples.begin(), samples.end(), [&] { return d(g); });

        auto h = o.model == "webrtc" ? VadFilterOnnx::AutoVadModel::create_webrtc()
                                     : VadFilterOnnx::AutoVadModel::create(o.model, 1);
        if (!h)
            throw std::runtime_error("failed to load model");
        VadFilterOnnx::VadConfig c;
        c.sample_rate = o.rate;
        auto m = h->init(c);
        if (!m)
            throw std::runtime_error("failed to initialize model");

        for (int i = 0; i < o.warmups; ++i)
            Decode(m.get(), o, samples);

        double sum = 0;
        std::cout << std::fixed << std::setprecision(6) << "mode: " << o.mode << '\n';
        for (int i = 0; i < o.runs; ++i) {
            double t = Decode(m.get(), o, samples);
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
