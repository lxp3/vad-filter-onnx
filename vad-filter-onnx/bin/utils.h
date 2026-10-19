#pragma once

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace VadBin {

struct Audio {
    int sample_rate = 0;
    std::vector<float> samples;
};

int ParseInt(const std::string &value, const std::string &option);
float ParseFloat(const std::string &value, const std::string &option);
Audio LoadWav(const std::filesystem::path &path);
void SaveWav(const std::filesystem::path &path, const std::vector<float> &samples, int sample_rate);
std::vector<float> Resample(const std::vector<float> &samples, int input_rate, int output_rate);
std::vector<std::pair<std::string, std::filesystem::path> > CollectAudio(
    const std::filesystem::path &path);

} // namespace VadBin
