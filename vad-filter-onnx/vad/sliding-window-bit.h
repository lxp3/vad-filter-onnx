#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <iostream>
#include <string>

namespace VadFilterOnnx {

class SlidingWindowBit {
  public:
    SlidingWindowBit(size_t max_size) : window(0), current_size(0) {
        if (max_size > 64) {
            std::cerr << "Warning: SlidingWindowBit max_size (" << max_size
                      << ") exceeds 64. Capping to 64." << std::endl;
        }
        this->max_size = std::min(max_size, (size_t)64);
        // 构造掩码，用于清理移出窗口的高位数据
        mask = (this->max_size >= 64) ? ~0ULL : (1ULL << this->max_size) - 1;
    }

    void push(bool value) {
        // FIFO: 左移腾出最低位，填入新值
        window = ((window << 1) | (value ? 1ULL : 0ULL)) & mask;

        if (current_size < max_size) {
            current_size++;
        }
    }

    /**
     * @brief Check if speech is detected within a given window size and threshold.
     * @param win_size The window size to check (must be <= max_size).
     * @param threshold The number of frames that must be speech to trigger detection.
     */
    bool check_speech(size_t win_size, size_t threshold) const {
        if (current_size < win_size)
            return false;
        uint64_t sub_mask = (win_size >= 64) ? ~0ULL : (1ULL << win_size) - 1;
        uint64_t sub_window = window & sub_mask;
        return std::popcount(sub_window) >= threshold;
    }

    /**
     * @brief Check if silence is detected within a given window size and threshold.
     * @param win_size The window size to check (must be <= max_size).
     * @param threshold The number of frames that must be silence to trigger detection.
     */
    bool check_silence(size_t win_size, size_t threshold) const {
        if (current_size < win_size)
            return false;
        uint64_t sub_mask = (win_size >= 64) ? ~0ULL : (1ULL << win_size) - 1;
        uint64_t sub_window = window & sub_mask;
        size_t num_zeros = win_size - std::popcount(sub_window);
        return num_zeros >= threshold;
    }

    // 统计 1 的数量 (O(1))
    size_t get_num_ones() const { return std::popcount(window); }

    size_t get_num_zeros() const { return current_size - get_num_ones(); }

    // --- 连续性统计函数 ---

    // 从右侧（最新进入的一侧，低位）数连续 of 0
    size_t num_right_zeros() const {
        if (current_size == 0)
            return 0;
        // 如果最低位是 1，则返回 0；否则返回低位连续 0 的个数
        if (window & 1ULL)
            return 0;
        return std::min((size_t)std::countr_zero(window | ~mask | (1ULL << current_size)),
                        current_size);
    }

    // 从右侧（最新）数连续的 1
    size_t num_right_ones() const {
        if (current_size == 0)
            return 0;
        if (!(window & 1ULL))
            return 0;
        return std::countr_zero(~window);
    }

    // 从左侧（最旧进入的一侧）数连续 of 0
    size_t num_left_zeros() const {
        if (current_size == 0)
            return 0;
        // 需要对齐到窗口的“左端”
        // 窗口有效位在 [0, current_size-1]，最旧的位在 index = current_size-1
        uint64_t reversed_window = window << (64 - current_size);
        return std::countl_zero(reversed_window);
    }

    // 从左侧（最旧）数连续的 1
    size_t num_left_ones() const {
        if (current_size == 0)
            return 0;
        uint64_t reversed_window = window << (64 - current_size);
        return std::countl_one(reversed_window);
    }

    void reset() {
        window = 0;
        current_size = 0;
    }

    void reverse() { window = (~window) & mask; }

    std::string to_string() const {
        std::string s;
        s.reserve(current_size);
        for (int i = static_cast<int>(current_size) - 1; i >= 0; --i) {
            s += ((window >> i) & 1ULL) ? '1' : '0';
        }
        return s;
    }

  private:
    uint64_t window;
    uint64_t mask;
    size_t max_size;
    size_t current_size;
};

} // namespace VadFilterOnnx
