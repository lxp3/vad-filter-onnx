#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <string>
#include <vector>

namespace VadFilterOnnx {

class SlidingWindowBit {
  public:
    SlidingWindowBit(size_t max_size)
        : window(block_count(max_size), 0), max_size(max_size), current_size(0) {}

    void push(bool value) {
        // FIFO: shift left to make bit 0 available for the newest value.
        uint64_t carry = value ? 1ULL : 0ULL;
        for (auto &block : window) {
            uint64_t next_carry = block >> 63;
            block = (block << 1) | carry;
            carry = next_carry;
        }
        clear_unused_bits();

        if (current_size < max_size) {
            current_size++;
        }
    }

    /**
     * @brief Check if speech is detected within a given window size and threshold.
     * @param win_size The window size to check (must be <= max_size).
     */
    size_t check_speech(size_t win_size) const {
        if (current_size < win_size)
            return 0;
        return count_ones(std::min(win_size, max_size));
    }

    /**
     * @brief Check if silence is detected within a given window size and threshold.
     * @param win_size The window size to check (must be <= max_size).
     */
    size_t check_silence(size_t win_size) const {
        if (current_size < win_size)
            return 0;
        win_size = std::min(win_size, max_size);
        return win_size - count_ones(win_size);
    }

    // Count 1s in the current valid window.
    size_t get_num_ones() const { return count_ones(current_size); }

    size_t get_num_zeros() const { return current_size - get_num_ones(); }

    // --- 连续性统计函数 ---

    // 从右侧（最新进入的一侧，低位）数连续 of 0
    size_t num_right_zeros() const {
        if (current_size == 0)
            return 0;
        size_t count = 0;
        while (count < current_size && !bit_at(count)) {
            count++;
        }
        return count;
    }

    // 从右侧（最新）数连续的 1
    size_t num_right_ones() const {
        if (current_size == 0)
            return 0;
        size_t count = 0;
        while (count < current_size && bit_at(count)) {
            count++;
        }
        return count;
    }

    // 从左侧（最旧进入的一侧）数连续 of 0
    size_t num_left_zeros() const {
        if (current_size == 0)
            return 0;
        size_t count = 0;
        size_t index = current_size;
        while (index > 0) {
            index--;
            if (bit_at(index))
                break;
            count++;
        }
        return count;
    }

    // 从左侧（最旧）数连续的 1
    size_t num_left_ones() const {
        if (current_size == 0)
            return 0;
        size_t count = 0;
        size_t index = current_size;
        while (index > 0) {
            index--;
            if (!bit_at(index))
                break;
            count++;
        }
        return count;
    }

    void reset() {
        std::fill(window.begin(), window.end(), 0);
        current_size = 0;
    }

    void reverse() {
        for (auto &block : window) {
            block = ~block;
        }
        clear_bits_from(current_size);
    }

    std::string to_string() const {
        std::string s;
        s.reserve(current_size);
        size_t index = current_size;
        while (index > 0) {
            index--;
            s += bit_at(index) ? '1' : '0';
        }
        return s;
    }

  private:
    static size_t block_count(size_t size) { return (size + 63) / 64; }

    static uint64_t low_bits_mask(size_t bits) {
        if (bits == 0)
            return 0;
        if (bits >= 64)
            return ~0ULL;
        return (1ULL << bits) - 1;
    }

    bool bit_at(size_t index) const {
        return (window[index / 64] & (1ULL << (index % 64))) != 0;
    }

    size_t count_ones(size_t bits) const {
        size_t count = 0;
        size_t full_blocks = bits / 64;
        for (size_t i = 0; i < full_blocks; ++i) {
            count += std::popcount(window[i]);
        }

        size_t remaining_bits = bits % 64;
        if (remaining_bits > 0) {
            count += std::popcount(window[full_blocks] & low_bits_mask(remaining_bits));
        }
        return count;
    }

    void clear_unused_bits() { clear_bits_from(max_size); }

    void clear_bits_from(size_t first_unused_bit) {
        if (window.empty())
            return;

        size_t block_index = first_unused_bit / 64;
        size_t bit_offset = first_unused_bit % 64;
        if (block_index >= window.size())
            return;

        if (bit_offset == 0) {
            std::fill(window.begin() + block_index, window.end(), 0);
        } else {
            window[block_index] &= low_bits_mask(bit_offset);
            std::fill(window.begin() + block_index + 1, window.end(), 0);
        }
    }

    std::vector<uint64_t> window;
    size_t max_size;
    size_t current_size;
};

} // namespace VadFilterOnnx
