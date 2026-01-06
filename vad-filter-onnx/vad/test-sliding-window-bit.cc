#include "sliding-window-bit.h"
#include <cassert>
#include <iostream>
#include <vector>

using namespace VadFilterOnnx;

int main() {
    std::cout << "Testing SlidingWindowBit..." << std::endl;

    // Test basic push and get_num_ones
    SlidingWindowBit sw(10);
    sw.push(true);
    sw.push(true);
    sw.push(false);
    sw.push(true);

    std::cout << "sw(10, 5) with FIFO(1 1 0 1): " << sw.to_string() << std::endl;
    assert(sw.get_num_ones() == 3);
    assert(sw.get_num_zeros() == 1);
    assert(sw.is_up() == false); // 3 <= 5

    sw.push(true);
    sw.push(true);
    sw.push(true); // Now 6 ones
    std::cout << "sw(10, 5) with FIFO(1 1 0 1 1 1): " << sw.to_string() << std::endl;
    assert(sw.get_num_ones() == 6);
    assert(sw.is_up() == true);

    // Test window sliding
    // Push 10 more zeros
    for (int i = 0; i < 10; ++i)
        sw.push(false);
    std::cout << "sw(10, 5) with FIFO(0 0 0 0 0 0 0 0 0 0): " << sw.to_string() << std::endl;
    assert(sw.get_num_ones() == 0);
    assert(sw.get_num_zeros() == 10);
    assert(sw.is_down() == true);

    // Test continuity
    sw.reset();
    std::cout << "sw(10, 5) with reset: " << sw.to_string() << std::endl;
    sw.push(true);  // [1]
    sw.push(true);  // [1, 1]
    sw.push(false); // [1, 1, 0] (right is 0)
    std::cout << "sw(10, 5) with FIFO(1 1 0): " << sw.to_string() << std::endl;
    assert(sw.num_right_zeros() == 1);
    assert(sw.num_right_ones() == 0);
    assert(sw.num_left_ones() == 2);
    assert(sw.num_left_zeros() == 0);

    sw.push(true); // [1, 1, 0, 1]
    assert(sw.num_right_ones() == 1);
    assert(sw.num_right_zeros() == 0);

    // Test max size limit (64)
    SlidingWindowBit sw64(100); // Should be capped at 64
    for (int i = 0; i < 100; ++i)
        sw64.push(true);
    assert(sw64.get_num_ones() == 64);

    // Test to_string()
    SlidingWindowBit sw_str(5);
    sw_str.push(1);
    sw_str.push(0);
    assert(sw_str.to_string() == "10");

    for (int i = 0; i < 10; ++i)
        sw_str.push(1);
    assert(sw_str.to_string() == "11111");
    std::cout << "to_string() test passed: " << sw_str.to_string() << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
