#define NOMINMAX

#include <simd_stl/math/BitMath.h>
#include <src/simd_stl/utility/Assert.h>

#include <iostream>
#include <limits>


void test_unsigned_int() {
    simd_stl_assert(simd_stl::math::count_trailing_zero_bits(0u) == std::numeric_limits<unsigned int>::digits);

    simd_stl_assert(simd_stl::math::count_trailing_zero_bits(1u) == 0);
    simd_stl_assert(simd_stl::math::count_trailing_zero_bits(2u) == 1);


    simd_stl_assert(simd_stl::math::count_trailing_zero_bits(4u) == 2); 
    simd_stl_assert(simd_stl::math::count_trailing_zero_bits(0b00101000u) == 3);

    simd_stl_assert(simd_stl::math::count_trailing_zero_bits(0b10000000000000000000000000000000u) == 31);
    simd_stl_assert(simd_stl::math::count_trailing_zero_bits(std::numeric_limits<unsigned int>::min() + 0x80000000u) == 31);

    simd_stl_assert(simd_stl::math::count_trailing_zero_bits(std::numeric_limits<unsigned int>::max()) == 0);
}

void test_unsigned_long_long() {
    simd_stl_assert(simd_stl::math::count_trailing_zero_bits(0ULL) == std::numeric_limits<unsigned long long>::digits);

    simd_stl_assert(simd_stl::math::count_trailing_zero_bits(1ULL) == 0);
    simd_stl_assert(simd_stl::math::count_trailing_zero_bits(2ULL) == 1);

    simd_stl_assert(simd_stl::math::count_trailing_zero_bits(0x8000000000000000ULL) == 63);
    simd_stl_assert(simd_stl::math::count_trailing_zero_bits(std::numeric_limits<unsigned long long>::min() + 0x8000000000000000ULL) == 63);

    simd_stl_assert(simd_stl::math::count_trailing_zero_bits(0b0010100000000000000000000000000000000000000000000000000000000000ULL) == 59);
    simd_stl_assert(simd_stl::math::count_trailing_zero_bits(std::numeric_limits<unsigned long long>::max()) == 0);
}

void test_signed_int_non_negative() {
    simd_stl_assert(simd_stl::math::count_trailing_zero_bits(1U) == 0);
    simd_stl_assert(simd_stl::math::count_trailing_zero_bits(8U) == 3);
    simd_stl_assert(simd_stl::math::count_trailing_zero_bits(0x40000000U) == 30);
}

int main() {
    test_unsigned_int();
    test_unsigned_long_long();
    test_signed_int_non_negative();

    return 0;
}