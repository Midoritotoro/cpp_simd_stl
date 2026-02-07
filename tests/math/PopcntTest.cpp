#define NOMINMAX

#include <simd_stl/math/BitMath.h>
#include <src/simd_stl/utility/Assert.h>

#include <iostream>
#include <limits>

void test_popcount_unsigned_int() {
    simd_stl_assert(simd_stl::math::population_count(0u) == 0);
    simd_stl_assert(simd_stl::math::population_count(1u) == 1);

    simd_stl_assert(simd_stl::math::population_count(2u) == 1);
    simd_stl_assert(simd_stl::math::population_count(3u) == 2);

    simd_stl_assert(simd_stl::math::population_count(0x80000000u) == 1);
    simd_stl_assert(simd_stl::math::population_count(0b00101000u) == 2);
    simd_stl_assert(simd_stl::math::population_count(0b10101010u) == 4);

    simd_stl_assert(simd_stl::math::population_count(std::numeric_limits<unsigned int>::max()) == std::numeric_limits<unsigned int>::digits); // 0b111...111
}

void test_popcount_unsigned_long_long() {
    simd_stl_assert(simd_stl::math::population_count(0ULL) == 0);
    simd_stl_assert(simd_stl::math::population_count(1ULL) == 1);

    simd_stl_assert(simd_stl::math::population_count(3ULL) == 2);
    simd_stl_assert(simd_stl::math::population_count(0x8000000000000000ULL) == 1);
    simd_stl_assert(simd_stl::math::population_count(0b0010100000000000000000000000000000000000000000000000000000000000ULL) == 2);

    simd_stl_assert(simd_stl::math::population_count(0b1101101101101101101101101101101101101101101101101101101101101101ULL) == 43);
    simd_stl_assert(simd_stl::math::population_count(std::numeric_limits<unsigned long long>::max()) == std::numeric_limits<unsigned long long>::digits); // 0b111...111
}

void test_popcount_signed_int_non_negative() {
    simd_stl_assert(simd_stl::math::population_count(0U) == 0);
    simd_stl_assert(simd_stl::math::population_count(7U) == 3);
    simd_stl_assert(simd_stl::math::population_count(0x40000001U) == 2);
}

int main() {
    test_popcount_unsigned_int();
    test_popcount_unsigned_long_long();
    test_popcount_signed_int_non_negative();

    return 0;
}