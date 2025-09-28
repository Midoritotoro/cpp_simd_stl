#define NOMINMAX

#include <simd_stl/math/BitMath.h>
#include <src/simd_stl/utility/Assert.h>

#include <iostream>
#include <limits>

void test_popcount_unsigned_int() {
    Assert(simd_stl::math::PopulationCount(0u) == 0);
    Assert(simd_stl::math::PopulationCount(1u) == 1);

    Assert(simd_stl::math::PopulationCount(2u) == 1);
    Assert(simd_stl::math::PopulationCount(3u) == 2);

    Assert(simd_stl::math::PopulationCount(0x80000000u) == 1);
    Assert(simd_stl::math::PopulationCount(0b00101000u) == 2);
    Assert(simd_stl::math::PopulationCount(0b10101010u) == 4);

    Assert(simd_stl::math::PopulationCount(std::numeric_limits<unsigned int>::max()) == std::numeric_limits<unsigned int>::digits); // 0b111...111
}

void test_popcount_unsigned_long_long() {
    Assert(simd_stl::math::PopulationCount(0ULL) == 0);
    Assert(simd_stl::math::PopulationCount(1ULL) == 1);

    Assert(simd_stl::math::PopulationCount(3ULL) == 2);
    Assert(simd_stl::math::PopulationCount(0x8000000000000000ULL) == 1);
    Assert(simd_stl::math::PopulationCount(0b0010100000000000000000000000000000000000000000000000000000000000ULL) == 2);

    Assert(simd_stl::math::PopulationCount(0b1101101101101101101101101101101101101101101101101101101101101101ULL) == 43);
    Assert(simd_stl::math::PopulationCount(std::numeric_limits<unsigned long long>::max()) == std::numeric_limits<unsigned long long>::digits); // 0b111...111
}

void test_popcount_signed_int_non_negative() {
    Assert(simd_stl::math::PopulationCount(0U) == 0);
    Assert(simd_stl::math::PopulationCount(7U) == 3);
    Assert(simd_stl::math::PopulationCount(0x40000001U) == 2);
}

int main() {
    test_popcount_unsigned_int();
    test_popcount_unsigned_long_long();
    test_popcount_signed_int_non_negative();

    return 0;
}