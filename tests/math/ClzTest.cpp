#define NOMINMAX

#include <simd_stl/math/BitMath.h>
#include <src/simd_stl/utility/Assert.h>

#include <iostream>
#include <limits>

#include <bit>


void test_clz_unsigned_int() {
    Assert(simd_stl::math::CountLeadingZeroBits(0u) == std::numeric_limits<unsigned int>::digits);
    Assert(simd_stl::math::CountLeadingZeroBits(1u) == std::numeric_limits<unsigned int>::digits - 1);

    Assert(simd_stl::math::CountLeadingZeroBits(2u) == std::numeric_limits<unsigned int>::digits - 2);
    Assert(simd_stl::math::CountLeadingZeroBits(0x80000000u) == 0);

    Assert(simd_stl::math::CountLeadingZeroBits(0x40000000u) == 1);
    Assert(simd_stl::math::CountLeadingZeroBits(0b00101000u) == std::numeric_limits<unsigned int>::digits - 6);

    Assert(simd_stl::math::CountLeadingZeroBits(std::numeric_limits<unsigned int>::max()) == 0);
}

void test_clz_unsigned_long_long() {
    Assert(simd_stl::math::CountLeadingZeroBits(0ULL) == std::numeric_limits<unsigned long long>::digits);
    Assert(simd_stl::math::CountLeadingZeroBits(1ULL) == std::numeric_limits<unsigned long long>::digits - 1);

    Assert(simd_stl::math::CountLeadingZeroBits(2ULL) == std::numeric_limits<unsigned long long>::digits - 2);
    Assert(simd_stl::math::CountLeadingZeroBits(0x8000000000000000ULL) == 0);

    Assert(simd_stl::math::CountLeadingZeroBits(0x4000000000000000ULL) == 1);
    Assert(simd_stl::math::CountLeadingZeroBits(0b0010100000000000000000000000000000000000000000000000000000000000ULL) == 2);

    Assert(simd_stl::math::CountLeadingZeroBits(std::numeric_limits<unsigned long long>::max()) == 0);
}

void test_clz_signed_int_non_negative() {
    Assert(simd_stl::math::CountLeadingZeroBits(1U) == (std::numeric_limits<unsigned int>::digits - 1));
    Assert(simd_stl::math::CountLeadingZeroBits(8U) == std::numeric_limits<unsigned int>::digits - 4);
    Assert(simd_stl::math::CountLeadingZeroBits(0x40000000U) == 1);
}

int main() {
    test_clz_unsigned_int();
    test_clz_unsigned_long_long();
    test_clz_signed_int_non_negative();

    return 0;
}