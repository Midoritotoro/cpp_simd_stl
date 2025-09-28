#define NOMINMAX

#include <simd_stl/math/BitMath.h>
#include <src/simd_stl/utility/Assert.h>

#include <iostream>
#include <limits>


void test_unsigned_int() {
    Assert(simd_stl::math::CountTrailingZeroBits(0u) == std::numeric_limits<unsigned int>::digits);

    Assert(simd_stl::math::CountTrailingZeroBits(1u) == 0);
    Assert(simd_stl::math::CountTrailingZeroBits(2u) == 1);


    Assert(simd_stl::math::CountTrailingZeroBits(4u) == 2); 
    Assert(simd_stl::math::CountTrailingZeroBits(0b00101000u) == 3);

    Assert(simd_stl::math::CountTrailingZeroBits(0b10000000000000000000000000000000u) == 31);
    Assert(simd_stl::math::CountTrailingZeroBits(std::numeric_limits<unsigned int>::min() + 0x80000000u) == 31);

    Assert(simd_stl::math::CountTrailingZeroBits(std::numeric_limits<unsigned int>::max()) == 0);
}

void test_unsigned_long_long() {
    Assert(simd_stl::math::CountTrailingZeroBits(0ULL) == std::numeric_limits<unsigned long long>::digits);

    Assert(simd_stl::math::CountTrailingZeroBits(1ULL) == 0);
    Assert(simd_stl::math::CountTrailingZeroBits(2ULL) == 1);

    Assert(simd_stl::math::CountTrailingZeroBits(0x8000000000000000ULL) == 63);
    Assert(simd_stl::math::CountTrailingZeroBits(std::numeric_limits<unsigned long long>::min() + 0x8000000000000000ULL) == 63);

    Assert(simd_stl::math::CountTrailingZeroBits(0b0010100000000000000000000000000000000000000000000000000000000000ULL) == 59);
    Assert(simd_stl::math::CountTrailingZeroBits(std::numeric_limits<unsigned long long>::max()) == 0);
}

void test_signed_int_non_negative() {
    Assert(simd_stl::math::CountTrailingZeroBits(1U) == 0);
    Assert(simd_stl::math::CountTrailingZeroBits(8U) == 3);
    Assert(simd_stl::math::CountTrailingZeroBits(0x40000000U) == 30);
}

int main() {
    test_unsigned_int();
    test_unsigned_long_long();
    test_signed_int_non_negative();

    return 0;
}