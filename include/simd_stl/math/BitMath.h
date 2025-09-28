#pragma once 

#include <src/simd_stl/math/CountTrailingZeros.h>
#include <src/simd_stl/math/CountLeadingZeros.h>

__SIMD_STL_MATH_NAMESPACE_BEGIN

template <typename _Type_>
constexpr inline _Type_ ClearLeftMostSet(const _Type_ value) {
    return value & (value - 1);
}

template <type_traits::standard_unsigned_integral _IntegralType_>
constexpr inline int CountTrailingZeroBits(_IntegralType_ value) noexcept {
#if defined(simd_stl_processor_x86) && !defined(simd_stl_processor_arm)
    if (!type_traits::is_constant_evaluated()) {
        if (arch::ProcessorFeatures::AVX2())
            // Поддерживается tzcnt
            return _TzcntCountTrailingZeroBits(value);
        else
            return _BsfCountTrailingZeroBits(value);
    }
    else
#endif // defined(simd_stl_processor_x86) && !defined(simd_stl_processor_arm)
        return _BitHacksCountTrailingZeroBits(value);
}

template <type_traits::standard_unsigned_integral _IntegralType_>
constexpr inline int CountLeadingZeroBits(_IntegralType_ value) noexcept {
#if defined(simd_stl_processor_x86) && !defined(simd_stl_processor_arm)
    if (!type_traits::is_constant_evaluated()) {
        if (arch::ProcessorFeatures::AVX2())
            // Поддерживается lzcnt
            return _LzcntCountLeadingZeroBits(value);
        else
            return _BsrCountLeadingZeroBits(value);
    }
    else
#endif // defined(simd_stl_processor_x86) && !defined(simd_stl_processor_arm)
        return _BitHacksCountLeadingZeroBits(value);
}

 
template <type_traits::standard_unsigned_integral _IntegralType_>
constexpr inline int PopulationCount(_IntegralType_ value) noexcept {
#if (defined(simd_stl_processor_x86_32) || defined(simd_stl_processor_x86_64) || defined(simd_stl_processor_arm_64))
    if (!type_traits::is_constant_evaluated()) {
        if (arch::ProcessorFeatures::POPCNT())
            // Поддерживается popcnt
            return _PopcntPopulationCount(value);
    }
#endif // (defined(simd_stl_processor_x86_32) || defined(simd_stl_processor_x86_64) || defined(simd_stl_processor_arm_64))
    
    return _BitHacksPopulationCount(value);
}

__SIMD_STL_MATH_NAMESPACE_END