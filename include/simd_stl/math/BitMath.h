#pragma once 

#include <src/simd_stl/math/CountTrailingZeros.h>
#include <src/simd_stl/math/CountLeadingZeros.h>

__SIMD_STL_MATH_NAMESPACE_BEGIN

template <typename _Type_>
constexpr simd_stl_always_inline _Type_ ClearLeftMostSet(const _Type_ _Value) {
    return _Value & (_Value - 1);
}

template <typename _IntegralType_>
constexpr simd_stl_always_inline int CountTrailingZeroBits(_IntegralType_ _Value) noexcept {
    static_assert(std::is_unsigned_v<_IntegralType_>);

#if defined(simd_stl_processor_x86) && !defined(simd_stl_processor_arm)
    if (!type_traits::is_constant_evaluated()) {
        if (arch::ProcessorFeatures::AVX2())
            // Поддерживается tzcnt
            return _TzcntCountTrailingZeroBits(_Value);
        else
            return _BsfCountTrailingZeroBits(_Value);
    }
    else
#endif // defined(simd_stl_processor_x86) && !defined(simd_stl_processor_arm)
        return _BitHacksCountTrailingZeroBits(_Value);
}

template <typename _IntegralType_>
constexpr simd_stl_always_inline int CountLeadingZeroBits(_IntegralType_ _Value) noexcept {
    static_assert(std::is_unsigned_v<_IntegralType_>);

#if defined(simd_stl_processor_x86) && !defined(simd_stl_processor_arm)
    if (!type_traits::is_constant_evaluated()) {
        if (arch::ProcessorFeatures::AVX2())
            // Поддерживается lzcnt
            return _LzcntCountLeadingZeroBits(_Value);
        else
            return _BsrCountLeadingZeroBits(_Value);
    }
    else
#endif // defined(simd_stl_processor_x86) && !defined(simd_stl_processor_arm)
        return _BitHacksCountLeadingZeroBits(_Value);
}

 
template <typename _IntegralType_>
constexpr simd_stl_always_inline int PopulationCount(_IntegralType_ _Value) noexcept {
    static_assert(std::is_unsigned_v<_IntegralType_>);

#if (defined(simd_stl_processor_x86_32) || defined(simd_stl_processor_x86_64) || defined(simd_stl_processor_arm_64))
    if (!type_traits::is_constant_evaluated()) {
        if (arch::ProcessorFeatures::POPCNT())
            // Поддерживается popcnt
            return _PopcntPopulationCount(_Value);
    }
#endif // (defined(simd_stl_processor_x86_32) || defined(simd_stl_processor_x86_64) || defined(simd_stl_processor_arm_64))
    
    return _BitHacksPopulationCount(_Value);
}

__SIMD_STL_MATH_NAMESPACE_END