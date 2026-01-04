#pragma once 

#include <src/simd_stl/math/CountTrailingZeros.h>
#include <src/simd_stl/math/CountLeadingZeros.h>

__SIMD_STL_MATH_NAMESPACE_BEGIN

template <typename _Type_>
constexpr simd_stl_always_inline _Type_ clear_left_most_set(const _Type_ __value) {
    return __value & (__value - 1);
}

template <typename _IntegralType_>
constexpr simd_stl_always_inline int count_trailing_zero_bits(_IntegralType_ __value) noexcept {
    static_assert(std::is_unsigned_v<_IntegralType_>);

#if defined(simd_stl_processor_x86) && !defined(simd_stl_processor_arm)
    if (!type_traits::is_constant_evaluated()) {
        if (arch::ProcessorFeatures::AVX2())
            return __tzcnt_ctz(__value);
        else
            return __bsf_ctz(__value);
    }
    else
#endif // defined(simd_stl_processor_x86) && !defined(simd_stl_processor_arm
    {
        return __bit_hacks_ctz(__value);
    }
}

template <typename _IntegralType_>
constexpr simd_stl_always_inline int count_leading_zero_bits(_IntegralType_ __value) noexcept {
    static_assert(std::is_unsigned_v<_IntegralType_>);

#if defined(simd_stl_processor_x86) && !defined(simd_stl_processor_arm)
    if (!type_traits::is_constant_evaluated()) {
        if (arch::ProcessorFeatures::AVX2())
            return __lzcnt_clz(__value);
        else
            return __bsr_clz(__value);
    }
    else
#endif // defined(simd_stl_processor_x86) && !defined(simd_stl_processor_arm)
    {
        return __bit_hacks_clz(__value);
    }
}

 
template <typename _IntegralType_>
constexpr simd_stl_always_inline int population_count(_IntegralType_ __value) noexcept {
    static_assert(std::is_unsigned_v<_IntegralType_>);

#if (defined(simd_stl_processor_x86_32) || defined(simd_stl_processor_x86_64) || defined(simd_stl_processor_arm_64))
    if (!type_traits::is_constant_evaluated()) {
        if (arch::ProcessorFeatures::POPCNT())
            return __popcnt_population_count(__value);
    }
    else
#endif // (defined(simd_stl_processor_x86_32) || defined(simd_stl_processor_x86_64) || defined(simd_stl_processor_arm_64))
    {
        return __bit_hacks_population_count(__value);
    }
}

__SIMD_STL_MATH_NAMESPACE_END