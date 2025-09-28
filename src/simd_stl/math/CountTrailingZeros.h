#pragma once 

#include <simd_stl/Types.h>
#include <src/simd_stl/type_traits/IntegralProperties.h>

#include <simd_stl/arch/ProcessorFeatures.h>
#include <simd_stl/arch/ProcessorDetection.h>

#include <simd_stl/compatibility/SimdCompatibility.h>
#include <src/simd_stl/type_traits/IsConstantEvaluated.h>

#include <include/simd_stl/math/IntegralTypesConversions.h>


__SIMD_STL_MATH_NAMESPACE_BEGIN

#if defined (simd_stl_processor_x86) && !defined(simd_stl_processor_arm)
#  if defined(simd_stl_cpp_clang) || defined(simd_stl_cpp_gnu)

#    if !defined(simd_stl_tzcnt_u32)
#      define simd_stl_tzcnt_u32 __builtin_ia32_tzcnt_u32
#    endif // !defined(simd_stl_tzcnt_u32)

#    if !defined(simd_stl_tzcnt_u64)
#      define simd_stl_tzcnt_u64 __builtin_ia32_tzcnt_u64
#    endif // !defined(simd_stl_tzcnt_u64)

#  elif defined(simd_stl_cpp_msvc)

#    if !defined(simd_stl_tzcnt_u32)
#      define simd_stl_tzcnt_u32 _tzcnt_u32
#    endif // !defined(simd_stl_tzcnt_u32)

#    if !defined(simd_stl_tzcnt_u64)
#      define simd_stl_tzcnt_u64 _tzcnt_u64
#    endif // !defined(simd_stl_tzcnt_u64)

#  endif // defined(simd_stl_cpp_clang) || defined(simd_stl_cpp_gnu) || defined(simd_stl_cpp_msvc)
#endif // defined (simd_stl_processor_x86) && !defined(simd_stl_processor_arm)

constexpr inline int _BitHacksCountTrailingZeroBits32Bit(uint32 value) noexcept {
    auto result = uint32(32);
    value &= -signed(value);

    if (value)
        --result;

    if (value & 0x0000FFFF)
        result -= 16;

    if (value & 0x00FF00FF)
        result -= 8;

    if (value & 0x0F0F0F0F)
        result -= 4;

    if (value & 0x33333333) 
        result -= 2;

    if (value & 0x55555555) 
        result -= 1;

    return result;
}

template <type_traits::standard_unsigned_integral _IntegralType_>
constexpr inline int _BitHacksCountTrailingZeroBits(_IntegralType_ value) noexcept {
    if constexpr (sizeof(_IntegralType_) == 8) {
        const auto low = static_cast<simd_stl::uint32>(value);
        return low
            ? _BitHacksCountTrailingZeroBits32Bit(low)
            : 32 + _BitHacksCountTrailingZeroBits32Bit(static_cast<uint32>(value >> 32));
    }
    else if constexpr (sizeof(_IntegralType_) == 4) {
        return _BitHacksCountTrailingZeroBits32Bit(static_cast<uint32>(value));
    }
    else if constexpr (sizeof(_IntegralType_) == 2) {
        uint32 result = 16;

        value &= simd_stl::uint16(-signed(value));
        
        if (value)
            --result;

        if (value & 0x000000FF)
            result -= 8;

        if (value & 0x00000F0F)
            result -= 4;

        if (value & 0x00003333)
            result -= 2;

        if (value & 0x00005555)
            result -= 1;

        return result;
    }
    else if constexpr (sizeof(_IntegralType_) == 1) {
        auto result = uint32(8);
        value &= simd_stl::uint8(-signed(value));

        if (value)
            --result;

        if (value & 0x0000000F)
            result -= 4;

        if (value & 0x00000033)
            result -= 2;

        if (value & 0x00000055)
            result -= 1;

        return result;
    }
}

#if defined (simd_stl_processor_x86) && !defined(simd_stl_processor_arm)

template <type_traits::standard_unsigned_integral _IntegralType_>
simd_stl_constexpr_cxx20 simd_stl_always_inline int _BsfCountTrailingZeroBits(_IntegralType_ value) noexcept {
    constexpr auto digits = std::numeric_limits<_IntegralType_>::digits;
    auto index = ulong(0);

    if      constexpr (digits == 64)
        _BitScanForward64(&index, value);
    else if constexpr (digits == 32)
        _BitScanForward(&index, value);
    else if constexpr (digits < 32)
        index = _BitHacksCountTrailingZeroBits(value);

    return index;
}

template <type_traits::standard_unsigned_integral _IntegralType_>
simd_stl_constexpr_cxx20 simd_stl_always_inline int _TzcntCountTrailingZeroBits(_IntegralType_ value) noexcept {
    constexpr auto digits   = std::numeric_limits<_IntegralType_>::digits;

    if      constexpr (digits == 64)
        return static_cast<int>(simd_stl_tzcnt_u64(value));
    else if constexpr (digits == 32)
        return static_cast<int>(simd_stl_tzcnt_u32(value));
    else if constexpr (digits < 32)
        return _BitHacksCountTrailingZeroBits(value);
}

#endif // defined(simd_stl_processor_x86) && !defined(simd_stl_processor_arm)

__SIMD_STL_MATH_NAMESPACE_END
