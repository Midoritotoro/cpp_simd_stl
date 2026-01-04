#pragma once 

#include <simd_stl/Types.h>
#include <src/simd_stl/type_traits/IntegralProperties.h>

#include <simd_stl/arch/ProcessorFeatures.h>
#include <simd_stl/arch/ProcessorDetection.h>

#include <simd_stl/compatibility/SimdCompatibility.h>
#include <src/simd_stl/type_traits/IsConstantEvaluated.h>

#include <include/simd_stl/math/IntegralTypesConversions.h>


__SIMD_STL_MATH_NAMESPACE_BEGIN

#if defined (simd_stl_processor_x86)
#  if defined(simd_stl_cpp_clang) || defined(simd_stl_cpp_gnu)

#    if !defined(__simd_stl_tzcnt_u32)
#      define __simd_stl_tzcnt_u32 __builtin_ia32_tzcnt_u32
#    endif // !defined(__simd_stl_tzcnt_u32)

#    if !defined(__simd_stl_tzcnt_u64)
#      define __simd_stl_tzcnt_u64 __builtin_ia32_tzcnt_u64
#    endif // !defined(__simd_stl_tzcnt_u64)

#  elif defined(simd_stl_cpp_msvc)

#    if !defined(__simd_stl_tzcnt_u32)
#      define __simd_stl_tzcnt_u32 _tzcnt_u32
#    endif // !defined(__simd_stl_tzcnt_u32)

#    if !defined(__simd_stl_tzcnt_u64)
#      define __simd_stl_tzcnt_u64 _tzcnt_u64
#    endif // !defined(__simd_stl_tzcnt_u64)

#  endif // defined(simd_stl_cpp_clang) || defined(simd_stl_cpp_gnu) || defined(simd_stl_cpp_msvc)
#endif // defined (simd_stl_processor_x86)

constexpr simd_stl_always_inline int __bit_hacks_ctz_u32(uint32 __value) noexcept {
    auto __result   = uint32(32);
    __value         &= -signed(__value);

    if (__value)
        --__result;

    if (__value & 0x0000FFFF)
        __result -= 16;

    if (__value & 0x00FF00FF)
        __result -= 8;

    if (__value & 0x0F0F0F0F)
        __result -= 4;

    if (__value & 0x33333333)
        __result -= 2;

    if (__value & 0x55555555)
        __result -= 1;

    return __result;
}

template <typename _IntegralType_>
constexpr simd_stl_always_inline int __bit_hacks_ctz(_IntegralType_ __value) noexcept {
    if constexpr (sizeof(_IntegralType_) == 8) {
        const auto __low = static_cast<uint32>(__value);
        return __low
            ? __bit_hacks_ctz_u32(__low)
            : 32 + __bit_hacks_ctz_u32(static_cast<uint32>(__value >> 32));
    }
    else if constexpr (sizeof(_IntegralType_) == 4) {
        return __bit_hacks_ctz_u32(static_cast<uint32>(__value));
    }
    else if constexpr (sizeof(_IntegralType_) == 2) {
        auto __result   = uint32(16);
        __value         &= uint16(-signed(__value));
        
        if (__value)
            --__result;

        if (__value & 0x000000FF)
            __result -= 8;

        if (__value & 0x00000F0F)
            __result -= 4;

        if (__value & 0x00003333)
            __result -= 2;

        if (__value & 0x00005555)
            __result -= 1;

        return __result;
    }
    else if constexpr (sizeof(_IntegralType_) == 1) {
        auto __result   = uint32(8);
        __value         &= uint8(-signed(__value));

        if (__value)
            --__result;

        if (__value & 0x0000000F)
            __result -= 4;

        if (__value & 0x00000033)
            __result -= 2;

        if (__value & 0x00000055)
            __result -= 1;

        return __result;
    }
}

#if defined (simd_stl_processor_x86)

template <typename _IntegralType_>
simd_stl_always_inline int __bsf_ctz(_IntegralType_ __value) noexcept {
    constexpr auto __digits = std::numeric_limits<_IntegralType_>::digits;
    auto __index = ulong(0);

    if      constexpr (__digits == 64)
        _BitScanForward64(&__index, __value);
    else if constexpr (__digits == 32)
        _BitScanForward(&__index, __value);
    else if constexpr (__digits < 32)
        __index = __bit_hacks_ctz(__value);

    return __index;
}

template <typename _IntegralType_>
simd_stl_always_inline int __tzcnt_ctz(_IntegralType_ __value) noexcept {
    constexpr auto __digits   = std::numeric_limits<_IntegralType_>::digits;

    if      constexpr (__digits == 64)
        return static_cast<int>(__simd_stl_tzcnt_u64(__value));
    else if constexpr (__digits == 32)
        return static_cast<int>(__simd_stl_tzcnt_u32(__value));
    else if constexpr (__digits < 32)
        return __bit_hacks_ctz(__value);
}

#endif // defined(simd_stl_processor_x86)

__SIMD_STL_MATH_NAMESPACE_END
