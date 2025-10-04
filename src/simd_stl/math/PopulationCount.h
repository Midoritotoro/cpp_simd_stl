#pragma once

#include <simd_stl/Types.h>
#include <src/simd_stl/type_traits/IntegralProperties.h>

#include <simd_stl/arch/ProcessorFeatures.h>
#include <simd_stl/arch/ProcessorDetection.h>

#include <simd_stl/compatibility/SimdCompatibility.h>
#include <src/simd_stl/type_traits/IsConstantEvaluated.h>

#include <include/simd_stl/math/IntegralTypesConversions.h>


__SIMD_STL_MATH_NAMESPACE_BEGIN

#if (defined(simd_stl_processor_x86_32) || defined(simd_stl_processor_x86_64) || defined(simd_stl_processor_arm_64))

#  if defined(simd_stl_cpp_gnu) || defined(simd_stl_cpp_clang)

#    if !defined(simd_stl_popcnt_u32)
#      define simd_stl_popcnt_u32 __builtin_popcount
#    endif // !defined(simd_stl_popcnt_u32)

#    if !defined(simd_stl_popcnt_u64)
#      define simd_stl_popcnt_u64 __builtin_popcountll
#    endif // !defined(simd_stl_popcnt_u64)

#  elif defined(simd_stl_cpp_msvc) 

#    if !defined(simd_stl_popcnt_u32)
#      define simd_stl_popcnt_u32 __popcnt
#    endif // !defined(simd_stl_popcnt_u32)

#    if !defined(simd_stl_popcnt_u64)
#      define simd_stl_popcnt_u64 __popcnt64
#    endif // !defined(simd_stl_popcnt_u64)

#  endif // defined(simd_stl_cpp_gnu) || defined(simd_stl_cpp_clang)

#endif // (defined(simd_stl_processor_x86_32) || defined(simd_stl_processor_x86_64) || defined(simd_stl_processor_arm_64))


template <type_traits::standard_unsigned_integral _IntegralType_>
constexpr int _BitHacksPopulationCount(_IntegralType_ value) noexcept {
    if      constexpr (sizeof(_IntegralType_) == 8) {
        return (((value) & 0xfff) * static_cast<uint64>(0x1001001001001)
            & static_cast<uint64>(0x84210842108421)) % 0x1f +
        (((value >> 12) & 0xfff) * static_cast<uint64>(0x1001001001001)
            & static_cast<uint64>(0x84210842108421)) % 0x1f +
        (((value >> 24) & 0xfff) * static_cast<uint64>(0x1001001001001)
            & static_cast<uint64>(0x84210842108421)) % 0x1f +
        (((value >> 36) & 0xfff) * static_cast<uint64>(0x1001001001001)
            & static_cast<uint64>(0x84210842108421)) % 0x1f +
        (((value >> 48) & 0xfff) * static_cast<uint64>(0x1001001001001)
            & static_cast<uint64>(0x84210842108421)) % 0x1f +
        (((value >> 60) & 0xfff) * static_cast<uint64>(0x1001001001001)
            & static_cast<uint64>(0x84210842108421)) % 0x1f;
    }
    else if constexpr (sizeof(_IntegralType_) == 4) {
        return (((value) & 0xfff) * static_cast<uint64>(0x1001001001001)
            & static_cast<uint64>(0x84210842108421)) % 0x1f +
        (((value >> 12) & 0xfff) * static_cast<uint64>(0x1001001001001)
            & static_cast<uint64>(0x84210842108421)) % 0x1f +
        (((value >> 24) & 0xfff) * static_cast<uint64>(0x1001001001001)
            & static_cast<uint64>(0x84210842108421)) % 0x1f;
    }
    else if constexpr (sizeof(_IntegralType_) == 2) {
        return (((value) & 0xfff) * static_cast<uint64>(0x1001001001001)
            & static_cast<uint64>(0x84210842108421)) % 0x1f +
            (((value >> 12) & 0xfff) * static_cast<uint64>(0x1001001001001)
                & static_cast<uint64>(0x84210842108421)) % 0x1f;
    }
    else if constexpr (sizeof(_IntegralType_) == 1) {
        return (((value) & 0xfff) * static_cast<uint64>(0x1001001001001)
            & static_cast<uint64>(0x84210842108421)) % 0x1f;
    }
}

#if (defined(simd_stl_processor_x86_32) || defined(simd_stl_processor_x86_64) || defined(simd_stl_processor_arm_64))

template <type_traits::standard_unsigned_integral _IntegralType_>
simd_stl_constexpr_cxx20 simd_stl_always_inline int _PopcntPopulationCount(_IntegralType_ value) noexcept {
    constexpr auto digits = std::numeric_limits<_IntegralType_>::digits;

    if      constexpr (digits == 64)
        return static_cast<int>(simd_stl_popcnt_u64(static_cast<uint64>(value)));
    else if constexpr (digits == 32)
        return static_cast<int>(simd_stl_popcnt_u32(static_cast<uint32>(value)));
    else if constexpr (digits == 16)
#if defined(simd_stl_cpp_msvc)
        return static_cast<int>(__popcnt16(static_cast<uint16>(value)));
#elif defined(simd_stl_cpp_gnu) || defined(simd_stl_cpp_clang)
        return _BitHacksPopulationCount(value);
#endif // defined(simd_stl_cpp_msvc) // defined(simd_stl_cpp_gnu) || defined(simd_stl_cpp_clang)
    
    return _BitHacksPopulationCount(value);
}

#endif // (defined(simd_stl_processor_x86_32) || defined(simd_stl_processor_x86_64) || defined(simd_stl_processor_arm_64))


__SIMD_STL_MATH_NAMESPACE_END
