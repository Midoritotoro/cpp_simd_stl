#pragma once 

#include <simd_stl/Types.h>

#include <simd_stl/arch/ProcessorFeatures.h>
#include <simd_stl/arch/ProcessorDetection.h>

#include <simd_stl/compatibility/SimdCompatibility.h>
#include <src/simd_stl/type_traits/IsConstantEvaluated.h>

#include <src/simd_stl/math/IntegralTypesConversions.h>

#include <bit>

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


#if defined (simd_stl_processor_x86) && !defined(simd_stl_processor_arm)
#  if defined(simd_stl_cpp_clang) || defined(simd_stl_cpp_gnu)

#    if !defined(simd_stl_lzcnt_u16)
#      define simd_stl_lzcnt_u16 __builtin_ia32_lzcnt_u16
#    endif // !defined(simd_stl_lzcnt_u16)

#    if !defined(simd_stl_lzcnt_u32)
#      define simd_stl_lzcnt_u32 __builtin_ia32_lzcnt_u32
#    endif // !defined(simd_stl_lzcnt_u32)

#    if !defined(simd_stl_lzcnt_u64)
#      define simd_stl_lzcnt_u64 __builtin_ia32_lzcnt_u64 
#    endif // !defined(simd_stl_lzcnt_u64)

#  elif defined(simd_stl_cpp_msvc)

#    if !defined(simd_stl_lzcnt_u16)
#      define simd_stl_lzcnt_u16 __lzcnt16
#    endif // !defined(simd_stl_lzcnt_u16)

#    if !defined(simd_stl_lzcnt_u32)
#      define simd_stl_lzcnt_u32 __lzcnt
#    endif // !defined(simd_stl_lzcnt_u32)

#    if !defined(simd_stl_lzcnt_u64)
#      define simd_stl_lzcnt_u64 __lzcnt64
#    endif // !defined(simd_stl_lzcnt_u64)

#  endif // defined(simd_stl_cpp_clang) || defined(simd_stl_cpp_gnu) || defined(simd_stl_cpp_msvc)
#endif // defined (simd_stl_processor_x86) && !defined(simd_stl_processor_arm)

template <typename _IntegralType_>
simd_stl_always_inline int _BitHacksCountTrailingZeroBits(_IntegralType_ value) noexcept {
    if constexpr (sizeof(_IntegralType_) == 8) {

    }
    else if constexpr (sizeof(_IntegralType_) == 2) {
        unsigned int c = 16;

        value &= simd_stl::uint16(-signed(value));
        c -= value;

        if (value & 0x000000FF)
            c -= 8;

        if (value & 0x00000F0F)
            c -= 4;

        if (value & 0x00003333)
            c -= 2;

        if (value & 0x00005555)
            c -= 1;

        return c;
    }
}

#if defined (simd_stl_processor_x86) && !defined(simd_stl_processor_arm)
    template <typename _IntegralType_>
    simd_stl_always_inline int _BsfCountTrailingZeroBits(_IntegralType_ value) noexcept {
        ulong index = 0;
        _BitScanForward(&index, value);
        return index;
    }
#endif // defined (simd_stl_processor_x86) && !defined(simd_stl_processor_arm)

#if defined(simd_stl_processor_x86) && !defined(simd_stl_processor_arm)
template <typename _IntegralType_> 
simd_stl_always_inline int _TzcntCountTrailingZeroBits(_IntegralType_ value) noexcept {
    constexpr auto digits   = std::numeric_limits<_IntegralType_>::digits;
    constexpr auto maximum  = MaximumIntegralLimit<_IntegralType_>();

    if constexpr (digits == 64)
        return static_cast<int>(simd_stl_tzcnt_u64(value));
    else if constexpr (digits == 32)
        return static_cast<int>(simd_stl_tzcnt_u32((value));
    else if constexpr (digits < 32)
            // Use bithacks
    {
    }
}
#endif // defined(simd_stl_processor_x86) && !defined(simd_stl_processor_arm)

template <typename _Type_>
constexpr _Type_ ClearLeftMostSet(const _Type_ value) {
    return value & (value - 1);
}

template <typename _IntegralType_>
constexpr int CountTrailingZeroBits(_IntegralType_ value) noexcept {
}

__SIMD_STL_MATH_NAMESPACE_END