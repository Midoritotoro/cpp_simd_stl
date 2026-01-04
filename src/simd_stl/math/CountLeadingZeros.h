#pragma once 

#include <src/simd_stl/math/PopulationCount.h>

__SIMD_STL_MATH_NAMESPACE_BEGIN

#if defined (simd_stl_processor_x86)
#  if defined(simd_stl_cpp_clang) || defined(simd_stl_cpp_gnu)

#    if !defined(__simd_stl_lzcnt_u16)
#      define __simd_stl_lzcnt_u16 __builtin_ia32_lzcnt_u16
#    endif // !defined(__simd_stl_lzcnt_u16)

#    if !defined(__simd_stl_lzcnt_u32)
#      define __simd_stl_lzcnt_u32 __builtin_ia32_lzcnt_u32
#    endif // !defined(__simd_stl_lzcnt_u32)

#    if !defined(__simd_stl_lzcnt_u64)
#      define __simd_stl_lzcnt_u64 __builtin_ia32_lzcnt_u64 
#    endif // !defined(__simd_stl_lzcnt_u64)

#  elif defined(simd_stl_cpp_msvc)

#    if !defined(__simd_stl_lzcnt_u16)
#      define __simd_stl_lzcnt_u16 __lzcnt16
#    endif // !defined(__simd_stl_lzcnt_u16)

#    if !defined(__simd_stl_lzcnt_u32)
#      define __simd_stl_lzcnt_u32 __lzcnt
#    endif // !defined(__simd_stl_lzcnt_u32)

#    if !defined(__simd_stl_lzcnt_u64)
#      define __simd_stl_lzcnt_u64 __lzcnt64
#    endif // !defined(__simd_stl_lzcnt_u64)

#  endif // defined(simd_stl_cpp_clang) || defined(simd_stl_cpp_gnu) || defined(simd_stl_cpp_msvc)
#endif // defined (simd_stl_processor_x86)

template <typename _IntegralType_> 
constexpr simd_stl_always_inline int __bit_hacks_clz(_IntegralType_ __value) noexcept {
	if constexpr (sizeof(_IntegralType_) == 8) {
        __value = __value | (__value >> 1);
        __value = __value | (__value >> 2);
        __value = __value | (__value >> 4);

        __value = __value | (__value >> 8);
        __value = __value | (__value >> 16);
        __value = __value | (__value >> 32);

        return __bit_hacks_population_count(static_cast<_IntegralType_>(~__value));
	}
    else if constexpr (sizeof(_IntegralType_) == 4) {
        __value = __value | (__value >> 1);
        __value = __value | (__value >> 2);

        __value = __value | (__value >> 4);
        __value = __value | (__value >> 8);

        __value = __value | (__value >> 16);
        
        return __bit_hacks_population_count(static_cast<_IntegralType_>(~__value));
    }
    else if constexpr (sizeof(_IntegralType_) == 2) {
        __value = __value | (__value >> 1);
        __value = __value | (__value >> 2);

        __value = __value | (__value >> 4);
        __value = __value | (__value >> 8);

        return __bit_hacks_population_count(static_cast<_IntegralType_>(~__value));
    }
    else if constexpr (sizeof(_IntegralType_) == 1) {
        __value = __value | (__value >> 1);

        __value = __value | (__value >> 2);
        __value = __value | (__value >> 4);

        return __bit_hacks_population_count(static_cast<_IntegralType_>(~__value));
    }
}


#if (defined(simd_stl_processor_x86_32) || defined(simd_stl_processor_x86_64))

template <typename _IntegralType_>
simd_stl_always_inline int __bsr_clz(_IntegralType_ __value) noexcept {
    constexpr auto __digits = std::numeric_limits<_IntegralType_>::digits;
    auto __index = ulong(0);

    if constexpr (__digits == 64)
        _BitScanReverse64(&__index, __value);
    else if constexpr (__digits == 32)
        _BitScanReverse(&__index, __value);
    else
        __index = _BitHacksCountLeadingZeroBits(__value);

    return static_cast<int>(__digits - 1 - __index);
}

#endif // (defined(simd_stl_processor_x86_32) || defined(simd_stl_processor_x86_64))

#if (defined(simd_stl_processor_x86_32) || defined(simd_stl_processor_x86_64))

template <typename _IntegralType_>
simd_stl_always_inline int __lzcnt_clz(_IntegralType_ __value) noexcept {
    constexpr auto __digits = std::numeric_limits<_IntegralType_>::digits;

    if      constexpr (__digits == 64)
        return static_cast<int>(__simd_stl_lzcnt_u64(static_cast<uint64>(__value)));
    else if constexpr (__digits == 32)
        return static_cast<int>(__simd_stl_lzcnt_u32(static_cast<uint32>(__value)));
    else if constexpr (__digits == 16)
        return static_cast<int>(__simd_stl_lzcnt_u16(static_cast<uint16>(__value)));
    else
        return __bit_hacks_clz(__value);
}

#endif // (defined(simd_stl_processor_x86_32) || defined(simd_stl_processor_x86_64))

__SIMD_STL_MATH_NAMESPACE_END
