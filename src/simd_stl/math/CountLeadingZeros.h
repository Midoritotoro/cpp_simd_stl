#pragma once 

#include <src/simd_stl/math/PopulationCount.h>

__SIMD_STL_MATH_NAMESPACE_BEGIN

#if defined (simd_stl_processor_x86)
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
#endif // defined (simd_stl_processor_x86)

template <type_traits::standard_unsigned_integral _IntegralType_> 
constexpr int _BitHacksCountLeadingZeroBits(_IntegralType_ value) noexcept {
	if constexpr (sizeof(_IntegralType_) == 8) {
        value = value | (value >> 1);
        value = value | (value >> 2);
        value = value | (value >> 4);

        value = value | (value >> 8);
        value = value | (value >> 16);
        value = value | (value >> 32);

        return _BitHacksPopulationCount(static_cast<_IntegralType_>(~value));
	}
    else if constexpr (sizeof(_IntegralType_) == 4) {
        value = value | (value >> 1);
        value = value | (value >> 2);

        value = value | (value >> 4);
        value = value | (value >> 8);

        value = value | (value >> 16);
        
        return _BitHacksPopulationCount(static_cast<_IntegralType_>(~value));
    }
    else if constexpr (sizeof(_IntegralType_) == 2) {
        value = value | (value >> 1);
        value = value | (value >> 2);

        value = value | (value >> 4);
        value = value | (value >> 8);

        return _BitHacksPopulationCount(static_cast<_IntegralType_>(~value));
    }
    else if constexpr (sizeof(_IntegralType_) == 1) {
        // 0b00001101
        
        // 0b00001101 | 0b00000110 = 0b00001111
        // 0b00001111 | 0b00000011 = 0b00001111
        // 0b00001111 | 0b00000000 = 0b00001111
        // ~0b00001111 = 0b11110000

        // popcnt(0b11110000) == 4;
        
        // 0b00111100

        // 0b00111100 | 0b00011110 = 0b00111110
        // 0b00111110 | 0b00001111 = 0b00111111
        // 0b00111111 | 0b00000011 = 0b00111111
        // ~0b00111111 = 0b11000000

        // popcnt(0b11000000) == 2;

        value = value | (value >> 1);

        value = value | (value >> 2);
        value = value | (value >> 4);

        return _BitHacksPopulationCount(static_cast<_IntegralType_>(~value));
    }
}


#if (defined(simd_stl_processor_x86_32) || defined(simd_stl_processor_x86_64))

template <type_traits::standard_unsigned_integral _IntegralType_>
constexpr int _BsrCountLeadingZeroBits(_IntegralType_ value) noexcept {
    constexpr auto digits = std::numeric_limits<_IntegralType_>::digits;
    auto index = ulong(0);

    if constexpr (digits == 64)
        _BitScanReverse64(&index, value);
    else if constexpr (digits == 32) 
        _BitScanReverse(&index, value);
    else
        index = _BitHacksCountLeadingZeroBits(value);

    return static_cast<int>(digits - 1 - index);
    
}

#endif // (defined(simd_stl_processor_x86_32) || defined(simd_stl_processor_x86_64))

#if (defined(simd_stl_processor_x86_32) || defined(simd_stl_processor_x86_64))

template <type_traits::standard_unsigned_integral _IntegralType_>
constexpr int _LzcntCountLeadingZeroBits(_IntegralType_ value) noexcept {
    constexpr auto digits = std::numeric_limits<_IntegralType_>::digits;

    if      constexpr (digits == 64)
        return static_cast<int>(simd_stl_lzcnt_u64(static_cast<uint64>(value)));
    else if constexpr (digits == 32)
        return static_cast<int>(simd_stl_lzcnt_u32(static_cast<uint32>(value)));
    else if constexpr (digits == 16)
        return static_cast<int>(simd_stl_lzcnt_u16(static_cast<uint16>(value)));
    else
        return _BitHacksCountLeadingZeroBits(value);
    
}

#endif // (defined(simd_stl_processor_x86_32) || defined(simd_stl_processor_x86_64))

__SIMD_STL_MATH_NAMESPACE_END
