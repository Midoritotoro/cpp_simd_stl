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
constexpr int _BitHacksCountLeadingZeroBits(_IntegralType_ _Value) noexcept {
	if constexpr (sizeof(_IntegralType_) == 8) {
        _Value = _Value | (_Value >> 1);
        _Value = _Value | (_Value >> 2);
        _Value = _Value | (_Value >> 4);

        _Value = _Value | (_Value >> 8);
        _Value = _Value | (_Value >> 16);
        _Value = _Value | (_Value >> 32);

        return _BitHacksPopulationCount(static_cast<_IntegralType_>(~_Value));
	}
    else if constexpr (sizeof(_IntegralType_) == 4) {
        _Value = _Value | (_Value >> 1);
        _Value = _Value | (_Value >> 2);

        _Value = _Value | (_Value >> 4);
        _Value = _Value | (_Value >> 8);

        _Value = _Value | (_Value >> 16);
        
        return _BitHacksPopulationCount(static_cast<_IntegralType_>(~_Value));
    }
    else if constexpr (sizeof(_IntegralType_) == 2) {
        _Value = _Value | (_Value >> 1);
        _Value = _Value | (_Value >> 2);

        _Value = _Value | (_Value >> 4);
        _Value = _Value | (_Value >> 8);

        return _BitHacksPopulationCount(static_cast<_IntegralType_>(~_Value));
    }
    else if constexpr (sizeof(_IntegralType_) == 1) {
        _Value = _Value | (_Value >> 1);

        _Value = _Value | (_Value >> 2);
        _Value = _Value | (_Value >> 4);

        return _BitHacksPopulationCount(static_cast<_IntegralType_>(~_Value));
    }
}


#if (defined(simd_stl_processor_x86_32) || defined(simd_stl_processor_x86_64))

template <type_traits::standard_unsigned_integral _IntegralType_>
constexpr int _BsrCountLeadingZeroBits(_IntegralType_ _Value) noexcept {
    constexpr auto _Digits = std::numeric_limits<_IntegralType_>::digits;
    auto _Index = ulong(0);

    if constexpr (_Digits == 64)
        _BitScanReverse64(&_Index, _Value);
    else if constexpr (_Digits == 32)
        _BitScanReverse(&_Index, _Value);
    else
        _Index = _BitHacksCountLeadingZeroBits(_Value);

    return static_cast<int>(_Digits - 1 - _Index);
    
}

#endif // (defined(simd_stl_processor_x86_32) || defined(simd_stl_processor_x86_64))

#if (defined(simd_stl_processor_x86_32) || defined(simd_stl_processor_x86_64))

template <type_traits::standard_unsigned_integral _IntegralType_>
constexpr int _LzcntCountLeadingZeroBits(_IntegralType_ _Value) noexcept {
    constexpr auto _Digits = std::numeric_limits<_IntegralType_>::digits;

    if      constexpr (_Digits == 64)
        return static_cast<int>(simd_stl_lzcnt_u64(static_cast<uint64>(_Value)));
    else if constexpr (_Digits == 32)
        return static_cast<int>(simd_stl_lzcnt_u32(static_cast<uint32>(_Value)));
    else if constexpr (_Digits == 16)
        return static_cast<int>(simd_stl_lzcnt_u16(static_cast<uint16>(_Value)));
    else
        return _BitHacksCountLeadingZeroBits(_Value);
    
}

#endif // (defined(simd_stl_processor_x86_32) || defined(simd_stl_processor_x86_64))

__SIMD_STL_MATH_NAMESPACE_END
