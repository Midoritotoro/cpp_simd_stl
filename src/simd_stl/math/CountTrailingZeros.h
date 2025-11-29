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
#endif // defined (simd_stl_processor_x86)

constexpr simd_stl_always_inline int _BitHacksCountTrailingZeroBits32Bit(uint32 _Value) noexcept {
    auto _Result = uint32(32);
    _Value &= -signed(_Value);

    if (_Value)
        --_Result;

    if (_Value & 0x0000FFFF)
        _Result -= 16;

    if (_Value & 0x00FF00FF)
        _Result -= 8;

    if (_Value & 0x0F0F0F0F)
        _Result -= 4;

    if (_Value & 0x33333333) 
        _Result -= 2;

    if (_Value & 0x55555555) 
        _Result -= 1;

    return _Result;
}

template <type_traits::standard_unsigned_integral _IntegralType_>
constexpr simd_stl_always_inline int _BitHacksCountTrailingZeroBits(_IntegralType_ _Value) noexcept {
    if constexpr (sizeof(_IntegralType_) == 8) {
        const auto _Low = static_cast<simd_stl::uint32>(_Value);
        return _Low
            ? _BitHacksCountTrailingZeroBits32Bit(_Low)
            : 32 + _BitHacksCountTrailingZeroBits32Bit(static_cast<uint32>(_Value >> 32));
    }
    else if constexpr (sizeof(_IntegralType_) == 4) {
        return _BitHacksCountTrailingZeroBits32Bit(static_cast<uint32>(_Value));
    }
    else if constexpr (sizeof(_IntegralType_) == 2) {
        uint32 _Result = 16;

        _Value &= simd_stl::uint16(-signed(_Value));
        
        if (_Value)
            --_Result;

        if (_Value & 0x000000FF)
            _Result -= 8;

        if (_Value & 0x00000F0F)
            _Result -= 4;

        if (_Value & 0x00003333)
            _Result -= 2;

        if (_Value & 0x00005555)
            _Result -= 1;

        return _Result;
    }
    else if constexpr (sizeof(_IntegralType_) == 1) {
        auto _Result = uint32(8);
        _Value &= simd_stl::uint8(-signed(_Value));

        if (_Value)
            --_Result;

        if (_Value & 0x0000000F)
            _Result -= 4;

        if (_Value & 0x00000033)
            _Result -= 2;

        if (_Value & 0x00000055)
            _Result -= 1;

        return _Result;
    }
}

#if defined (simd_stl_processor_x86)

template <type_traits::standard_unsigned_integral _IntegralType_>
simd_stl_always_inline int _BsfCountTrailingZeroBits(_IntegralType_ _Value) noexcept {
    constexpr auto _Digits = std::numeric_limits<_IntegralType_>::digits;
    auto _Index = ulong(0);

    if      constexpr (_Digits == 64)
        _BitScanForward64(&_Index, _Value);
    else if constexpr (_Digits == 32)
        _BitScanForward(&_Index, _Value);
    else if constexpr (_Digits < 32)
        _Index = _BitHacksCountTrailingZeroBits(_Value);

    return _Index;
}

template <type_traits::standard_unsigned_integral _IntegralType_>
simd_stl_always_inline int _TzcntCountTrailingZeroBits(_IntegralType_ _Value) noexcept {
    constexpr auto _Digits   = std::numeric_limits<_IntegralType_>::digits;

    if      constexpr (_Digits == 64)
        return static_cast<int>(simd_stl_tzcnt_u64(_Value));
    else if constexpr (_Digits == 32)
        return static_cast<int>(simd_stl_tzcnt_u32(_Value));
    else if constexpr (_Digits < 32)
        return _BitHacksCountTrailingZeroBits(_Value);
}

#endif // defined(simd_stl_processor_x86)

__SIMD_STL_MATH_NAMESPACE_END
