#pragma once 

#include <src/simd_stl/numeric/IntrinBitcast.h>
#include <simd_stl/numeric/BasicSimdElementReference.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    class _ToVector_,
    class _FromVector_,
    std::enable_if_t<(_Is_valid_basic_simd_v<_ToVector_> || _Is_intrin_type_v<_ToVector_>) &&
        (_Is_valid_basic_simd_v<_FromVector_> || _Is_intrin_type_v<_FromVector_>), int> = 0>
simd_stl_nodiscard simd_stl_always_inline _ToVector_ simd_cast(_FromVector_ from) noexcept {
    if constexpr (_Is_valid_basic_simd_v<_FromVector_> && _Is_valid_basic_simd_v<_ToVector_>)
        return _IntrinBitcast<typename _ToVector_::vector_type>(from.unwrap());
    else if constexpr (_Is_intrin_type_v<_FromVector_> && _Is_valid_basic_simd_v<_ToVector_>)
        return _IntrinBitcast<typename _ToVector_::vector_type>(from);
    else if constexpr (_Is_valid_basic_simd_v<_FromVector_> && _Is_intrin_type_v<_ToVector_>)
        return _IntrinBitcast<_ToVector_>(from.unwrap());
    else if constexpr (_Is_intrin_type_v<_FromVector_> && _Is_intrin_type_v<_ToVector_>)
        return _IntrinBitcast<_ToVector_>(from);
}

__SIMD_STL_NUMERIC_NAMESPACE_END
