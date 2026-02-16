#pragma once 

#include <src/simd_stl/datapar/compare/Equal.h>
#include <src/simd_stl/datapar/bitwise/BitNot.h>


__SIMD_STL_DATAPAR_NAMESPACE_BEGIN

template <
    arch::ISA	_ISA_,
    uint32		_Width_,
    class		_DesiredType_>
struct _Simd_not_equal {
    template <class _IntrinType_>
    simd_stl_nodiscard simd_stl_static_operator simd_stl_always_inline auto operator()(
        _IntrinType_ __left,
        _IntrinType_ __right) simd_stl_const_operator noexcept
    {
        const auto __compared = _Simd_equal<_ISA_, _Width_, _DesiredType_>()(__left, __right);
        using _ComparedType = decltype(__compared);

        if constexpr (std::is_integral_v<_ComparedType>)
            return ~__compared;
        else
            return _Simd_bit_not<_ISA_, _Width_>()(__compared);
    }
};

__SIMD_STL_DATAPAR_NAMESPACE_END
