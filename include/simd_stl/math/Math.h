#pragma once 

#include <simd_stl/Types.h>
#include <simd_stl/compatibility/Compatibility.h>


__SIMD_STL_MATH_NAMESPACE_BEGIN

template <typename _Type_>
__simd_nodiscard_inline_constexpr _Type_ abs(_Type_ _Value) noexcept {
	static_assert(std::is_integral_v<_Type_> || std::is_floating_point_v<_Type_>);

	if constexpr (std::is_unsigned_v<_Type_>)
		return _Value;
	else
		return (_Value < 0) ? -_Value : _Value;
}

__SIMD_STL_MATH_NAMESPACE_END
