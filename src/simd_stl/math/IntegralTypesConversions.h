#pragma once 

#include <numeric>
#include <simd_stl/SimdStlNamespace.h>


__SIMD_STL_MATH_NAMESPACE_BEGIN


template <class _Type_>
constexpr _Type_ MinimumIntegralLimit() noexcept {
	// same as (numeric_limits<_Ty>::min)(), less throughput cost
	static_assert(std::is_integral_v<_Type_>);

	if constexpr (std::is_unsigned_v< _Type_>)
		return 0;
	
	constexpr auto unsignedMax = static_cast<std::make_unsigned_t<_Type_>>(-1);
	return static_cast<_Type_>((unsignedMax >> 1) + 1); // N4950 [conv.integral]/3
}

template <class _Type_>
constexpr _Type_ MaximumIntegralLimit() noexcept { 
	// same as (numeric_limits<_Ty>::max)(), less throughput cost
	static_assert(std::is_integral_v<_Type_>);

	if constexpr (std::is_unsigned_v<_Type_>)
		return static_cast<_Type_>(-1);
	
	constexpr auto unsignedMax = static_cast<std::make_unsigned_t<_Type_>>(-1);
	return static_cast<_Type_>(unsignedMax >> 1);
}

// false if the _TypeFrom_ type cannot be converted to _TypeTo without from data loss
template <
	typename _TypeTo_,
	typename _TypeFrom_,
	typename = std::enable_if_t<
		std::is_integral_v<_TypeFrom_>
		&& std::is_integral_v<_TypeTo_>
		&& !std::is_same_v<_TypeFrom_, _TypeTo_>>>
constexpr inline bool ConvertIntegral(
	const _TypeFrom_	from,
	_TypeTo_&			to) noexcept
{
	if constexpr (std::is_same_v<_TypeFrom_, _TypeTo_>) {
		to = from;
		return true;
	}

	constexpr auto toMaximumLimit = MaximumIntegralLimit<_TypeTo_>();
	constexpr auto toMinimumLimit = MinimumIntegralLimit<_TypeTo_>();

	if constexpr (std::is_signed_v<_TypeFrom_> && std::is_signed_v<_TypeTo_>)
		if constexpr (sizeof(_TypeFrom_) > sizeof(_TypeTo_))
			if (from > toMaximumLimit || from < toMinimumLimit)
				return false;
	
	else if (std::is_unsigned_v<_TypeFrom_> && std::is_unsigned_v<_TypeTo_>)
		if constexpr (sizeof(_TypeFrom_) > sizeof(_TypeTo_))
			if (from > toMaximumLimit)
				return false;
	
	else if (std::is_signed_v<_TypeFrom_> && std::is_unsigned_v<_TypeTo_>)
		if constexpr (sizeof(_TypeFrom_) > sizeof(_TypeTo_))
			if (from < toMinimumLimit)
				return false;
	
	else /* std::is_unsigned_v<_TypeFrom_> && std::is_signed_v<_TypeTo_> */
		if constexpr (sizeof(_TypeFrom_) > sizeof(_TypeTo_))
			if (from > toMaximumLimit)
				return false;

	to = static_cast<_TypeTo_>(from);

	return true;
}

__SIMD_STL_MATH_NAMESPACE_END
