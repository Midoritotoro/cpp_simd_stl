#pragma once 

#include <numeric>
#include <src/simd_stl/type_traits/TypeTraits.h>

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
	typename = ::std::enable_if_t<
		::std::is_integral_v<_TypeFrom_>
		&& ::std::is_integral_v<_TypeTo_>
		&& !::std::is_same_v<_TypeFrom_, _TypeTo_>>>
constexpr inline bool ConvertIntegral(
	const _TypeFrom_	_From,
	_TypeTo_&			_To) noexcept
{
	if constexpr (std::is_same_v<_TypeFrom_, _TypeTo_>) {
		_To = _From;
		return true;
	}

	constexpr auto toMaximumLimit = MaximumIntegralLimit<_TypeTo_>();
	constexpr auto toMinimumLimit = MinimumIntegralLimit<_TypeTo_>();

	if constexpr (std::is_signed_v<_TypeFrom_> && std::is_signed_v<_TypeTo_>)
		if constexpr (sizeof(_TypeFrom_) > sizeof(_TypeTo_))
			if (_From > toMaximumLimit || _From < toMinimumLimit)
				return false;
	
	else if (std::is_unsigned_v<_TypeFrom_> && std::is_unsigned_v<_TypeTo_>)
		if constexpr (sizeof(_TypeFrom_) > sizeof(_TypeTo_))
			if (_From > toMaximumLimit)
				return false;
	
	else if (std::is_signed_v<_TypeFrom_> && std::is_unsigned_v<_TypeTo_>)
		if constexpr (sizeof(_TypeFrom_) > sizeof(_TypeTo_))
			if (_From < toMinimumLimit)
				return false;
	
	else /* std::is_unsigned_v<_TypeFrom_> && std::is_signed_v<_TypeTo_> */
		if constexpr (sizeof(_TypeFrom_) > sizeof(_TypeTo_))
			if (_From > toMaximumLimit)
				return false;

	_To = static_cast<_TypeTo_>(_From);

	return true;
}

// Имеет ли смысл сравнение _Type_ value с _InputIterator_::value_type
template <
    class _InputIterator_,
    class _Type_>
simd_stl_nodiscard simd_stl_always_inline constexpr bool couldCompareEqualToValueType(const _Type_& _Value) noexcept {
    if constexpr (std::disjunction_v<
#ifdef __cpp_lib_byte
        std::is_same<_Type_, std::byte>,
#endif // defined(__cpp_lib_byte)
        std::is_same<_Type_, bool>, std::is_pointer<_Type_>, std::is_same<_Type_, std::nullptr_t>>) 
    {
        return true;
    } 
    else {
        using _ElementType_ = type_traits::IteratorValueType<_InputIterator_>;
        static_assert(std::is_integral_v<_ElementType_> && std::is_integral_v<_Type_>);

        if constexpr (std::is_same_v<_ElementType_, bool>) {
            return _Value == true || _Value == false;
        } 
		else if constexpr (std::is_signed_v<_ElementType_>) {
            constexpr auto _Minimum = MinimumIntegralLimit<_ElementType_>();
            constexpr auto _Maximum = MaximumIntegralLimit<_ElementType_>();

            if constexpr (std::is_signed_v<_Type_>) {
                return _Minimum <= _Value && _Value <= _Maximum;
            } 
			else {
                if constexpr (_ElementType_{-1} == static_cast<_Type_>(-1)) 
                    return _Value <= _Maximum || static_cast<_Type_>(_Minimum) <= _Value;
                else
                    return _Value <= _Maximum;
            }
        } else {
            constexpr auto _Maximum = MaximumIntegralLimit<_ElementType_>();

            if constexpr (std::is_unsigned_v<_Type_>) {
                return _Value <= _Maximum;
            } 
			else {
                if constexpr (_Type_{-1} == static_cast<_ElementType_>(-1))
                    return _Value <= _Maximum;
                else
                    return 0 <= _Value && _Value <= _Maximum;
            }
        }
    }
}

__SIMD_STL_MATH_NAMESPACE_END
