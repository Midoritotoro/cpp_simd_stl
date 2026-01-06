#pragma once 

#include <src/simd_stl/algorithm/unchecked/minmax/MinmaxUnchecked.h>
#include <simd_stl/concurrency/Execution.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
__simd_nodiscard_inline std::pair<_Type_, _Type_> minmax(
	const _Type_& __left,
	const _Type_& __right) noexcept
{
	return (__left < __right) 
		? std::pair<_Type_, _Type_>{ __left, __right } 
		: std::pair<_Type_, _Type_>{ __right, __left };
}

template <
	class _Type_,
	class _Predicate_>
__simd_nodiscard_inline std::pair<_Type_, _Type_> minmax(
	const _Type_&	__left,
	const _Type_&	__right,
	_Predicate_		__predicate) noexcept
{
	return __predicate(__right, __left) 
		? std::pair<_Type_, _Type_>{ __right, __left } 
		: std::pair<_Type_, _Type_>{ __left, __right };
}

template <class _InputIterator_>
__simd_nodiscard_inline __minmax_return_type<_InputIterator_> minmax_range(
	_InputIterator_ __first,
	_InputIterator_ __last) noexcept
{
	simd_stl_assert(__first != __last && "minmax_range requires non-empty range");
	return __minmax_unchecked(__unwrap_iterator(__first), __unwrap_iterator(__last));
}

template <
	class _InputIterator_,
	class _Predicate_>
__simd_nodiscard_inline __minmax_return_type<_InputIterator_> minmax_range(
	_InputIterator_ __first,
	_InputIterator_ __last,
	_Predicate_		__predicate) noexcept
{
	simd_stl_assert(__first != __last && "minmax_range requires non-empty range");
	return __minmax_unchecked(__unwrap_iterator(__first), __unwrap_iterator(__last), type_traits::__pass_function(__predicate));
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
