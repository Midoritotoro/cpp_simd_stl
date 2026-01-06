#pragma once 

#include <src/simd_stl/algorithm/unchecked/find/ContainsUnchecked.h>
#include <simd_stl/concurrency/Execution.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _Iterator_,
	class _Type_ = type_traits::iterator_value_type<_Iterator_>>
__simd_nodiscard_inline_constexpr bool contains(
	_Iterator_											__first,
	_Iterator_											__last,
	const typename std::type_identity<_Type_>::type&	__value) noexcept
{
	__verify_range(__first, __last);
	return __contains_unchecked(__unwrap_iterator(__first), __unwrap_iterator(__last), __value);
}

template <
	class _ExecutionPolicy_,
	class _Iterator_,
	class _Type_ = type_traits::iterator_value_type<_Iterator_>,
	concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
__simd_nodiscard_inline_constexpr bool contains(
	_ExecutionPolicy_&&,
	_Iterator_		__first,
	_Iterator_		__last,
	const _Type_&	__value) noexcept
{
	return simd_stl::algorithm::contains(__first, __last, __value);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
