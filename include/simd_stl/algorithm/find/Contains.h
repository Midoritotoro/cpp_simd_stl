#pragma once 

#include <src/simd_stl/algorithm/unchecked/find/ContainsUnchecked.h>
#include <simd_stl/concurrency/Execution.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _Iterator_,
	class _Type_>
__simd_nodiscard_inline_constexpr bool contains(
	_Iterator_		__first,
	_Iterator_		_Last,
	const _Type_&	_Value) noexcept
{
	__verifyRange(__first, _Last);
	return _ContainsUnchecked(_UnwrapIterator(__first), _UnwrapIterator(_Last), _Value);
}

template <
	class _ExecutionPolicy_,
	class _Iterator_,
	class _Type_,
	concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
__simd_nodiscard_inline_constexpr bool contains(
	_ExecutionPolicy_&&,
	_Iterator_		__first,
	_Iterator_		_Last,
	const _Type_&	_Value) noexcept
{
	return simd_stl::algorithm::contains(__first, _Last, _Value);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
