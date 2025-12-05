#pragma once 

#include <src/simd_stl/algorithm/unchecked/ContainsUnchecked.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _Iterator_,
	class _Type_>
_Simd_nodiscard_inline_constexpr bool contains(
	_Iterator_		_First,
	_Iterator_		_Last,
	const _Type_&	_Value) noexcept
{
	__verifyRange(_First, _Last);
	return _ContainsUnchecked(_UnwrapIterator(_First), _UnwrapIterator(_Last), _Value);
}

template <
	class _ExecutionPolicy_,
	class _Iterator_,
	class _Type_>
_Simd_nodiscard_inline_constexpr bool contains(
	_ExecutionPolicy_&&,
	_Iterator_		_First,
	_Iterator_		_Last,
	const _Type_&	_Value) noexcept
{
	return simd_stl::algorithm::contains(_First, _Last, _Value);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
