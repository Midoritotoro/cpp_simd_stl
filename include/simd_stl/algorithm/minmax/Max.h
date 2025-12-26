#pragma once 

#include <src/simd_stl/algorithm/unchecked/minmax/MaxUnchecked.h>
#include <simd_stl/concurrency/Execution.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
_Simd_nodiscard_inline _Type_ max(
	const _Type_& _Left,
	const _Type_& _Right) noexcept
{
	return (_Left > _Right) ? _Left : _Right;
}

template <
	class _Type_,
	class _Predicate_>
_Simd_nodiscard_inline _Type_ max(
	const _Type_&	_Left,
	const _Type_&	_Right,
	_Predicate_		_Predicate) noexcept
{
	return _Predicate(_Left, _Right) ? _Right : _Left;
}

template <
	class _InputIterator_,
	class _Type_>
_Simd_nodiscard_inline _Type_ max_range(
	_InputIterator_ _First,
	_InputIterator_ _Last) noexcept
{
	Assert(_First != _Last && "max_range requires non-empty range");
	return _MaxUnchecked(_UnwrapIterator(_First), _UnwrapIterator(_Last));
}

template <
	class _InputIterator_,
	class _Type_,
	class _Predicate_>
_Simd_nodiscard_inline _Type_ max_range(
	_InputIterator_ _First,
	_InputIterator_ _Last,
	_Predicate_		_Predicate) noexcept
{
	Assert(_First != _Last && "max_range requires non-empty range");
	return _MaxUnchecked(_UnwrapIterator(_First), _UnwrapIterator(_Last), type_traits::passFunction(_Predicate));
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
