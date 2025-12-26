#pragma once 

#include <src/simd_stl/algorithm/unchecked/minmax/MinmaxUnchecked.h>
#include <simd_stl/concurrency/Execution.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
_Simd_nodiscard_inline std::pair<_Type_, _Type_> minmax(
	const _Type_& _Left,
	const _Type_& _Right) noexcept
{
	return (_Left < _Right) ? std::pair<_Type_, _Type_>{ _Left, _Right } : std::pair<_Type_, _Type_>{ _Right, _Left };
}

template <
	class _Type_,
	class _Predicate_>
_Simd_nodiscard_inline std::pair<_Type_, _Type_> minmax(
	const _Type_&	_Left,
	const _Type_&	_Right,
	_Predicate_		_Predicate) noexcept
{
	return _Predicate(_Right, _Left) ? std::pair<_Type_, _Type_>{ _Right, _Left } : std::pair<_Type_, _Type_>{ _Left, _Right };
}

template <class _InputIterator_>
_Simd_nodiscard_inline _Minmax_return_type<_InputIterator_> minmax_range(
	_InputIterator_ _First,
	_InputIterator_ _Last) noexcept
{
	Assert(_First != _Last && "minmax_range requires non-empty range");
	return _MinmaxUnchecked(_UnwrapIterator(_First), _UnwrapIterator(_Last));
}

template <
	class _InputIterator_,
	class _Predicate_>
_Simd_nodiscard_inline _Minmax_return_type<_InputIterator_> minmax_range(
	_InputIterator_ _First,
	_InputIterator_ _Last,
	_Predicate_		_Predicate) noexcept
{
	Assert(_First != _Last && "minmax_range requires non-empty range");
	return _MinmaxUnchecked(_UnwrapIterator(_First), _UnwrapIterator(_Last), type_traits::passFunction(_Predicate));
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
