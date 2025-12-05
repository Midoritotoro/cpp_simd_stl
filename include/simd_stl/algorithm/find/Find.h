#pragma once 

#include <src/simd_stl/algorithm/unchecked/FindUnchecked.h>
#include <src/simd_stl/algorithm/unchecked/FindIfUnchecked.h>
#include <src/simd_stl/algorithm/unchecked/FindIfNotUnchecked.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _Iterator_,
	class _Type_ = type_traits::IteratorValueType<_Iterator_>>
_Simd_nodiscard_inline_constexpr _Iterator_ find(
	_Iterator_		_First,
	_Iterator_		_Last,
	const _Type_&	_Value) noexcept
{
	__verifyRange(_First, _Last);

	_SeekPossiblyWrappedIterator(_First, _FindUnchecked(_UnwrapIterator(_First), _UnwrapIterator(_Last), _Value));

	return _First;
}

template <
	class _InputIterator_,
	class _Predicate_>
_Simd_nodiscard_inline_constexpr _InputIterator_ find_if_not(
	_InputIterator_	_First, 
	_InputIterator_	_Last, 
	_Predicate_		_Predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_, type_traits::IteratorValueType<_InputIterator_>>)
{
	__verifyRange(_First, _Last);

	_SeekPossiblyWrappedIterator(_First, _FindIfNotUnchecked(_UnwrapIterator(_First),
		_UnwrapIterator(_Last), type_traits::passFunction(_Predicate)));

	return _First;
}

template <
	class _InputIterator_,
	class _Predicate_>
_Simd_nodiscard_inline_constexpr _InputIterator_ find_if(
	_InputIterator_	_First, 
	_InputIterator_	_Last, 
	_Predicate_		_Predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_, type_traits::IteratorValueType<_InputIterator_>>)
{
	__verifyRange(_First, _Last);

	_SeekPossiblyWrappedIterator(_First, _FindIfUnchecked(_UnwrapIterator(_First), 
		_UnwrapIterator(_Last), type_traits::passFunction(_Predicate)));

	return _First;
}

template <
	class _ExecutionPolicy_,
	class _Iterator_,
	class _Type_,
	concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
simd_stl_nodiscard _Iterator_ find(
	_ExecutionPolicy_&&,
	_Iterator_		_First,
	_Iterator_		_Last,
	const _Type_&	_Value) noexcept;

__SIMD_STL_ALGORITHM_NAMESPACE_END
