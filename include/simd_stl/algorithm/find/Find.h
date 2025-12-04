#pragma once 

#include <src/simd_stl/algorithm/unchecked/FindUnchecked.h>
#include <src/simd_stl/algorithm/unchecked/FindIfUnchecked.h>
#include <src/simd_stl/algorithm/unchecked/FindIfNotUnchecked.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _Iterator_,
	class _Type_ = type_traits::IteratorValueType<_Iterator_>>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 _Iterator_ find(
	_Iterator_			first,
	const _Iterator_	last,
	const _Type_&		value) noexcept
{
	__verifyRange(first, last);

	_SeekPossiblyWrappedIterator(first, _FindUnchecked<decltype(_UnwrapIterator(first)), 
		type_traits::IteratorValueType<_Iterator_>>(_UnwrapIterator(first), _UnwrapIterator(last), value));
	return first;
}

template <
	class _InputIterator_,
	class _Predicate_>
simd_stl_nodiscard simd_stl_constexpr_cxx20 simd_stl_always_inline _InputIterator_ find_if_not(
	_InputIterator_			first, 
	const _InputIterator_	last, 
	_Predicate_				predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_,
			type_traits::IteratorValueType<_InputIterator_>
		>
	)
{
	__verifyRange(first, last);
	_SeekPossiblyWrappedIterator(first, _FindIfNotUnchecked(_UnwrapIterator(first),
		_UnwrapIterator(last), type_traits::passFunction(predicate)));

	return first;
}

template <
	class _InputIterator_,
	class _Predicate_>
simd_stl_nodiscard simd_stl_constexpr_cxx20 simd_stl_always_inline _InputIterator_ find_if(
	_InputIterator_			first, 
	const _InputIterator_	last, 
	_Predicate_				predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_,
			type_traits::IteratorValueType<_InputIterator_>
		>
	)
{
	__verifyRange(first, last);
	_SeekPossiblyWrappedIterator(first, _FindIfUnchecked(_UnwrapIterator(first), 
		_UnwrapIterator(last), type_traits::passFunction(predicate)));

	return first;
}

template <
	class _ExecutionPolicy_,
	class _Iterator_,
	class _Type_,
	concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
simd_stl_nodiscard _Iterator_ find(
	_ExecutionPolicy_&&,
	_Iterator_			first,
	const _Iterator_	last,
	const _Type_&		value) noexcept;

__SIMD_STL_ALGORITHM_NAMESPACE_END
