#pragma once 

#include <src/simd_stl/algorithm/unchecked/find/SearchUnchecked.h>
#include <simd_stl/concurrency/Execution.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _FirstForwardIterator_,
	class _SecondForwardIterator_,
	class _Predicate_> 
__simd_nodiscard_inline_constexpr _FirstForwardIterator_ search(
	_FirstForwardIterator_	_First1,
	_FirstForwardIterator_	_Last1,
	_SecondForwardIterator_ _First2,
	_SecondForwardIterator_ _Last2,
	_Predicate_				_Predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_,
			type_traits::iterator_value_type<_FirstForwardIterator_>,
			type_traits::iterator_value_type<_SecondForwardIterator_>>)
{
	__verifyRange(_First1, _Last1);
	__verifyRange(_First2, _Last2);
	
	__seek_possibly_wrapped_iterator(_First1, _SearchUnchecked(_UnwrapIterator(_First1),
		_UnwrapIterator(_Last1), _UnwrapIterator(_First2), _UnwrapIterator(_Last2), 
		type_traits::passFunction(_Predicate)));

	return _First1;
}

template <
	class _FirstForwardIterator_,
	class _SecondForwardIterator_>
__simd_nodiscard_inline_constexpr _FirstForwardIterator_ search(
	_FirstForwardIterator_	_First1,
	_FirstForwardIterator_	_Last1,
	_SecondForwardIterator_ _First2,
	_SecondForwardIterator_ _Last2) noexcept(
		type_traits::is_nothrow_invocable_v<
			type_traits::equal_to<>,
			type_traits::iterator_value_type<_FirstForwardIterator_>,
			type_traits::iterator_value_type<_SecondForwardIterator_>>)
{
	return simd_stl::algorithm::search(_First1, _Last1, _First2, _Last2, type_traits::equal_to<>{});
}

template <
	class _ForwardIterator_, 
	class _Searcher_>
__simd_nodiscard_inline_constexpr _ForwardIterator_ search(
	_ForwardIterator_ _First, 
	_ForwardIterator_ _Last,
	const _Searcher_& _Searcher) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Searcher_, type_traits::iterator_value_type<_ForwardIterator_>>)
{
	return _Searcher(_First, _Last).first;
}

template <
	class _ExecutionPolicy_,
	class _FirstForwardIterator_,
	class _SecondForwardIterator_,
	class _Predicate_,
	concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
__simd_nodiscard_inline_constexpr _FirstForwardIterator_ search(
	_ExecutionPolicy_&&,
	_FirstForwardIterator_	_First1,
	_FirstForwardIterator_	_Last1,
	_SecondForwardIterator_ _First2,
	_SecondForwardIterator_ _Last2,
	_Predicate_				_Predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_,
			type_traits::iterator_value_type<_FirstForwardIterator_>,
			type_traits::iterator_value_type<_SecondForwardIterator_>>)
{
	return simd_stl::algorithm::search(_First1, _Last1, _First2, _Last2, type_traits::passFunction(_Predicate));
}

template <
	class _ExecutionPolicy_,
	class _FirstForwardIterator_,
	class _SecondForwardIterator_,
	concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
__simd_nodiscard_inline_constexpr _FirstForwardIterator_ search(
	_ExecutionPolicy_&&,
	_FirstForwardIterator_	_First1,
	_FirstForwardIterator_	_Last1,
	_SecondForwardIterator_ _First2,
	_SecondForwardIterator_ _Last2) noexcept(
		type_traits::is_nothrow_invocable_v<
			type_traits::equal_to<>,
			type_traits::iterator_value_type<_FirstForwardIterator_>,
			type_traits::iterator_value_type<_SecondForwardIterator_>>)
{
	return simd_stl::algorithm::search(_First1, _Last1, _First2, _Last2);
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
