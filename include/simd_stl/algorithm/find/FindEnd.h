#pragma once 

#include <src/simd_stl/algorithm/unchecked/find/FindEndUnchecked.h>
#include <simd_stl/concurrency/Execution.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _FirstForwardIterator_,
	class _SecondForwardIterator_,
	class _Predicate_> 
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 _FirstForwardIterator_ find_end(
	_FirstForwardIterator_	_First1,
	_FirstForwardIterator_	_Last1,
	_SecondForwardIterator_ _First2,
	_SecondForwardIterator_ _Last2,
	_Predicate_				_Predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_,
			type_traits::IteratorValueType<_FirstForwardIterator_>,
			type_traits::IteratorValueType<_SecondForwardIterator_>>)
{
	__verifyRange(_First1, _Last1);
	__verifyRange(_First2, _Last2);
	
	_SeekPossiblyWrappedIterator(_First1, _FindEndUnchecked(_UnwrapIterator(_First1),
		_UnwrapIterator(_Last1), _UnwrapIterator(_First2), _UnwrapIterator(_Last2),
		type_traits::passFunction(_Predicate)));
}


template <
	class _FirstForwardIterator_,
	class _SecondForwardIterator_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 _FirstForwardIterator_ find_end(
	_FirstForwardIterator_	_First1,
	_FirstForwardIterator_	_Last1,
	_SecondForwardIterator_ _First2,
	_SecondForwardIterator_ _Last2) noexcept(
		type_traits::is_nothrow_invocable_v<
			type_traits::equal_to<>,
			type_traits::IteratorValueType<_FirstForwardIterator_>,
			type_traits::IteratorValueType<_SecondForwardIterator_>
		>
	)
{
	return simd_stl::algorithm::find_end(_First1, _Last1, _First2, _Last2, type_traits::equal_to<>{});
}

template <
	class _ExecutionPolicy_,
	class _FirstForwardIterator_,
	class _SecondForwardIterator_,
	class _Predicate_,
	concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 _FirstForwardIterator_ find_end(
	_ExecutionPolicy_&&,
	_FirstForwardIterator_	_First1,
	_FirstForwardIterator_	_Last1,
	_SecondForwardIterator_ _First2,
	_SecondForwardIterator_ _Last2,
	_Predicate_				_Predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_,
			type_traits::IteratorValueType<_FirstForwardIterator_>,
			type_traits::IteratorValueType<_SecondForwardIterator_>
		>
	)
{
	return simd_stl::algorithm::find_end(_First1, _Last1, _First2, _Last2, type_traits::passFunction(_Predicate));
}

template <
	class _ExecutionPolicy_,
	class _FirstForwardIterator_,
	class _SecondForwardIterator_,
	concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 _FirstForwardIterator_ find_end(
	_ExecutionPolicy_&&,
	_FirstForwardIterator_	_First1,
	_FirstForwardIterator_	_Last1,
	_SecondForwardIterator_ _First2,
	_SecondForwardIterator_ _Last2) noexcept(
		type_traits::is_nothrow_invocable_v<
			type_traits::equal_to<>,
			type_traits::IteratorValueType<_FirstForwardIterator_>,
			type_traits::IteratorValueType<_SecondForwardIterator_>>)
{
	return simd_stl::algorithm::find_end(_First1, _Last1, _First2, _Last2);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
