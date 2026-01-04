#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>

#include <src/simd_stl/algorithm/vectorized/find/SearchVectorized.h>
#include <src/simd_stl/type_traits/CanMemcmpElements.h>

#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>
#include <src/simd_stl/type_traits/OperatorWrappers.h>

#include <simd_stl/concurrency/Execution.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _FirstForwardIterator_,
	class _SecondForwardIterator_,
	class _Predicate_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 bool ends_with(
	_FirstForwardIterator_	first1,
	_FirstForwardIterator_	last1,
	_SecondForwardIterator_ first2,
	_SecondForwardIterator_ last2,
	_Predicate_				predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_,
			type_traits::iterator_value_type<_FirstForwardIterator_>,
			type_traits::iterator_value_type<_SecondForwardIterator_>
		>
	)
{
	using _Value_ = type_traits::iterator_value_type<_FirstForwardIterator_>;

	using _FirstForwardIteratorUnwrappedType_	= unwrapped_iterator_type<_FirstForwardIterator_>;
	using _SecondForwardIteratorUnwrappedType_	= unwrapped_iterator_type<_SecondForwardIterator_>;

	__verifyRange(first1, last1);
	__verifyRange(first2, last2);

	auto first1Unwrapped	= _UnwrapIterator(first1);
	auto first2Unwrapped	= _UnwrapIterator(first2);

	auto last1Unwrapped		= _UnwrapIterator(last1);
	auto last2Unwrapped		= _UnwrapIterator(last2);

	if constexpr (
		type_traits::is_iterator_random_ranges_v<_FirstForwardIteratorUnwrappedType_> &&
		type_traits::is_iterator_random_ranges_v<_SecondForwardIteratorUnwrappedType_>
	)
	{
		const auto firstRangeLength		= __iterators_difference(first1Unwrapped, last1Unwrapped);
		const auto secondRangeLength	= __iterators_difference(first2Unwrapped, last2Unwrapped);

		if (firstRangeLength < secondRangeLength)
			return false;

		if (secondRangeLength == 1)
			return (predicate(*--last1Unwrapped, *first2Unwrapped));

		first1Unwrapped += (firstRangeLength - secondRangeLength);

		for (sizetype current = 0; current < secondRangeLength; ++current, ++first1Unwrapped, ++first2Unwrapped)
			if (predicate(*first1Unwrapped, *first2Unwrapped) == false)
				return false;
		
		return true;
	}
	else if constexpr (
		type_traits::is_iterator_forward_ranges_v<_FirstForwardIteratorUnwrappedType_> &&
		type_traits::is_iterator_forward_ranges_v<_SecondForwardIteratorUnwrappedType_>
	)
	{
		const auto secondRangeLength = std::distance(first2Unwrapped, last2Unwrapped);

		if (secondRangeLength == 0)
			return true;

		const auto firstRangeLength = std::distance(first1Unwrapped, last1Unwrapped);

		if (firstRangeLength < secondRangeLength)
			return false;

		std::advance(first1Unwrapped, firstRangeLength - secondRangeLength);

		for (; first2Unwrapped != last2Unwrapped; ++first1Unwrapped, ++first2Unwrapped)
			if (!predicate(*first1Unwrapped, *first2Unwrapped))
				return false;

		return true;
	}
	else if constexpr (
		type_traits::is_iterator_bidirectional_ranges_v<_FirstForwardIteratorUnwrappedType_> &&
		type_traits::is_iterator_bidirectional_ranges_v<_SecondForwardIteratorUnwrappedType_>
	)
	{
		while (last2Unwrapped != first2Unwrapped) {
			if (last1Unwrapped == first1Unwrapped)
				return false;
			
			--last1Unwrapped;
			--last2Unwrapped;

			if (!predicate(*last1Unwrapped, *last2Unwrapped))
				return false;
		}

		return true;
	}
}

template <
	class _FirstForwardIterator_,
	class _SecondForwardIterator_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 bool ends_with(
	_FirstForwardIterator_	first1,
	_FirstForwardIterator_	last1,
	_SecondForwardIterator_ first2,
	_SecondForwardIterator_ last2) noexcept(
		type_traits::is_nothrow_invocable_v<
			type_traits::equal_to<>,
			type_traits::iterator_value_type<_FirstForwardIterator_>,
			type_traits::iterator_value_type<_SecondForwardIterator_>
		>
	)
{
	return simd_stl::algorithm::ends_with(first1, last1, first2, last2, type_traits::equal_to<>{});
}

template <
	class _ExecutionPolicy_,
	class _FirstForwardIterator_,
	class _SecondForwardIterator_,
	class _Predicate_,
	concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 bool ends_with(
	_ExecutionPolicy_&&,
	_FirstForwardIterator_	first1,
	_FirstForwardIterator_	last1,
	_SecondForwardIterator_ first2,
	_SecondForwardIterator_ last2,
	_Predicate_				predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_,
			type_traits::iterator_value_type<_FirstForwardIterator_>,
			type_traits::iterator_value_type<_SecondForwardIterator_>
		>
	)
{
	return simd_stl::algorithm::ends_with(first1, last1, first2, last2, type_traits::passFunction(predicate));
}

template <
	class _ExecutionPolicy_,
	class _FirstForwardIterator_,
	class _SecondForwardIterator_,
	concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 bool ends_with(
	_ExecutionPolicy_&&,
	_FirstForwardIterator_	first1,
	_FirstForwardIterator_	last1,
	_SecondForwardIterator_ first2,
	_SecondForwardIterator_ last2) noexcept(
		type_traits::is_nothrow_invocable_v<
			type_traits::equal_to<>,
			type_traits::iterator_value_type<_FirstForwardIterator_>,
			type_traits::iterator_value_type<_SecondForwardIterator_>
		>
	)
{
	return simd_stl::algorithm::ends_with(first1, last1, first2, last2);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
