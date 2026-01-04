#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>

#include <src/simd_stl/algorithm/vectorized/SearchVectorized.h>
#include <src/simd_stl/type_traits/CanMemcmpElements.h>

#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>
#include <src/simd_stl/type_traits/OperatorWrappers.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _FirstForwardIterator_,
	class _SecondForwardIterator_,
	class _Predicate_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 bool starts_with(
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

	auto first1Unwrapped = _UnwrapIterator(first1);
	auto first2Unwrapped = _UnwrapIterator(first2);

	const auto last1Unwrapped = _UnwrapIterator(last1);
	const auto last2Unwrapped = _UnwrapIterator(last2);

	if constexpr (
		type_traits::is_iterator_random_ranges_v<_FirstForwardIteratorUnwrappedType_> &&
		type_traits::is_iterator_random_ranges_v<_SecondForwardIteratorUnwrappedType_>
	)
	{
		const auto firstRangeLength		= IteratorsDifference(first1Unwrapped, last1Unwrapped);
		const auto secondRangeLength	= IteratorsDifference(first2Unwrapped, last2Unwrapped);

		if (firstRangeLength < secondRangeLength)
			return false;

		for (sizetype current = 0; current < secondRangeLength; ++current, ++first1Unwrapped, ++first2Unwrapped)
			if (predicate(*first1Unwrapped, *first2Unwrapped) == false)
				return false;
		
		return true;
	}
	else {
		for (; first1Unwrapped != last1Unwrapped && first2Unwrapped != last2Unwrapped; ++first1Unwrapped, ++first2Unwrapped)
			if (!predicate(*first1Unwrapped, *first2Unwrapped))
				return false;

		return (first2Unwrapped == last2Unwrapped);
	}
}

template <
	class _FirstForwardIterator_,
	class _SecondForwardIterator_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 bool starts_with(
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
	return simd_stl::algorithm::starts_with(first1, last1, first2, last2, type_traits::equal_to<>{});
}

template <
	class _ExecutionPolicy_,
	class _FirstForwardIterator_,
	class _SecondForwardIterator_,
	class _Predicate_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 bool starts_with(
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
	return simd_stl::algorithm::starts_with(first1, last1, first2, last2, type_traits::passFunction(predicate));
}

template <
	class _ExecutionPolicy_,
	class _FirstForwardIterator_,
	class _SecondForwardIterator_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 bool starts_with(
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
	return simd_stl::algorithm::starts_with(first1, last1, first2, last2);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
