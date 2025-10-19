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
	class _Function_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 bool ends_with(
	_FirstForwardIterator_	first1,
	_FirstForwardIterator_	last1,
	_SecondForwardIterator_ first2,
	_SecondForwardIterator_ last2,
	_Function_				predicate) noexcept
{
	using _Value_ = type_traits::IteratorValueType<_FirstForwardIterator_>;

#if defined(simd_stl_cpp_msvc) 
	using _FirstForwardIteratorUnwrappedType_ = std::_Unwrapped_t<_FirstForwardIterator_>;
	using _SecondForwardIteratorUnwrappedType_ = std::_Unwrapped_t<_SecondForwardIterator_>;
#else
	using _FirstForwardIteratorUnwrappedType_ = _FirstForwardIterator_;
	using _SecondForwardIteratorUnwrappedType_ = _SecondForwardIterator_;
#endif // defined(simd_stl_cpp_msvc)

	__verifyRange(first1, last1);
	__verifyRange(first2, last2);

	auto first1Unwrapped = __unwrapIterator(first1);
	auto first2Unwrapped = __unwrapIterator(first2);

	auto last1Unwrapped = __unwrapIterator(last1);
	auto last2Unwrapped = __unwrapIterator(last2);

	if constexpr (
		type_traits::is_iterator_random_ranges_v<_FirstForwardIteratorUnwrappedType_> &&
		type_traits::is_iterator_random_ranges_v<_SecondForwardIteratorUnwrappedType_>
	)
	{
		const auto firstRangeLength		= IteratorsDifference(first1Unwrapped, last1Unwrapped);
		const auto secondRangeLength	= IteratorsDifference(first2Unwrapped, last2Unwrapped);

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
		type_traits::is_iterator_forward_ranges_v<_FirstForwardIteratorUnwrappedType_> &&
		type_traits::is_iterator_forward_ranges_v<_SecondForwardIteratorUnwrappedType_>
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
	_SecondForwardIterator_ last2) noexcept
{
	return simd_stl::algorithm::ends_with(first1, last1, first2, last2, type_traits::equal_to<>{});
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
