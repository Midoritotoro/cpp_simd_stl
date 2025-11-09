#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>

#include <src/simd_stl/algorithm/vectorized/SearchVectorized.h>
#include <src/simd_stl/type_traits/CanMemcmpElements.h>

#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>
#include <src/simd_stl/type_traits/OperatorWrappers.h>

#include <src/simd_stl/algorithm/vectorized/FindEndVectorized.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _FirstForwardIterator_,
	class _SecondForwardIterator_,
	class _Predicate_> 
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 _FirstForwardIterator_ find_end(
	_FirstForwardIterator_	first1,
	_FirstForwardIterator_	last1,
	_SecondForwardIterator_ first2,
	_SecondForwardIterator_ last2,
	_Predicate_				predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_,
			type_traits::IteratorValueType<_FirstForwardIterator_>,
			type_traits::IteratorValueType<_SecondForwardIterator_>
		>
	)
{
	using _Value_ = type_traits::IteratorValueType<_FirstForwardIterator_>;

	using _FirstForwardIteratorUnwrappedType_	= unwrapped_iterator_type<_FirstForwardIterator_>;
	using _SecondForwardIteratorUnwrappedType_	= unwrapped_iterator_type<_SecondForwardIterator_>;

	__verifyRange(first1, last1);
	__verifyRange(first2, last2);

	auto first1Unwrapped		= _UnwrapIterator(first1);
	const auto last1Unwrapped	= _UnwrapIterator(last1);

	const auto first2Unwrapped	= _UnwrapIterator(first2);
	const auto last2Unwrapped	= _UnwrapIterator(last2);
		
	if constexpr (
		type_traits::is_iterator_random_ranges_v<_FirstForwardIteratorUnwrappedType_> &&
		type_traits::is_iterator_random_ranges_v<_SecondForwardIteratorUnwrappedType_>
	) 
	{
		auto firstRangeLength			= IteratorsDifference(first1Unwrapped, last1Unwrapped);
		const auto secondRangeLength	= IteratorsDifference(first2Unwrapped, last2Unwrapped);

		if (firstRangeLength < secondRangeLength || secondRangeLength == 0)
			return last1;
	
		if constexpr (type_traits::is_vectorized_search_algorithm_safe_v<
			_FirstForwardIteratorUnwrappedType_, _SecondForwardIteratorUnwrappedType_, _Predicate_>)
		{
			const auto first1Address = std::to_address(first1Unwrapped);
			const auto first2Address = std::to_address(first2Unwrapped);

			const auto position = FindEndVectorized<_Value_>(first1Address, firstRangeLength, first2Address, secondRangeLength);

			if constexpr (std::is_pointer_v<_FirstForwardIterator_>)
				_SeekPossiblyWrappedIterator(first1, reinterpret_cast<const _Value_*>(position));
			else
				_SeekPossiblyWrappedIterator(first1, first1 + static_cast<type_traits::IteratorDifferenceType<_FirstForwardIterator_>>(
					reinterpret_cast<const _Value_*>(position) - first1Address));

			return first1;
		}
	}
	else if constexpr (
		type_traits::is_iterator_bidirectional_ranges_v<_FirstForwardIteratorUnwrappedType_> && 
		type_traits::is_iterator_bidirectional_ranges_v<_SecondForwardIteratorUnwrappedType_>)
	{
		for (auto candidateUnwrapped = last1Unwrapped;; --candidateUnwrapped) {
			auto next1Unwrapped = candidateUnwrapped;
			auto next2Unwrapped = last2Unwrapped;

			for (;;) {
				if (first2Unwrapped == next2Unwrapped) {
					_SeekPossiblyWrappedIterator(first1, next1Unwrapped);
					return first1;
				}

				if (first1Unwrapped == next1Unwrapped)
					return last1;

				--next1Unwrapped;
				--next2Unwrapped;

				if (predicate(*next1Unwrapped, *next2Unwrapped) == false)
					break;
			}
		}
	}
	else
	{
		auto resultUnwrapped = last1Unwrapped;

		for (;;) {
			auto next1Unwrapped = first1Unwrapped;
			auto next2Unwrapped = first2Unwrapped;

			for (;;) {
				const auto needleEnd = (next2Unwrapped == last2Unwrapped);
				if (needleEnd)
					resultUnwrapped = first1Unwrapped;

				if (next1Unwrapped == last1Unwrapped) {
					_SeekPossiblyWrappedIterator(first1, resultUnwrapped);
					return first1;
				}

				if (needleEnd || predicate(*next1Unwrapped, *next2Unwrapped) == false)

					++next1Unwrapped;
				++next2Unwrapped;
			}

			++first1Unwrapped;

			_SeekPossiblyWrappedIterator(first1, resultUnwrapped);
			return first1;
		}
	}
}


template <
	class _FirstForwardIterator_,
	class _SecondForwardIterator_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 _FirstForwardIterator_ find_end(
	_FirstForwardIterator_	first1,
	_FirstForwardIterator_	last1,
	_SecondForwardIterator_ first2,
	_SecondForwardIterator_ last2) noexcept(
		type_traits::is_nothrow_invocable_v<
			type_traits::equal_to<>,
			type_traits::IteratorValueType<_FirstForwardIterator_>,
			type_traits::IteratorValueType<_SecondForwardIterator_>
		>
	)
{
	return simd_stl::algorithm::find_end(first1, last1, first2, last2, type_traits::equal_to<>{});
}


template <
	class _ExecutionPolicy_,
	class _FirstForwardIterator_,
	class _SecondForwardIterator_,
	class _Predicate_> 
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 _FirstForwardIterator_ find_end(
	_ExecutionPolicy_&&,
	_FirstForwardIterator_	first1,
	_FirstForwardIterator_	last1,
	_SecondForwardIterator_ first2,
	_SecondForwardIterator_ last2,
	_Predicate_				predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_,
			type_traits::IteratorValueType<_FirstForwardIterator_>,
			type_traits::IteratorValueType<_SecondForwardIterator_>
		>
	)
{
	return simd_stl::algorithm::find_end(first1, last1, first2, last2, type_traits::passFunction(predicate));
}

template <
	class _ExecutionPolicy_,
	class _FirstForwardIterator_,
	class _SecondForwardIterator_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 _FirstForwardIterator_ find_end(
	_ExecutionPolicy_&&,
	_FirstForwardIterator_	first1,
	_FirstForwardIterator_	last1,
	_SecondForwardIterator_ first2,
	_SecondForwardIterator_ last2) noexcept(
		type_traits::is_nothrow_invocable_v<
			type_traits::equal_to<>,
			type_traits::IteratorValueType<_FirstForwardIterator_>,
			type_traits::IteratorValueType<_SecondForwardIterator_>
		>
	)
{
	return simd_stl::algorithm::find_end(first1, last1, first2, last2);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
