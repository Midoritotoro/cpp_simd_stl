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
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 _FirstForwardIterator_ search(
	_FirstForwardIterator_	first1,
	_FirstForwardIterator_	last1,
	_SecondForwardIterator_ first2,
	_SecondForwardIterator_ last2,
	_Predicate_				function) noexcept(
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
		
	if constexpr (
		type_traits::is_iterator_random_ranges_v<_FirstForwardIteratorUnwrappedType_> &&
		type_traits::is_iterator_random_ranges_v<_SecondForwardIteratorUnwrappedType_>
	) 
	{
		auto first1Unwrapped		= _UnwrapIterator(first1);
		const auto last1Unwrapped	= _UnwrapIterator(last1);

		const auto first2Unwrapped	= _UnwrapIterator(first2);
		const auto last2Unwrapped	= _UnwrapIterator(last2);

		const auto first1Address	= std::to_address(first1Unwrapped);
		const auto first2Address	= std::to_address(first2Unwrapped);

		auto firstRangeLength		= IteratorsDifference(first1Unwrapped, last1Unwrapped);
		const auto secondRangeLength = IteratorsDifference(first2Unwrapped, last2Unwrapped);

		if (firstRangeLength < secondRangeLength)
			return last1;
	
		if constexpr (type_traits::is_vectorized_search_algorithm_safe_v<
			_FirstForwardIteratorUnwrappedType_, _SecondForwardIteratorUnwrappedType_, _Predicate_>)
		{
#if simd_stl_has_cxx20
			if (type_traits::is_constant_evaluated() == false)
#endif // simd_stl_has_cxx20
			{
				const auto position = SearchVectorized(first1Address, firstRangeLength, first2Address, secondRangeLength);
				return (simd_stl_unlikely(position == nullptr)) ? last1 : first1 + (reinterpret_cast<const _Value_*>(position) - first1Address);
			}
		}

		const auto searched = _Search<arch::CpuFeature::None>()(
			first1Address, firstRangeLength, first2Address,
			secondRangeLength, type_traits::passFunction(function));

		return (simd_stl_unlikely(searched == nullptr)) ? last1 : first1 + (reinterpret_cast<const _Value_*>(searched) - first1Address);
	}


	const auto lastPossible = _UnwrapIterator(last1) - IteratorsDifference(_UnwrapIterator(first2), _UnwrapIterator(last2));
	auto first1Unwrapped = _UnwrapIterator(first1);

	for (;; ++first1Unwrapped) {
		auto mid1 = first1;
		
		for (auto mid2 = first2;; ++mid1, (void) ++mid2) {
			if (mid2 == last2)
				return first1;

			if (mid1 == last1)
				return last1;

			if (!function(*mid1, *mid2))
				break;
		}
	}

	return last1;
}

template <
	class _FirstForwardIterator_,
	class _SecondForwardIterator_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 _FirstForwardIterator_ search(
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
	return simd_stl::algorithm::search(first1, last1, first2, last2, type_traits::equal_to<>{});
}

template <
	class _ForwardIterator_, 
	class _Searcher_>
simd_stl_nodiscard simd_stl_always_inline _ForwardIterator_ search(
	const _ForwardIterator_ first, 
	const _ForwardIterator_ last,
	const _Searcher_&		searcher) noexcept(
		type_traits::is_nothrow_invocable_v<
			type_traits::invocable_type<_Searcher_>,
			type_traits::IteratorValueType<_ForwardIterator_>
		>
	)
{
	return searcher(first, last).first;
}


template <
	class _ExecutionPolicy_,
	class _FirstForwardIterator_,
	class _SecondForwardIterator_,
	class _Predicate_> 
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 _FirstForwardIterator_ search(
	_ExecutionPolicy_&&,
	_FirstForwardIterator_	first1,
	_FirstForwardIterator_	last1,
	_SecondForwardIterator_ first2,
	_SecondForwardIterator_ last2,
	_Predicate_				function) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_,
			type_traits::IteratorValueType<_FirstForwardIterator_>,
			type_traits::IteratorValueType<_SecondForwardIterator_>
		>
	)
{
	return simd_stl::algorithm::search(first1, last1, first2, last2, type_traits::passFunction(function));
}

template <
	class _ExecutionPolicy_,
	class _FirstForwardIterator_,
	class _SecondForwardIterator_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 _FirstForwardIterator_ search(
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
	return simd_stl::algorithm::search(first1, last1, first2, last2);
}

template <
	class _ExecutionPolicy_,
	class _ForwardIterator_, 
	class _Searcher_>
simd_stl_nodiscard simd_stl_always_inline _ForwardIterator_ search(
	_ExecutionPolicy_&&,
	_ForwardIterator_	first, 
	_ForwardIterator_	last,
	const _Searcher_&	searcher) noexcept(
		type_traits::is_nothrow_invocable_v<
			type_traits::invocable_type<_Searcher_>,
			type_traits::IteratorValueType<_ForwardIterator_>
		>
	)
{
	return simd_stl::algorithm::search(first, last, searcher);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
