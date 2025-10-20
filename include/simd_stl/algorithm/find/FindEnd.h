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
	_Predicate_				function) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_,
			type_traits::IteratorValueType<_FirstForwardIterator_>,
			type_traits::IteratorValueType<_SecondForwardIterator_>
		>
	)
{
	using _Value_ = type_traits::IteratorValueType<_FirstForwardIterator_>;

	using _FirstForwardIteratorUnwrappedType_	= type_traits::_Unwrapped_iterator_type<_FirstForwardIterator_>;
	using _SecondForwardIteratorUnwrappedType_	= type_traits::_Unwrapped_iterator_type<_SecondForwardIterator_>;

	__verifyRange(first1, last1);
	__verifyRange(first2, last2);
		
	if constexpr (
		type_traits::is_iterator_random_ranges_v<_FirstForwardIteratorUnwrappedType_> &&
		type_traits::is_iterator_random_ranges_v<_SecondForwardIteratorUnwrappedType_>
	) 
	{
		auto first1Unwrapped		= __unwrapIterator(first1);
		const auto last1Unwrapped	= __unwrapIterator(last1);

		const auto first2Unwrapped	= __unwrapIterator(first2);
		const auto last2Unwrapped	= __unwrapIterator(last2);

		const auto first1Address	= std::to_address(first1Unwrapped);
		const auto first2Address	= std::to_address(first2Unwrapped);

		auto firstRangeLength			= IteratorsDifference(first1Unwrapped, last1Unwrapped);
		const auto secondRangeLength	= IteratorsDifference(first2Unwrapped, last2Unwrapped);

		if (firstRangeLength < secondRangeLength)
			return last1;
	
		if constexpr (type_traits::is_vectorized_search_algorithm_safe_v<
			_FirstForwardIteratorUnwrappedType_, _SecondForwardIteratorUnwrappedType_, _Predicate_>)
		{
			// const auto position = FindEndVectorized
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

__SIMD_STL_ALGORITHM_NAMESPACE_END
