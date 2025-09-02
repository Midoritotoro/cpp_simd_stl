#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <simd_stl/compatibility/Nodiscard.h>
#include <simd_stl/compatibility/Inline.h>

#include <src/simd_stl/algorithm/vectorized/FindVectorized.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _FirstForwardIterator_,
	class _SecondForwardIterator_> 
simd_stl_nodiscard simd_stl_constexpr_cxx20 _FirstForwardIterator_ search(
	_FirstForwardIterator_	first1,
	_FirstForwardIterator_	last1,
	_SecondForwardIterator_ first2,
	_SecondForwardIterator_ last2) noexcept
{
#if !defined(NDEBUG)
	VerifyRange(first1, last1);
	VerifyRange(first2, last2);
#endif

	if constexpr (
		type_traits::is_vectorized_find_algorithm_safe_v<_FirstForwardIterator_, type_traits::IteratorValueType<_FirstForwardIterator_>> &&
		type_traits::is_vectorized_find_algorithm_safe_v<_SecondForwardIterator_, type_traits::IteratorValueType<_SecondForwardIterator_>>
	) {
#if simd_stl_has_cxx20
		if (type_traits::is_constant_evaluated() == false)
#endif // simd_stl_has_cxx20
		{

		}
	}

	const auto firstRangeLength = IteratorsDifference(first1, last1);
	const auto secondRangeLength = IteratorsDifference(first2, last2);

	for (; secondRangeLength <= firstRangeLength; ++first1, --firstRangeLength) {
		auto mid1 = first1;

		for (auto mid2 = first2; ; ++mid1, ++mid2)
			if (mid2 == last2)
				return (first1);
			else if (!(*mid1 == *mid2))
				break;
	}

	return first1;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END