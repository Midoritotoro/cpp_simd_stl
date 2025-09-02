#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <src/simd_stl/algorithm/vectorized/SearchVectorized.h>


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
			const auto first1Address = std::to_address(first1);
			const auto position = SearchVectorized(
				first1Address, std::to_address(last1),
				std::to_address(first2), std::to_address(last2));

			return first1 + (reinterpret_cast<const type_traits::IteratorValueType<_FirstForwardIterator_>*>(position) - first1Address);
		}
	}

#if defined(simd_stl_cpp_msvc)
	auto firstRangeLength			= IteratorsDifference(std::_Get_unwrapped(first1), std::_Get_unwrapped(last1));
	const auto secondRangeLength	= IteratorsDifference(std::_Get_unwrapped(first2), std::_Get_unwrapped(last2));
#else
	auto firstRangeLength			= IteratorsDifference(first1, last1);
	const auto secondRangeLength	= IteratorsDifference(first2, last2);
#endif // defined(simd_stl_cpp_msvc)

    if (firstRangeLength == secondRangeLength)
        return (memcmp(std::to_address(first1), std::to_address(first2), firstRangeLength) == 0) ? first1 : last1;

    const auto first = *first2;
    const sizetype maxpos = sizetype(firstRangeLength) - sizetype(secondRangeLength) + 1;

    for (sizetype i = 0; i < maxpos; i++) {
        if (first1[i] != first) {
            i++;

            while (i < maxpos && first1[i] != first)
                i++;

            if (i == maxpos)
                break;
        }

        sizetype j = 1;

        for (; j < secondRangeLength; ++j)
            if (first1[i + j] != first2[j])
                break;

        if (j == secondRangeLength)
            return (first1 + i);
    }

	return last1;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END