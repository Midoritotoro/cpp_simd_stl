#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <src/simd_stl/algorithm/vectorized/FindVectorized.h>
#include <src/simd_stl/algorithm/vectorized/SearchVectorized.h>

#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>

template <
	class _FirstForwardIterator_,
	class _SecondForwardIterator_,
	class _Predicate_>
simd_stl_nodiscard simd_stl_constexpr_cxx20 simd_stl_always_inline _FirstForwardIterator_ find_first_of(
	_FirstForwardIterator_			first1, 
	const _FirstForwardIterator_	last1, 
	const _SecondForwardIterator_	first2,
	const _SecondForwardIterator_	last2, 
	_Predicate_						predicate)
{
	__verifyRange(first1, last1);
	__verifyRange(first2, last2);

#if defined(simd_stl_cpp_msvc)
	using _FirstForwardUnwrappedIterator_	= std::_Unwrapped_t<_FirstForwardIterator_>;
	using _SecondForwardUnwrappedIterator_	= std::_Unwrapped_t<_SecondForwardIterator_>;
#else 
	using _FirstForwardUnwrappedIterator_	= _FirstForwardIterator_;
	using _SecondForwardUnwrappedIterator_	= _SecondForwardIterator_;
#endif

	auto first1Unwrapped		= __unwrapIterator(first1);
	auto last1Unwrapped			= __unwrapIterator(last1);

	auto first2Unwrapped		= __unwrapIterator(first2);
	const auto last2Unwrapped	= __unwrapIterator(last2);

	if constexpr (
		type_traits::is_iterator_random_ranges_v<_SecondForwardUnwrappedIterator_> 
		&& std::is_same_v<_Predicate_, type_traits::equal_to<>>) 
	{
		const auto length = IteratorsDifference(first2Unwrapped, last2Unwrapped);

		if (length == 1) {
			first1Unwrapped = FindVectorized<type_traits::IteratorValueType<_SecondForwardUnwrappedIterator_>>(
				std::to_address(first1Unwrapped), std::to_address(last1Unwrapped), *first2Unwrapped);

#if defined(simd_stl_cpp_msvc)
			std::_Seek_wrapped(first1, first1Unwrapped);
#else 
			first1 = first1Unwrapped;
#endif
			return first1;
		}
	}

	if constexpr (type_traits::is_vectorized_search_algorithm_safe_v<
		_FirstForwardUnwrappedIterator_, _SecondForwardUnwrappedIterator_, _Predicate_>) 
	{
#if simd_stl_has_cxx20
        if (type_traits::is_constant_evaluated() == false) 
#endif // simd_stl_has_cxx20
		{
            const auto first1Pointer	= std::to_address(first1Unwrapped);
            const auto position			= SearchVectorized<type_traits::IteratorValueType<_SecondForwardUnwrappedIterator_>>(
				first1Pointer, IteratorsDifference(first1Unwrapped, last1Unwrapped), 
				std::to_address(first2Unwrapped), IteratorsDifference(first2Unwrapped, last2Unwrapped));

            if constexpr (std::is_pointer_v<decltype(first1Unwrapped)>)
				first1Unwrapped = position;
            else
				first1Unwrapped += static_cast<type_traits::IteratorDifferenceType<_FirstForwardUnwrappedIterator_>>(position - first1Pointer);
            
#if defined(simd_stl_cpp_msvc)
			std::_Seek_wrapped(first1, first1Unwrapped);
#else 
			first1 = first1Unwrapped;
#endif

            return first1;
        }
    }

	for (; first1Unwrapped != last1Unwrapped; ++first1Unwrapped) {
        for (auto mid2Unwrapped = first2Unwrapped; mid2Unwrapped != last2Unwrapped; ++mid2Unwrapped) {
            if (predicate(*first1Unwrapped, *mid2Unwrapped)) {

#if defined(simd_stl_cpp_msvc)
				std::_Seek_wrapped(first1, first1Unwrapped);
#else 
				first1 = first1Unwrapped;
#endif

                return first1;
            }
        }
    }

#if defined(simd_stl_cpp_msvc)
	std::_Seek_wrapped(first1, first1Unwrapped);
#else 
	first1 = first1Unwrapped;
#endif

    return first1;
}
