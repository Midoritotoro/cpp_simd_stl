#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>

#include <src/simd_stl/algorithm/vectorized/SearchVectorized.h>
#include <src/simd_stl/type_traits/CanMemcmpElements.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _FirstForwardIterator_,
	class _SecondForwardIterator_,
	class _Function_> 
simd_stl_nodiscard simd_stl_constexpr_cxx20 _FirstForwardIterator_ search(
	_FirstForwardIterator_	first1,
	_FirstForwardIterator_	last1,
	_SecondForwardIterator_ first2,
	_SecondForwardIterator_ last2,
	_Function_				function) noexcept
{
	using _Value_ = type_traits::IteratorValueType<_FirstForwardIterator_>;

#if !defined(NDEBUG)
	VerifyRange(first1, last1);
	VerifyRange(first2, last2);
#endif

#if defined(simd_stl_cpp_msvc)
	auto first1Unwrapped		= std::_Get_unwrapped(first1);
	const auto last1Unwrapped	= std::_Get_unwrapped(last1);

	const auto first2Unwrapped	= std::_Get_unwrapped(first2);
	const auto last2Unwrapped	= std::_Get_unwrapped(last2);
#endif
	
#if defined(simd_stl_cpp_msvc)
	const auto first1Address = std::to_address(first1Unwrapped);
	const auto first2Address = std::to_address(first2Unwrapped);
#else 
	const auto first1Address = std::to_address(first1);
	const auto first2Address = std::to_address(first2);
#endif


#if defined(simd_stl_cpp_msvc)
	auto firstRangeLength			= IteratorsDifference(first1Unwrapped, last1Unwrapped);
	const auto secondRangeLength	= IteratorsDifference(first2Unwrapped, last2Unwrapped);
#else
	auto firstRangeLength			= IteratorsDifference(first1, last1);
	const auto secondRangeLength	= IteratorsDifference(first2, last2);
#endif // defined(simd_stl_cpp_msvc)
		
	if constexpr (
		type_traits::is_iterator_random_ranges_v<_FirstForwardIterator_> && 
		type_traits::is_iterator_random_ranges_v<_SecondForwardIterator_>
	) {
		if (firstRangeLength < secondRangeLength)
			return last1;
	
		if constexpr (type_traits::is_vectorized_search_algorithm_safe_v<_FirstForwardIterator_, _SecondForwardIterator_, _Function_>) {
#if simd_stl_has_cxx20
			if (type_traits::is_constant_evaluated() == false)
#endif // simd_stl_has_cxx20
			{
				const auto position = SearchVectorized(
					first1Address, firstRangeLength, first2Address, secondRangeLength);

				return first1 + (reinterpret_cast<const _Value_*>(position) - first1Address);
			}
		}

		const auto searched = _Search<arch::CpuFeature::None>()(
			first1Address, firstRangeLength, first2Address, secondRangeLength);

		return (searched == nullptr) 
			? last1 
			: first1 + (reinterpret_cast<const _Value_*>(searched) - first1Address);
	}


	const auto lastPossible = last1Unwrapped - secondRangeLength;
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
simd_stl_nodiscard simd_stl_constexpr_cxx20 _FirstForwardIterator_ search(
	_FirstForwardIterator_	first1,
	_FirstForwardIterator_	last1,
	_SecondForwardIterator_ first2,
	_SecondForwardIterator_ last2) noexcept
{
	return simd_stl::algorithm::search(first1, last1, first2, last2, std::equal_to<>{});
}

__SIMD_STL_ALGORITHM_NAMESPACE_END