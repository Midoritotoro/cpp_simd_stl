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
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 _FirstForwardIterator_ search(
	_FirstForwardIterator_	first1,
	_FirstForwardIterator_	last1,
	_SecondForwardIterator_ first2,
	_SecondForwardIterator_ last2,
	_Function_				function) noexcept
{
	using _Value_ = type_traits::IteratorValueType<_FirstForwardIterator_>;

#if defined(simd_stl_cpp_msvc) 
	using _FirstForwardIteratorUnwrappedType_	= std::_Unwrapped_t<_FirstForwardIterator_>;
	using _SecondForwardIteratorUnwrappedType_	= std::_Unwrapped_t<_SecondForwardIterator_>;
#else
	using _FirstForwardIteratorUnwrappedType_	= _FirstForwardIterator_;
	using _SecondForwardIteratorUnwrappedType_	= _SecondForwardIterator_;
#endif // defined(simd_stl_cpp_msvc)

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

		auto firstRangeLength		= IteratorsDifference(first1Unwrapped, last1Unwrapped);
		const auto secondRangeLength = IteratorsDifference(first2Unwrapped, last2Unwrapped);

		if (firstRangeLength < secondRangeLength)
			return last1;
	
		if constexpr (type_traits::is_vectorized_search_algorithm_safe_v<
			_FirstForwardIteratorUnwrappedType_, _SecondForwardIteratorUnwrappedType_, _Function_>)
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


	const auto lastPossible = __unwrapIterator(last1) - IteratorsDifference(__unwrapIterator(first2), __unwrapIterator(last2));
	auto first1Unwrapped = __unwrapIterator(first1);

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
	_SecondForwardIterator_ last2) noexcept
{
	return simd_stl::algorithm::search(first1, last1, first2, last2, type_traits::equal_to<>{});
}

template <
	class _ForwardIterator_, 
	class _Searcher_>
simd_stl_nodiscard simd_stl_always_inline _ForwardIterator_ search(
	const _ForwardIterator_ first, 
	const _ForwardIterator_ last,
	const _Searcher_&		searcher) noexcept
{
	return searcher(first, last).first;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
