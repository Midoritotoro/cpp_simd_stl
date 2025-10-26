#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <src/simd_stl/algorithm/vectorized/FindVectorized.h>
#include <src/simd_stl/algorithm/vectorized/SearchVectorized.h>

#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _FirstForwardIterator_,
	class _SecondForwardIterator_,
	class _Predicate_>
simd_stl_nodiscard simd_stl_constexpr_cxx20 simd_stl_always_inline _FirstForwardIterator_ find_first_of(
	_FirstForwardIterator_			first1, 
	const _FirstForwardIterator_	last1, 
	const _SecondForwardIterator_	first2,
	const _SecondForwardIterator_	last2, 
	_Predicate_						predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_,
			type_traits::IteratorValueType<_FirstForwardIterator_>,
			type_traits::IteratorValueType<_SecondForwardIterator_>
		>
	)
{
	__verifyRange(first1, last1);
	__verifyRange(first2, last2);

	using _FirstForwardUnwrappedIterator_	= unwrapped_iterator_type<_FirstForwardIterator_>;
	using _SecondForwardUnwrappedIterator_	= unwrapped_iterator_type<_SecondForwardIterator_>;

	auto first1Unwrapped		= _UnwrapIterator(first1);
	auto last1Unwrapped			= _UnwrapIterator(last1);

	auto first2Unwrapped		= _UnwrapIterator(first2);
	const auto last2Unwrapped	= _UnwrapIterator(last2);

	if constexpr (
		type_traits::is_iterator_random_ranges_v<_SecondForwardUnwrappedIterator_> &&
		std::is_same_v<_Predicate_, type_traits::equal_to<>>) 
	{
		const auto length = IteratorsDifference(first2Unwrapped, last2Unwrapped);

		if (length == 1) {
			const auto value = *first2Unwrapped;
			using _ValueType_ = type_traits::IteratorValueType<_SecondForwardUnwrappedIterator_>;

			if (math::couldCompareEqualToValueType<_FirstForwardUnwrappedIterator_>(value) == false)
				return last1;

			const auto first1Address = std::to_address(first1Unwrapped);
			const void* position = nullptr;

			if constexpr (type_traits::is_vectorized_find_algorithm_safe_v<_FirstForwardUnwrappedIterator_, _ValueType_>)
				position = FindVectorized(first1Address, std::to_address(last1Unwrapped), value);
			else
				position = FindScalar(first1Address, std::to_address(last1Unwrapped), value);

			if constexpr (std::is_pointer_v<_FirstForwardIterator_>)
				return reinterpret_cast<_ValueType_*>(position);
			else
				return first1 + static_cast<type_traits::IteratorDifferenceType<_FirstForwardIterator_>>(
					reinterpret_cast<const _ValueType_*>(position) - first1Address);
		}
	}

//	if constexpr (type_traits::is_vectorized_search_algorithm_safe_v<
//		_FirstForwardUnwrappedIterator_, _SecondForwardUnwrappedIterator_, _Predicate_>) 
//	{
//#if simd_stl_has_cxx20
//        if (type_traits::is_constant_evaluated() == false) 
//#endif // simd_stl_has_cxx20
//		{
//			using _ValueType_ = type_traits::IteratorValueType<_SecondForwardUnwrappedIterator_>;
//
//            const auto first1Address	= std::to_address(first1Unwrapped);
//           
//			if constexpr (std::is_pointer_v<_FirstForwardIterator_>)
//				return position;
//			else
//				return first1 + static_cast<type_traits::IteratorDifferenceType<_FirstForwardIterator_>>(position - first1Address);
//        }
//    }

	for (; first1Unwrapped != last1Unwrapped; ++first1Unwrapped) {
        for (auto mid2Unwrapped = first2Unwrapped; mid2Unwrapped != last2Unwrapped; ++mid2Unwrapped) {
            if (predicate(*first1Unwrapped, *mid2Unwrapped)) {
				_SeekPossiblyWrappedIterator(first1, first1Unwrapped);
                return first1;
            }
        }
    }

	_SeekPossiblyWrappedIterator(first1, first1Unwrapped);
    return first1;
}


template <
	class _FirstForwardIterator_,
	class _SecondForwardIterator_>
simd_stl_nodiscard simd_stl_constexpr_cxx20 simd_stl_always_inline _FirstForwardIterator_ find_first_of(
	const _FirstForwardIterator_		first1,
	const _FirstForwardIterator_		last1,
	const _SecondForwardIterator_		first2,
	const _SecondForwardIterator_		last2) noexcept(
		type_traits::is_nothrow_invocable_v<
			type_traits::equal_to<>,
			type_traits::IteratorValueType<_FirstForwardIterator_>,
			type_traits::IteratorValueType<_SecondForwardIterator_>
		>
	)
{
	return simd_stl::algorithm::find_first_of(first1, last1, first2, last2, type_traits::equal_to<>{});
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
