#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <simd_stl/compatibility/Nodiscard.h>
#include <simd_stl/compatibility/Inline.h>

#include <src/simd_stl/algorithm/vectorized/CountVectorized.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _Iterator_,
	class _Type_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 sizetype count(
	const _Iterator_	first,
	const _Iterator_	last,
	const _Type_&		value) noexcept
{
	__verifyRange(first, last);

	using _IteratorUnwrappedType_ = unwrapped_iterator_type<_Iterator_>;

	auto firstUnwrapped			= _UnwrapIterator(first);
	const auto lastUnwrapped	= _UnwrapIterator(last);

	if constexpr (type_traits::is_iterator_random_ranges_v<_IteratorUnwrappedType_>) {
		const auto length = ByteLength(firstUnwrapped, lastUnwrapped);

		if constexpr (type_traits::is_vectorized_find_algorithm_safe_v<_IteratorUnwrappedType_, _Type_>) {
#if simd_stl_has_cxx20
			if (type_traits::is_constant_evaluated() == false)
#endif
			{
				if (math::couldCompareEqualToValueType<_IteratorUnwrappedType_>(value) == false)
					return 0;

				return CountVectorized<_Type_>(std::to_address(firstUnwrapped), length, value);
			}
		}
	}

    type_traits::IteratorDifferenceType<_Iterator_> count = 0;

	for (; firstUnwrapped != lastUnwrapped; ++firstUnwrapped)
		count += (*firstUnwrapped == value);

	return count;
}

template <
	class _InputIterator_,
	class _Predicate_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 type_traits::IteratorDifferenceType<_InputIterator_> count_if(
	_InputIterator_			first,
	const _InputIterator_	last,
	_Predicate_ 			predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_,
			type_traits::IteratorValueType<_InputIterator_>
		>
	)
{
	__verifyRange(first, last);

	auto firstUnwrapped			= _UnwrapIterator(first);
	const auto lastUnwrapped	= _UnwrapIterator(last);

	auto count = type_traits::IteratorDifferenceType<_InputIterator_>(0);

	for (; firstUnwrapped != lastUnwrapped; ++firstUnwrapped)
		if (predicate(*firstUnwrapped))
			++count;

	return count;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
