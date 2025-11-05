#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <src/simd_stl/algorithm/vectorized/RemoveVectorized.h>

#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _Iterator_,
	class _Type_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 _Iterator_ remove(
	_Iterator_			first,
	const _Iterator_	last,
	const _Type_&		value) noexcept
{
	__verifyRange(first, last);

	using _IteratorUnwrappedType_ = unwrapped_iterator_type<_Iterator_>;

	auto firstUnwrapped			= _UnwrapIterator(first);
	const auto lastUnwrapped	= _UnwrapIterator(last);

	if constexpr (type_traits::is_vectorized_find_algorithm_safe_v<_IteratorUnwrappedType_, _Type_>) {
#if simd_stl_has_cxx20
		if (type_traits::is_constant_evaluated() == false)
#endif
		{
			if (math::couldCompareEqualToValueType<_IteratorUnwrappedType_>(value) == false)
				return last;

			const auto firstAddress = std::to_address(firstUnwrapped);
			const auto position		= _RemoveVectorized(firstAddress, std::to_address(lastUnwrapped), value);

			if constexpr (std::is_pointer_v<_Iterator_>)
				_SeekPossiblyWrappedIterator(first, reinterpret_cast<const _Type_*>(position));
			else
				_SeekPossiblyWrappedIterator(first, first + static_cast<type_traits::IteratorDifferenceType<_Iterator_>>(
					reinterpret_cast<const _Type_*>(position) - firstAddress));

			return first;
		}
	}

	for (; firstUnwrapped != lastUnwrapped; ++firstUnwrapped)
		if (*firstUnwrapped == value)
			break;

	for (auto current = firstUnwrapped; ++current != lastUnwrapped;) {
		const auto currentValue = std::move(*current);

		if (currentValue != value)
			*firstUnwrapped++ = std::move(currentValue);
	}

	_SeekPossiblyWrappedIterator(first, firstUnwrapped);
	return first;
}

template <
	class _Iterator_,
	class _UnaryPredicate_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 _Iterator_ remove_if(
	_Iterator_			first,
	const _Iterator_	last,
	_UnaryPredicate_	predicate) noexcept(
		type_traits::is_nothrow_invocable_v<_UnaryPredicate_,
		type_traits::IteratorValueType<_Iterator_>>)
{
	__verifyRange(first, last);

	using _IteratorUnwrappedType_ = unwrapped_iterator_type<_Iterator_>;

	auto firstUnwrapped			= _UnwrapIterator(first);
	const auto lastUnwrapped	= _UnwrapIterator(last);

	for (; firstUnwrapped != lastUnwrapped; ++firstUnwrapped)
		if (predicate(*firstUnwrapped))
			break;

	if (firstUnwrapped == lastUnwrapped)
		return last;

	for (auto current = firstUnwrapped; ++current != lastUnwrapped;) {
		const auto currentValue = std::move(*current);

		if (predicate(currentValue) == false)
			*firstUnwrapped++ = std::move(currentValue);
	}

	_SeekPossiblyWrappedIterator(first, firstUnwrapped);
	return first;
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
