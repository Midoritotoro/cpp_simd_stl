#pragma once 

#include <src/simd_stl/algorithm/vectorized/RemoveCopyVectorized.h>
#include <simd_stl/algorithm/find/Find.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _InputIterator_,
	class _OutputIterator_,
	class _Type_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 _OutputIterator_ remove_copy(
	_InputIterator_			first,
	const _InputIterator_	last,
	_OutputIterator_		destination,
	const _Type_&			value) noexcept
{
	__verifyRange(first, last);

	using _InputIteratorUnwrappedType_	= unwrapped_iterator_type<_InputIterator_>;
	using _OutputIteratorUnwrappedType_ = unwrapped_iterator_type<_OutputIterator_>;

	auto firstUnwrapped			= _UnwrapIterator(first);
	const auto lastUnwrapped	= _UnwrapIterator(last);

	auto destinationUnwrapped	= _UnwrapIterator(destination);

	if constexpr (
		type_traits::is_vectorized_find_algorithm_safe_v<_InputIteratorUnwrappedType_, _Type_>	&&
		type_traits::is_vectorized_find_algorithm_safe_v<_OutputIteratorUnwrappedType_, _Type_> &&
		type_traits::IteratorCopyCategory<_InputIteratorUnwrappedType_, _OutputIteratorUnwrappedType_>::BitcopyAssignable)
	{
#if simd_stl_has_cxx20
		if (type_traits::is_constant_evaluated() == false)
#endif
		{
			if (math::couldCompareEqualToValueType<_InputIteratorUnwrappedType_>(value) == false)
				return destination;

			const auto destinationAddress = std::to_address(destinationUnwrapped);
			auto position = _RemoveCopyVectorized<type_traits::IteratorValueType<_InputIteratorUnwrappedType_>>(
				std::to_address(firstUnwrapped), std::to_address(lastUnwrapped), destinationAddress, value);

			if constexpr (std::is_pointer_v<_OutputIterator_>)
				destinationUnwrapped = reinterpret_cast<type_traits::IteratorValueType<_OutputIteratorUnwrappedType_>*>(position);
			else
				destinationUnwrapped += static_cast<type_traits::IteratorDifferenceType<_OutputIteratorUnwrappedType_>>(
					reinterpret_cast<type_traits::IteratorValueType<_OutputIteratorUnwrappedType_>*>(position) - destinationAddress);

			_SeekPossiblyWrappedIterator(destination, destinationUnwrapped);
			return destination;
		}
	}

	for (; firstUnwrapped != lastUnwrapped; ++firstUnwrapped) {
		const auto firstValue = std::move(*firstUnwrapped);

		if (firstValue != value)
			*destinationUnwrapped++ = std::move(firstValue);
	}

	_SeekPossiblyWrappedIterator(destination, destinationUnwrapped);
	return destination;
}

template <
	class _InputIterator_,
	class _OutputIterator_,
	class _UnaryPredicate_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 _OutputIterator_ remove_copy_if(
	_InputIterator_			first,
	const _InputIterator_	last,
	_OutputIterator_		destination,
	_UnaryPredicate_		predicate) noexcept
{
	__verifyRange(first, last);

	auto firstUnwrapped			= _UnwrapIterator(first);
	const auto lastUnwrapped	= _UnwrapIterator(last);

	auto destinationUnwrapped	= _UnwrapIterator(destination);

	for (; firstUnwrapped != lastUnwrapped; ++firstUnwrapped)
		if (predicate(*firstUnwrapped) == false)
			*destinationUnwrapped++ = std::move(*firstUnwrapped);


	_SeekPossiblyWrappedIterator(destination, destinationUnwrapped);
	return destination;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
