#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <src/simd_stl/algorithm/vectorized/FindVectorized.h>

#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _Iterator_,
	class _Type_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 _Iterator_ find(
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
			const auto position = FindVectorized(firstAddress, std::to_address(lastUnwrapped), value);

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

	_SeekPossiblyWrappedIterator(first, firstUnwrapped);
	return first;
}

template <
	class _InputIterator_,
	class _Predicate_>
simd_stl_nodiscard simd_stl_constexpr_cxx20 simd_stl_always_inline _InputIterator_ find_if_not(
	_InputIterator_			first, 
	const _InputIterator_	last, 
	_Predicate_				predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_,
			type_traits::IteratorValueType<_InputIterator_>
		>
	)
{
	__verifyRange(first, last);

	auto firstUnwrapped			= _UnwrapIterator(first);
	const auto lastUnwrapped	= _UnwrapIterator(last);

	for (; firstUnwrapped != lastUnwrapped; ++firstUnwrapped)
		if (predicate(*firstUnwrapped) == false)
			break;

	_SeekPossiblyWrappedIterator(first, firstUnwrapped);

	return first;
}

template <
	class _InputIterator_,
	class _Predicate_>
simd_stl_nodiscard simd_stl_constexpr_cxx20 simd_stl_always_inline _InputIterator_ find_if(
	_InputIterator_			first, 
	const _InputIterator_	last, 
	_Predicate_				predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_,
			type_traits::IteratorValueType<_InputIterator_>
		>
	)
{
	__verifyRange(first, last);

	auto firstUnwrapped			= _UnwrapIterator(first);
	const auto lastUnwrapped	= _UnwrapIterator(last);

	for (; firstUnwrapped != lastUnwrapped; ++firstUnwrapped)
		if (predicate(*firstUnwrapped))
			break;

	_SeekPossiblyWrappedIterator(first, firstUnwrapped);
	return first;
}

template <
	class _ExecutionPolicy_,
	class _Iterator_,
	class _Type_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 _Iterator_ find(
	_ExecutionPolicy_&&,
	_Iterator_			first,
	const _Iterator_	last,
	const _Type_&		value) noexcept
{
	return simd_stl::algorithm::find(first, last, value);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
