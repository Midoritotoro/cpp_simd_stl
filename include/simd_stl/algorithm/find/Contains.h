#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <simd_stl/compatibility/Nodiscard.h>
#include <simd_stl/compatibility/Inline.h>

#include <src/simd_stl/algorithm/vectorized/ContainsVectorized.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _Iterator_,
	class _Type_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 bool contains(
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
				return false;

			return ContainsVectorized(std::to_address(firstUnwrapped), std::to_address(lastUnwrapped), value);
		}
	}

	for (; firstUnwrapped != lastUnwrapped; ++firstUnwrapped)
		if (*firstUnwrapped == value)
			return true;

	return false;
}

template <
	class _ExecutionPolicy_,
	class _Iterator_,
	class _Type_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 bool contains(
	_Iterator_			first,
	const _Iterator_	last,
	const _Type_&		value) noexcept
{
	return simd_stl::algorithm::contains(first, last, value);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
