#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <simd_stl/compatibility/Nodiscard.h>
#include <simd_stl/compatibility/Inline.h>

#include <src/simd_stl/algorithm/vectorized/FindVectorized.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _Iterator_,
	class _Type_>
simd_stl_nodiscard simd_stl_constexpr_cxx20 _Iterator_ find(
	_Iterator_			first,
	const _Iterator_	last,
	const _Type_&		value) noexcept
{
	__verifyRange(first, last);

	auto firstUnwrapped			= __unwrapIterator(first);
	const auto lastUnwrapped	= __unwrapIterator(last);

	if constexpr (type_traits::is_vectorized_find_algorithm_safe_v<_Iterator_, _Type_>) {
#if simd_stl_has_cxx20
		if (type_traits::is_constant_evaluated() == false)
#endif
		{
			const auto firstAddress = std::to_address(firstUnwrapped);
			const auto position = FindVectorized(firstAddress, std::to_address(lastUnwrapped), value);

			return firstUnwrapped + (reinterpret_cast<const _Type_*>(position) - firstAddress);
		}
	}

	for (; firstUnwrapped != lastUnwrapped; ++firstUnwrapped)
		if (*firstUnwrapped == value)
			break;

	return first;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
