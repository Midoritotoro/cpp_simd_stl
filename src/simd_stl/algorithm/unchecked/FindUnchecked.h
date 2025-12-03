#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <src/simd_stl/algorithm/vectorized/FindVectorized.h>

#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>
#include <simd_stl/concurrency/Execution.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _UnwrappedIterator_,
	class _Type_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 _UnwrappedIterator_ _FindUnchecked(
	_UnwrappedIterator_			firstUnwrapped,
	const _UnwrappedIterator_	lastUnwrapped,
	const _Type_&				value) noexcept
{
	using _DifferenceType_ = type_traits::IteratorDifferenceType<_UnwrappedIterator_>;

	if constexpr (type_traits::is_vectorized_find_algorithm_safe_v<_UnwrappedIterator_, _Type_>) {
#if simd_stl_has_cxx20
		if (type_traits::is_constant_evaluated() == false)
#endif
		{
			if (math::couldCompareEqualToValueType<_UnwrappedIterator_>(value) == false)
				return lastUnwrapped;

			const auto firstAddress = std::to_address(firstUnwrapped);
			const auto position = _FindVectorized(firstAddress, std::to_address(lastUnwrapped), value);

			if constexpr (std::is_pointer_v<_UnwrappedIterator_>)
				return position;
			else
				return firstUnwrapped + static_cast<_DifferenceType_>(position - firstAddress);
		}
	}

	for (; firstUnwrapped != lastUnwrapped; ++firstUnwrapped)
		if (*firstUnwrapped == value)
			break;

	return firstUnwrapped;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
