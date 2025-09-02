#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <simd_stl/compatibility/Nodiscard.h>
#include <simd_stl/compatibility/Inline.h>

#include <src/simd_stl/algorithm/vectorized/FindVectorized.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _Iterator_,
	class _Type_>
simd_stl_nodiscard simd_stl_constexpr_cxx20 _Iterator_ find(
	_Iterator_			firstIterator,
	const _Iterator_	lastIterator,
	const _Type_&		value) noexcept
{
#if !defined(NDEBUG)
	VerifyRange(firstIterator, lastIterator);
#endif // !defined(NDEBUG)

	if constexpr (type_traits::is_vectorized_find_algorithm_safe_v<_Iterator_, _Type_>) {
#if simd_stl_has_cxx20
		if (type_traits::is_constant_evaluated() == false)
#endif
		{
			const auto firstAddress = std::to_address(firstIterator);
			const auto position = FindVectorized(firstAddress, std::to_address(lastIterator), value);

			return firstIterator + (reinterpret_cast<const _Type_*>(position) - firstAddress);
		}
	}

	for (; firstIterator != lastIterator; ++firstIterator)
		if (*firstIterator == value)
			break;

	return firstIterator;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
