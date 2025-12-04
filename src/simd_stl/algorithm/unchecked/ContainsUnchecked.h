#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <src/simd_stl/algorithm/vectorized/ContainsVectorized.h>

#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>
#include <simd_stl/concurrency/Execution.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _UnwrappedIterator_,
	class _Type_>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 bool _ContainsUnchecked(
	_UnwrappedIterator_			_FirstUnwrapped,
	const _UnwrappedIterator_	_LastUnwrapped,
	const _Type_&				_Value) noexcept
{
	if constexpr (type_traits::is_vectorized_find_algorithm_safe_v<_UnwrappedIterator_, _Type_>) {
#if simd_stl_has_cxx20
		if (type_traits::is_constant_evaluated() == false)
#endif
		{
			if (math::couldCompareEqualToValueType<_UnwrappedIterator_>(_Value) == false)
				return false;

			return _ContainsVectorized(std::to_address(_FirstUnwrapped), std::to_address(_LastUnwrapped), _Value);
		}
	}

	for (; _FirstUnwrapped != _LastUnwrapped; ++_FirstUnwrapped)
		if (*_FirstUnwrapped == _Value)
			return true;

	return false;
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
