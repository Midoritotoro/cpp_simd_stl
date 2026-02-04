#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <src/simd_stl/algorithm/vectorized/find/ContainsVectorized.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>

#include <src/simd_stl/datapar/IsComparable.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _UnwrappedIterator_,
	class _Type_ = type_traits::iterator_value_type<_UnwrappedIterator_>>
__simd_nodiscard_inline_constexpr bool __contains_unchecked(
	_UnwrappedIterator_									__first_unwrapped,
	_UnwrappedIterator_									__last_unwrapped,
	const typename std::type_identity<_Type_>::type&	__value) noexcept
{
	if constexpr (type_traits::__is_vectorized_find_algorithm_safe_v<_UnwrappedIterator_, _Type_>) {
#if simd_stl_has_cxx20
		if (type_traits::is_constant_evaluated() == false)
#endif // simd_stl_has_cxx20
		{
			return __contains_vectorized(std::to_address(__first_unwrapped), std::to_address(__last_unwrapped), __value);
		}
	}

	for (; __first_unwrapped != __last_unwrapped; ++__first_unwrapped)
		if (*__first_unwrapped == __value)
			return true;

	return false;
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
