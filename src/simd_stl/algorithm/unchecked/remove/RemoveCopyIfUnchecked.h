#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <src/simd_stl/algorithm/vectorized/remove/RemoveVectorized.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _UnwrappedInputIterator_,
	class _UnwrappedOutputIterator_,
	class _UnaryPredicate_>
__simd_nodiscard_inline_constexpr _UnwrappedOutputIterator_ __remove_copy_if_unchecked(
	_UnwrappedInputIterator_	__first_unwrapped,
	_UnwrappedInputIterator_	__last_unwrapped,
	_UnwrappedOutputIterator_	__destination_unwrapped,
	_UnaryPredicate_			__predicate) noexcept
{
	for (; __first_unwrapped != __last_unwrapped; ++__first_unwrapped)
		if (__predicate(*__first_unwrapped) == false)
			*__destination_unwrapped++ = std::move(*__first_unwrapped);

	return __destination_unwrapped;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
