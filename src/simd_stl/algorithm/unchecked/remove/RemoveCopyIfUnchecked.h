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
__simd_nodiscard_inline_constexpr _UnwrappedOutputIterator_ _RemoveCopyIfUnchecked(
	_UnwrappedInputIterator_	_FirstUnwrapped,
	_UnwrappedInputIterator_	_LastUnwrapped,
	_UnwrappedOutputIterator_	_DestinationUnwrapped,
	_UnaryPredicate_			_Predicate) noexcept
{
	for (; _FirstUnwrapped != _LastUnwrapped; ++_FirstUnwrapped)
		if (_Predicate(*_FirstUnwrapped) == false)
			*_DestinationUnwrapped++ = std::move(*_FirstUnwrapped);

	return _DestinationUnwrapped;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
