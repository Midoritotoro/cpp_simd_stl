#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <src/simd_stl/algorithm/vectorized/FindLastVectorized.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _InputUnwrappedIterator_,
	class _Predicate_>
_Simd_nodiscard_inline_constexpr _InputUnwrappedIterator_ _FindLastIfNotUnchecked(
	_InputUnwrappedIterator_	_FirstUnwrapped,
	_InputUnwrappedIterator_	_LastUnwrapped,
	_Predicate_					_Predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_, type_traits::IteratorValueType<_InputUnwrappedIterator_>>)
{
	const auto _Last = _LastUnwrapped;

	while (_LastUnwrapped != _FirstUnwrapped) {
		--_LastUnwrapped;

		if (_Predicate(*_LastUnwrapped) == false)
			return _LastUnwrapped;
	}

	return _Last;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
