#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _UnwrappedInputIterator_,
	class _Predicate_>
_Simd_nodiscard_inline_constexpr type_traits::IteratorDifferenceType<_UnwrappedInputIterator_> _CountIfUnchecked(
	_UnwrappedInputIterator_	_FirstUnwrapped,
	_UnwrappedInputIterator_	_LastUnwrapped,
	_Predicate_ 				_Predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
		_Predicate_, type_traits::IteratorValueType<_UnwrappedInputIterator_>>)
{
	auto _Count = type_traits::IteratorDifferenceType<_UnwrappedInputIterator_>(0);

	for (; _FirstUnwrapped != _LastUnwrapped; ++_FirstUnwrapped)
		if (_Predicate(*_FirstUnwrapped))
			++_Count;

	return _Count;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
