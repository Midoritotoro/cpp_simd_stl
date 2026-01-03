#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _UnwrappedIterator_,
	class _UnaryPredicate_>
__simd_nodiscard_inline_constexpr _UnwrappedIterator_ _RemoveIfUnchecked(
	_UnwrappedIterator_			_FirstUnwrapped,
	_UnwrappedIterator_			_LastUnwrapped,
	_UnaryPredicate_			_Predicate) noexcept(
		type_traits::is_nothrow_invocable_v<_UnaryPredicate_,
			type_traits::IteratorValueType<_UnwrappedIterator_>>)
{
	for (; _FirstUnwrapped != _LastUnwrapped; ++_FirstUnwrapped)
		if (_Predicate(*_FirstUnwrapped))
			break;

	if (_FirstUnwrapped == _LastUnwrapped)
		return _LastUnwrapped;

	for (auto _Current = _FirstUnwrapped; ++_Current != _LastUnwrapped;) {
		const auto _CurrentValue = std::move(*_Current);

		if (_Predicate(_CurrentValue) == false)
			*_FirstUnwrapped++ = std::move(_CurrentValue);
	}

	return _FirstUnwrapped;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
