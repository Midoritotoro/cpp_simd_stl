#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _UnwrappedIterator_,
	class _UnaryPredicate_>
__simd_nodiscard_inline_constexpr _UnwrappedIterator_ __remove_if_unchecked(
	_UnwrappedIterator_			__first_unwrapped,
	_UnwrappedIterator_			__last_unwrapped,
	_UnaryPredicate_			__predicate) noexcept(
		type_traits::is_nothrow_invocable_v<_UnaryPredicate_,
			type_traits::iterator_value_type<_UnwrappedIterator_>>)
{
	for (; __first_unwrapped != __last_unwrapped; ++__first_unwrapped)
		if (__predicate(*__first_unwrapped))
			break;

	if (__first_unwrapped == __last_unwrapped)
		return __last_unwrapped;

	for (auto __current = __first_unwrapped; ++__current != __last_unwrapped;) {
		const auto __current_value = std::move(*__current);

		if (__predicate(__current_value) == false)
			*__first_unwrapped++ = std::move(__current_value);
	}

	return __first_unwrapped;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
