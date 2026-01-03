#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _UnwrappedInputIterator_,
	class _Predicate_>
__simd_nodiscard_inline_constexpr type_traits::IteratorDifferenceType<_UnwrappedInputIterator_> __count_if_unchecked(
	_UnwrappedInputIterator_	__first_unwrapped,
	_UnwrappedInputIterator_	__last_unwrapped,
	_Predicate_ 				__predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
		_Predicate_, type_traits::IteratorValueType<_UnwrappedInputIterator_>>)
{
	auto __count = type_traits::IteratorDifferenceType<_UnwrappedInputIterator_>(0);

	for (; __first_unwrapped != __last_unwrapped; ++__first_unwrapped)
		if (__predicate(*__first_unwrapped))
			++__count;

	return __count;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
