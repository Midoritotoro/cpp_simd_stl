#pragma once 

#include <src/simd_stl/algorithm/vectorized/minmax/MinVectorized.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>

#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _UnwrappedInputIterator_,
	class _BinaryPredicate_>
__simd_nodiscard_inline_constexpr type_traits::iterator_value_type<_UnwrappedInputIterator_> __min_unchecked(
	_UnwrappedInputIterator_	__first_unwrapped,
	_UnwrappedInputIterator_	__last_unwrapped,
	_BinaryPredicate_			__predicate) noexcept
{
	using _ValueType	= type_traits::iterator_value_type<_UnwrappedInputIterator_>;

	auto __minimum = *__first_unwrapped;

	while (++__first_unwrapped != __last_unwrapped)
		if (__predicate(*__first_unwrapped, __minimum))
			__minimum = *__first_unwrapped;

	return __minimum;
}

template <class _UnwrappedInputIterator_>
__simd_nodiscard_inline_constexpr type_traits::iterator_value_type<_UnwrappedInputIterator_> __min_unchecked(
	_UnwrappedInputIterator_ __first_unwrapped,
	_UnwrappedInputIterator_ __last_unwrapped) noexcept
{
	using _ValueType	= type_traits::iterator_value_type<_UnwrappedInputIterator_>;

	if constexpr (type_traits::is_vectorized_find_algorithm_safe_v<_UnwrappedInputIterator_, _ValueType>) {
#if simd_stl_has_cxx20
		if (type_traits::is_constant_evaluated() == false)
#endif // simd_stl_has_cxx20
		{
			return __min_vectorized<_ValueType>(std::to_address(__first_unwrapped), std::to_address(__last_unwrapped));
		}
	}

	return __min_unchecked(__first_unwrapped, __last_unwrapped, type_traits::less<>{});
}

__SIMD_STL_ALGORITHM_NAMESPACE_END