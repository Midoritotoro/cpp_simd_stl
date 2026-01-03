#pragma once 

#include <src/simd_stl/algorithm/vectorized/minmax/MinVectorized.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>

#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _UnwrappedInputIterator_,
	class _BinaryPredicate_>
__simd_nodiscard_inline_constexpr type_traits::IteratorValueType<_UnwrappedInputIterator_> _MinUnchecked(
	_UnwrappedInputIterator_	_FirstUnwrapped,
	_UnwrappedInputIterator_	_LastUnwrapped,
	_BinaryPredicate_			_Predicate) noexcept
{
	using _ValueType	= type_traits::IteratorValueType<_UnwrappedInputIterator_>;

	auto _Minimum = *_FirstUnwrapped;

	while (++_FirstUnwrapped != _LastUnwrapped)
		if (_Predicate(*_FirstUnwrapped, _Minimum))
			_Minimum = *_FirstUnwrapped;

	return _Minimum;
}

template <class _UnwrappedInputIterator_>
__simd_nodiscard_inline_constexpr type_traits::IteratorValueType<_UnwrappedInputIterator_> _MinUnchecked(
	_UnwrappedInputIterator_ _FirstUnwrapped,
	_UnwrappedInputIterator_ _LastUnwrapped) noexcept
{
	using _ValueType	= type_traits::IteratorValueType<_UnwrappedInputIterator_>;

	if constexpr (type_traits::is_vectorized_find_algorithm_safe_v<_UnwrappedInputIterator_, _ValueType>) {
#if simd_stl_has_cxx20
		if (type_traits::is_constant_evaluated() == false)
#endif //simd_stl_has_cxx20
		{
			return _MinVectorized<_ValueType>(std::to_address(_FirstUnwrapped), std::to_address(_LastUnwrapped));
		}
	}

	return _MinUnchecked(_FirstUnwrapped, _LastUnwrapped, type_traits::less<>{});
}

__SIMD_STL_ALGORITHM_NAMESPACE_END