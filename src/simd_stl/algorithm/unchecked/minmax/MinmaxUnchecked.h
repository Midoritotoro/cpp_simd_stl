#pragma once 

#include <src/simd_stl/algorithm/vectorized/minmax/MinmaxVectorized.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>

#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _UnwrappedInputIterator_>
using _Minmax_return_type = std::pair<
	type_traits::IteratorValueType<_UnwrappedInputIterator_>,
	type_traits::IteratorValueType<_UnwrappedInputIterator_>>;

template <
	class _UnwrappedInputIterator_,
	class _BinaryPredicate_>
_Simd_nodiscard_inline_constexpr _Minmax_return_type<_UnwrappedInputIterator_> _MinmaxUnchecked(
	_UnwrappedInputIterator_	_FirstUnwrapped,
	_UnwrappedInputIterator_	_LastUnwrapped,
	_BinaryPredicate_			_Predicate) noexcept
{
	using _ValueType	= type_traits::IteratorValueType<_UnwrappedInputIterator_>;

	auto _Minmax = std::pair<_ValueType, _ValueType>(*_FirstUnwrapped, *_FirstUnwrapped);

    for (; ++_FirstUnwrapped != _LastUnwrapped; ) {
        if (_Predicate(*_FirstUnwrapped, _Minmax.second))
			_Minmax.second = *_FirstUnwrapped;
		if (_Predicate(_Minmax.first, *_FirstUnwrapped))
			_Minmax.first = *_FirstUnwrapped;
    }

	return _Minmax;
}

template <class _UnwrappedInputIterator_>
_Simd_nodiscard_inline_constexpr _Minmax_return_type<_UnwrappedInputIterator_> _MinmaxUnchecked(
	_UnwrappedInputIterator_ _FirstUnwrapped,
	_UnwrappedInputIterator_ _LastUnwrapped) noexcept
{
	using _ValueType = type_traits::IteratorValueType<_UnwrappedInputIterator_>;

	if constexpr (type_traits::is_vectorized_find_algorithm_safe_v<_UnwrappedInputIterator_, _ValueType>) {
#if simd_stl_has_cxx20
		if (type_traits::is_constant_evaluated() == false)
#endif //simd_stl_has_cxx20
		{
			return _MinmaxVectorized<_ValueType>(std::to_address(_FirstUnwrapped), std::to_address(_LastUnwrapped));
		}
	}

	return _MinmaxUnchecked(_FirstUnwrapped, _LastUnwrapped, type_traits::greater<>{});
}

__SIMD_STL_ALGORITHM_NAMESPACE_END