#pragma once 

#include <src/simd_stl/algorithm/vectorized/minmax/MaxVectorized.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>

#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _UnwrappedInputIterator_,
	class _BinaryPredicate_>
_Simd_nodiscard_inline_constexpr std::optional<type_traits::IteratorValueType<_UnwrappedInputIterator_>> _MaxUnchecked(
	_UnwrappedInputIterator_	_FirstUnwrapped,
	_UnwrappedInputIterator_	_LastUnwrapped,
	_BinaryPredicate_			_Predicate) noexcept
{
	using _ValueType	= type_traits::IteratorValueType<_UnwrappedInputIterator_>;
	using _Optional		= std::optional<type_traits::IteratorValueType<_UnwrappedInputIterator_>>;

	if (_FirstUnwrapped == _LastUnwrapped)
		return _Optional();

	auto _Maximum = *_FirstUnwrapped;

	while (++_FirstUnwrapped != _LastUnwrapped)
		if (_Predicate(*_FirstUnwrapped, _Maximum))
			_Maximum = *_FirstUnwrapped;

	return _Maximum;
}

template <class _UnwrappedInputIterator_>
_Simd_nodiscard_inline_constexpr std::optional<type_traits::IteratorValueType<_UnwrappedInputIterator_>> _MaxUnchecked(
	_UnwrappedInputIterator_ _FirstUnwrapped,
	_UnwrappedInputIterator_ _LastUnwrapped) noexcept
{
	using _ValueType = type_traits::IteratorValueType<_UnwrappedInputIterator_>;
	using _Optional  = std::optional<type_traits::IteratorValueType<_UnwrappedInputIterator_>>;

	if (_FirstUnwrapped == _LastUnwrapped)
		return _Optional();

	if constexpr (type_traits::is_vectorized_find_algorithm_safe_v<_UnwrappedInputIterator_, _ValueType>) {
#if simd_stl_has_cxx20
		if (type_traits::is_constant_evaluated() == false)
#endif //simd_stl_has_cxx20
		{
			return _MaxVectorized<_ValueType>(std::to_address(_FirstUnwrapped), std::to_address(_LastUnwrapped));
		}
	}

#if defined(simd_stl_cpp_msvc)
	__assume(_FirstUnwrapped != _LastUnwrapped);
#endif

	return _MaxUnchecked(_FirstUnwrapped, _LastUnwrapped, type_traits::greater<>{});
}

__SIMD_STL_ALGORITHM_NAMESPACE_END