#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <src/simd_stl/algorithm/vectorized/remove/RemoveVectorized.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _UnwrappedInputIterator_,
	class _Type_ = type_traits::IteratorValueType<_UnwrappedInputIterator_>>
simd_stl_nodiscard simd_stl_always_inline simd_stl_constexpr_cxx20 _UnwrappedInputIterator_ _RemoveUnchecked(
	_UnwrappedInputIterator_							_FirstUnwrapped,
	_UnwrappedInputIterator_							_LastUnwrapped,
	const typename std::type_identity<_Type_>::type&	_Value) noexcept
{
	using _DifferenceType = type_traits::IteratorDifferenceType<_UnwrappedInputIterator_>;

	if constexpr (type_traits::is_vectorized_find_algorithm_safe_v<_UnwrappedInputIterator_, _Type_>) {
#if simd_stl_has_cxx20
		if (type_traits::is_constant_evaluated() == false)
#endif // simd_stl_has_cxx20
		{
			if (math::couldCompareEqualToValueType<_UnwrappedInputIterator_>(_Value) == false)
				return _LastUnwrapped;

			const auto _FirstAddress = std::to_address(_FirstUnwrapped);
			const auto _Position = _RemoveVectorized<
				type_traits::IteratorValueType<_UnwrappedInputIterator_>>(_FirstAddress, std::to_address(_LastUnwrapped), _Value);

			if constexpr (std::is_pointer_v<_UnwrappedInputIterator_>)
				return _Position;
			else
				return _FirstUnwrapped + static_cast<_DifferenceType>(_Position - _FirstAddress);
		}
	}

	for (; _FirstUnwrapped != _LastUnwrapped; ++_FirstUnwrapped)
		if (*_FirstUnwrapped == _Value)
			break;

	for (auto _Current = _FirstUnwrapped; ++_Current != _LastUnwrapped;) {
		const auto _CurrentValue = std::move(*_Current);

		if (_CurrentValue != _Value)
			*_FirstUnwrapped++ = std::move(_CurrentValue);
	}

	return _FirstUnwrapped;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
