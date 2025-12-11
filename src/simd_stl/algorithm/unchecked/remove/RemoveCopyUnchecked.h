#pragma once 

#include <src/simd_stl/algorithm/vectorized/remove/RemoveCopyVectorized.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _UnwrappedInputIterator_,
	class _UnwrappedOutputIterator_,
	class _Type_ = type_traits::IteratorValueType<_UnwrappedInputIterator_>>
_Simd_nodiscard_inline_constexpr _UnwrappedOutputIterator_ _RemoveCopyUnchecked(
	_UnwrappedInputIterator_							_FirstUnwrapped,
	_UnwrappedInputIterator_							_LastUnwrapped,
	_UnwrappedOutputIterator_							_DestinationUnwrapped,
	const typename std::type_identity<_Type_>::type&	_Value) noexcept
{
	constexpr auto _Is_vectorizable = type_traits::is_vectorized_find_algorithm_safe_v<_UnwrappedInputIterator_, _Type_> &&
		type_traits::is_vectorized_find_algorithm_safe_v<_UnwrappedOutputIterator_, _Type_> &&
		type_traits::IteratorCopyCategory<_UnwrappedInputIterator_, _UnwrappedOutputIterator_>::BitcopyAssignable;

	using _DifferenceType = type_traits::IteratorDifferenceType<_UnwrappedOutputIterator_>;

	if constexpr (_Is_vectorizable)
	{
#if simd_stl_has_cxx20
		if (type_traits::is_constant_evaluated() == false)
#endif // simd_stl_has_cxx20
		{
			if (math::couldCompareEqualToValueType<_UnwrappedInputIterator_>(_Value) == false)
				return _DestinationUnwrapped;

			const auto _DestinationAddress = std::to_address(_DestinationUnwrapped);
			auto _Position = _RemoveCopyVectorized<_Type_>(std::to_address(_FirstUnwrapped),
				std::to_address(_LastUnwrapped), _DestinationAddress, _Value);

			if constexpr (std::is_pointer_v<_UnwrappedOutputIterator_>)
				return _Position;
			else
				return _DestinationUnwrapped + static_cast<_DifferenceType>(_Position - _DestinationAddress);
		}
	}

	for (; _FirstUnwrapped != _LastUnwrapped; ++_FirstUnwrapped) {
		const auto _FirstValue = std::move(*_FirstUnwrapped);

		if (_FirstValue != _Value)
			*_DestinationUnwrapped++ = std::move(_FirstValue);
	}

	return _DestinationUnwrapped;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
