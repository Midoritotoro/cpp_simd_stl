#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <src/simd_stl/algorithm/vectorized/find/FindVectorized.h>

#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>
#include <simd_stl/concurrency/Execution.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _UnwrappedIterator_,
	class _Type_>
_Simd_nodiscard_inline_constexpr _UnwrappedIterator_ _FindUnchecked(
	_UnwrappedIterator_	_FirstUnwrapped,
	_UnwrappedIterator_	_LastUnwrapped,
	const _Type_&		_Value) noexcept
{
	using _DifferenceType = type_traits::IteratorDifferenceType<_UnwrappedIterator_>;

	if constexpr (type_traits::is_vectorized_find_algorithm_safe_v<_UnwrappedIterator_, _Type_>) {
#if simd_stl_has_cxx20
		if (type_traits::is_constant_evaluated() == false)
#endif // simd_stl_has_cxx20
		{
			if (math::couldCompareEqualToValueType<_UnwrappedIterator_>(_Value) == false)
				return _LastUnwrapped;

			const auto _FirstAddress = std::to_address(_FirstUnwrapped);
			const auto _Position = _FindVectorized(_FirstAddress, std::to_address(_LastUnwrapped), _Value);

			if constexpr (std::is_pointer_v<_UnwrappedIterator_>)
				return _Position;
			else
				return _FirstUnwrapped + static_cast<_DifferenceType>(_Position - _FirstAddress);
		}
	}

	for (; _FirstUnwrapped != _LastUnwrapped; ++_FirstUnwrapped)
		if (*_FirstUnwrapped == _Value)
			break;

	return _FirstUnwrapped;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END