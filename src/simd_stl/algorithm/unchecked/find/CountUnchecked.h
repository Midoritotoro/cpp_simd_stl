#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <src/simd_stl/algorithm/vectorized/find/CountVectorized.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
	class _UnwrappedIterator_,
	class _Type_ = type_traits::IteratorValueType<_UnwrappedIterator_>>
_Simd_nodiscard_inline_constexpr sizetype _CountUnchecked(
	_UnwrappedIterator_									_FirstUnwrapped,
	_UnwrappedIterator_									_LastUnwrapped,
	const typename std::type_identity<_Type_>::type&	_Value) noexcept
{
	if constexpr (type_traits::is_iterator_random_ranges_v<_UnwrappedIterator_>) {
		const auto _Size = ByteLength(_FirstUnwrapped, _LastUnwrapped);

		if constexpr (type_traits::is_vectorized_find_algorithm_safe_v<_UnwrappedIterator_, _Type_>) {
#if simd_stl_has_cxx20
			if (type_traits::is_constant_evaluated() == false)
#endif // simd_stl_has_cxx20
			{
				return _CountVectorized(std::to_address(_FirstUnwrapped), _Size, _Value);
			}
		}
	}

    sizetype _Count = 0;

	for (; _FirstUnwrapped != _LastUnwrapped; ++_FirstUnwrapped)
		if (*_FirstUnwrapped == _Value)
			++_Count;

	return _Count;
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
