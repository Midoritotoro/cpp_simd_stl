#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <src/simd_stl/algorithm/vectorized/replace/ReplaceVectorized.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    class _UnwrappedForwardIterator_,
    class _Type_ = type_traits::IteratorValueType<_UnwrappedForwardIterator_>>
__simd_inline_constexpr void _ReplaceUnchecked(
    _UnwrappedForwardIterator_                          _FirstUnwrapped,
    _UnwrappedForwardIterator_                          _LastUnwrapped,
    const typename std::type_identity<_Type_>::type&    _OldValue,
    const typename std::type_identity<_Type_>::type&    _NewValue) noexcept
{
    if constexpr (type_traits::is_vectorized_find_algorithm_safe_v<_UnwrappedForwardIterator_, _Type_>) {
#if simd_stl_has_cxx20
        if (type_traits::is_constant_evaluated() == false)
#endif // simd_stl_has_cxx20
        {
            return _ReplaceVectorized(std::to_address(_FirstUnwrapped), 
                std::to_address(_LastUnwrapped), _OldValue, _NewValue);
        }
    }

    for (; _FirstUnwrapped != _LastUnwrapped; ++_FirstUnwrapped)
        if (*_FirstUnwrapped == _OldValue)
            *_FirstUnwrapped = _NewValue;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
