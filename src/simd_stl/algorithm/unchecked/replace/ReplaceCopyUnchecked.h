#pragma once 


#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <src/simd_stl/algorithm/vectorized/replace/ReplaceCopyVectorized.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    class _UnwrappedInputIterator_,
    class _UnwrappedOutputIterator_,
    class _Type_ = type_traits::IteratorValueType<_UnwrappedInputIterator_>>
_Simd_inline_constexpr void _ReplaceCopyUnchecked(
    _UnwrappedInputIterator_                            _FirstUnwrapped,
    _UnwrappedInputIterator_                            _LastUnwrapped,
    _UnwrappedOutputIterator_                           _DestinationUnwrapped,
    const typename std::type_identity<_Type_>::type&    _OldValue,
    const typename std::type_identity<_Type_>::type&    _NewValue) noexcept
{
    if constexpr (type_traits::is_vectorized_find_algorithm_safe_v<_UnwrappedInputIterator_, _Type_>
        && type_traits::is_vectorized_find_algorithm_safe_v<_UnwrappedOutputIterator_, _Type_>) 
    {
#if simd_stl_has_cxx20
        if (type_traits::is_constant_evaluated() == false)
#endif // simd_stl_has_cxx20
        {
            return _ReplaceCopyVectorized(std::to_address(_FirstUnwrapped), 
                std::to_address(_LastUnwrapped), std::to_address(_DestinationUnwrapped), _OldValue, _NewValue);
        }
    }

    for (; _FirstUnwrapped != _LastUnwrapped; ++_FirstUnwrapped, ++_DestinationUnwrapped)
        *_DestinationUnwrapped = (*_FirstUnwrapped == _OldValue) ? _NewValue : *_FirstUnwrapped;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
