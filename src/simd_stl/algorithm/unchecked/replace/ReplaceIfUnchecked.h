#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <src/simd_stl/algorithm/vectorized/replace/ReplaceVectorized.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    class _UnwrappedForwardIterator_,
    class _UnaryPredicate_,
    class _Type_ = type_traits::IteratorValueType<_UnwrappedForwardIterator_>>
__simd_inline_constexpr void _ReplaceIfUnchecked(
    _UnwrappedForwardIterator_                          _FirstUnwrapped,
    _UnwrappedForwardIterator_                          _LastUnwrapped,
    _UnaryPredicate_                                    _Predicate,
    const typename std::type_identity<_Type_>::type&    _NewValue) noexcept(
        type_traits::is_nothrow_invocable_v<_UnaryPredicate_, _Type_>)
{
    for (; _FirstUnwrapped != _LastUnwrapped; ++_FirstUnwrapped)
        if (_Predicate(*_FirstUnwrapped))
            *_FirstUnwrapped = _NewValue;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
