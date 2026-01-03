#pragma once 


#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <src/simd_stl/algorithm/vectorized/replace/ReplaceCopyVectorized.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    class _UnwrappedInputIterator_,
    class _UnwrappedOutputIterator_,
    class _UnaryPredicate_,
    class _Type_ = type_traits::IteratorValueType<_UnwrappedInputIterator_>>
__simd_inline_constexpr simd_stl_always_inline void _ReplaceCopyIfUnchecked(
    _UnwrappedInputIterator_                            _FirstUnwrapped,
    _UnwrappedInputIterator_                            _LastUnwrapped,
    _UnwrappedOutputIterator_                           _DestinationUnwrapped,
    _UnaryPredicate_                                    _Predicate,
    const typename std::type_identity<_Type_>::type&    _NewValue) noexcept(
        type_traits::is_nothrow_invocable_v<_UnaryPredicate_, _Type_>)
{
    for (; _FirstUnwrapped != _LastUnwrapped; ++_FirstUnwrapped, ++_DestinationUnwrapped)
        *_DestinationUnwrapped = _Predicate(*_FirstUnwrapped) ? _NewValue : *_FirstUnwrapped;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
