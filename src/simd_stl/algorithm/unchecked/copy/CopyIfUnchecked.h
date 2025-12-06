#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>
#include <simd_stl/concurrency/Execution.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    class _InputIterator_,
    class _OutputIterator_,
    class _Predicate_>
_Simd_inline_constexpr _OutputIterator_ _CopyIfUnchecked(
    _InputIterator_     _FirstUnwrapped,
    _InputIterator_     _LastUnwrapped,
    _OutputIterator_    _DestinationUnwrapped,
    _Predicate_         _Predicate) noexcept(
        type_traits::is_nothrow_invocable_v<
            _Predicate_, type_traits::IteratorValueType<_InputIterator_>>)
{
    for (; _FirstUnwrapped != _LastUnwrapped; ++_FirstUnwrapped)
        if (_Predicate(*_FirstUnwrapped))
            *_DestinationUnwrapped++ = *_FirstUnwrapped;

    return _DestinationUnwrapped;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
