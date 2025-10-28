#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <simd_stl/compatibility/Nodiscard.h>
#include <simd_stl/compatibility/Inline.h>

#include <src/simd_stl/algorithm/vectorized/FillVectorized.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    class _ForwardIterator_,
    class _ValueType_>
simd_stl_constexpr_cxx20 simd_stl_always_inline _ForwardIterator_ fill(
    _ForwardIterator_   first,
    _ForwardIterator_   last,
    const _ValueType_&  value) noexcept
{
    __verifyRange(first, last);

    auto firstUnwrapped         = _UnwrapIterator(first);
    const auto lastUnwrapped    = _UnwrapIterator(last);

    const auto difference       = IteratorsDifference(firstUnwrapped, lastUnwrapped);

    if constexpr (type_traits::is_vectorized_find_algorithm_safe_v<_ForwardIterator_, _ForwardIterator_>) {
        auto destinationUnwrapped = _UnwrapIteratorOffset(destination, difference);

        auto firstAddress       = std::to_address(firstUnwrapped);
        const auto lastAddress  = std::to_address(lastUnwrapped);

        const auto byteLength   = ByteLength(firstAddress, lastAddress);
        auto destinationAddress = std::to_address(destinationUnwrapped);

        _MemsetVectorized(destinationAddress, value, byteLength);
        _SeekPossiblyWrappedIterator(destination, destination + difference);
    }
    else {
        auto destinationUnwrapped = _UnwrapIterator(destination);

        for (; firstUnwrapped != lastUnwrapped; ++destinationUnwrapped, ++firstUnwrapped)
            *destinationUnwrapped = *firstUnwrapped;

        _SeekPossiblyWrappedIterator(destination, destinationUnwrapped);
    }

    return destination;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
