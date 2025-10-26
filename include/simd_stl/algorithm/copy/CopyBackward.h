#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <simd_stl/compatibility/Nodiscard.h>
#include <simd_stl/compatibility/Inline.h>

#include <src/simd_stl/algorithm/vectorized/CopyVectorized.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    class _BidirectionalFirstIterator_,
    class _BidirectionalSecondIterator_>
simd_stl_constexpr_cxx20 simd_stl_always_inline _BidirectionalSecondIterator_ copy_backward(
    _BidirectionalFirstIterator_    first,
    _BidirectionalFirstIterator_    last,
    _BidirectionalSecondIterator_   destinationLast) noexcept
{
    __verifyRange(first, last);

    const auto firstUnwrapped       = _UnwrapIterator(first);
    auto lastUnwrapped              = _UnwrapIterator(last);

    const auto difference           = IteratorsDifference(firstUnwrapped, lastUnwrapped);

    if constexpr (type_traits::IteratorCopyCategory<_BidirectionalFirstIterator_, _BidirectionalSecondIterator_>::BitcopyAssignable) {
        auto destinationLastUnwrapped = _UnwrapIteratorOffset(destinationLast, -difference);

        auto firstAddress           = std::to_address(firstUnwrapped);
        const auto lastAddress      = std::to_address(lastUnwrapped);

        const auto byteLength       = ByteLength(firstAddress, lastAddress);
        auto destinationLastAddress = std::to_address(destinationLastUnwrapped);

        auto destinationLastChar    = const_cast<char*>(reinterpret_cast<const volatile char*>(std::to_address(destinationLastUnwrapped)));

        AVX_memcpy(destinationLastChar - byteLength, firstAddress, byteLength);
        _SeekPossiblyWrappedIterator(destinationLast, destinationLast - difference);
    }
    else {
        auto destinationLastUnwrapped = _UnwrapIterator(destinationLast);

        while (firstUnwrapped != lastUnwrapped)
            *(--destinationLastUnwrapped) = *(--lastUnwrapped);

        _SeekPossiblyWrappedIterator(destinationLast, destinationLastUnwrapped);
    }

    return destinationLast;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END