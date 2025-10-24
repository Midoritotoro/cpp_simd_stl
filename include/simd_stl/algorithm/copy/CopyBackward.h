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

    const auto firstUnwrapped       = __unwrapIterator(first);
    auto lastUnwrapped              = __unwrapIterator(last);

    const auto difference           = IteratorsDifference(firstUnwrapped, lastUnwrapped);

    if constexpr (type_traits::IteratorCopyCategory<_BidirectionalFirstIterator_, _BidirectionalSecondIterator_>::BitcopyAssignable) {
        using _BidirectionalFirstUnwrappedIterator_ = type_traits::_Unwrapped_iterator_type<_BidirectionalFirstIterator_>;

        _BidirectionalFirstUnwrappedIterator_ destinationLastUnwrapped;

        if constexpr (std::_Unwrappable_for_offset_v<_BidirectionalFirstIterator_>)
            destinationLastUnwrapped = __unwrapSizedIterator(destinationLast, difference);
        else 
            destinationLastUnwrapped = __unwrapIterator(destinationLast);

        auto firstAddress           = std::to_address(firstUnwrapped);
        const auto lastAddress      = std::to_address(lastUnwrapped);

        const auto byteLength       = ByteLength(firstAddress, lastAddress);
        auto destinationLastAddress = std::to_address(destinationLastUnwrapped);

        auto destinationLastChar    = const_cast<char*>(reinterpret_cast<const volatile char*>(std::to_address(destinationLastUnwrapped)));

        CopyVectorized(firstAddress, destinationLastChar - byteLength, byteLength);
        __seekWrappedIterator(destinationLast, destinationLast + difference);
    }
    else {
        auto destinationLastUnwrapped = __unwrapIterator(destinationLast);

        while (firstUnwrapped != lastUnwrapped)
            *(--destinationLastUnwrapped) = *(--lastUnwrapped);

        __seekWrappedIterator(destinationLast, destinationLastUnwrapped);
    }

    return destinationLast;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END