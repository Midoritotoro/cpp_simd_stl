#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <simd_stl/compatibility/Nodiscard.h>
#include <simd_stl/compatibility/Inline.h>

#include <src/simd_stl/algorithm/vectorized/CopyVectorized.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    class _InputIterator_,
    class _OutputIterator_>
_Simd_inline_constexpr _OutputIterator_ move(
    _InputIterator_     first,
    _InputIterator_     last,
    _OutputIterator_    destination) noexcept
{
    using _InputIteratorUnwrapped_  = unwrapped_iterator_type<_InputIterator_>;
    using _OutputIteratorUnwrapped_ = unwrapped_iterator_type<_OutputIterator_>;

    __verifyRange(first, last);

    auto firstUnwrapped = _UnwrapIterator(first);
    const auto lastUnwrapped = _UnwrapIterator(last);

    const auto difference = IteratorsDifference(firstUnwrapped, lastUnwrapped);

    if constexpr (type_traits::IteratorCopyCategory<_InputIteratorUnwrapped_, _OutputIteratorUnwrapped_>::BitcopyAssignable) {
        auto destinationUnwrapped = _UnwrapIteratorOffset(destination, difference);

        auto firstAddress = std::to_address(firstUnwrapped);
        const auto lastAddress = std::to_address(lastUnwrapped);

        const auto byteLength = ByteLength(firstAddress, lastAddress);
        auto destinationAddress = std::to_address(destinationUnwrapped);

        _MemmoveVectorized(destinationAddress, firstAddress, byteLength);
        _SeekPossiblyWrappedIterator(destination, destination + difference);
    }
    else {
        auto destinationUnwrapped = _UnwrapIterator(destination);

        for (; firstUnwrapped != lastUnwrapped; ++destinationUnwrapped, ++firstUnwrapped)
            *destinationUnwrapped = std::move(*firstUnwrapped);

        _SeekPossiblyWrappedIterator(destination, destinationUnwrapped);
    }

    return destination;
}

template <
    class _ExecutionPolicy_,
    class _InputIterator_,
    class _OutputIterator_>
_OutputIterator_ move(
    _ExecutionPolicy_&&,
    _InputIterator_     first,
    _InputIterator_     last,
    _OutputIterator_    destination) noexcept
{
    return simd_stl::algorithm::move(first, last, destination);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END