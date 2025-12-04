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
_Simd_inline_constexpr _OutputIterator_ copy(
    _InputIterator_     first,
    _InputIterator_     last,
    _OutputIterator_    destination) noexcept
{
    __verifyRange(first, last);

    auto firstUnwrapped         = _UnwrapIterator(first);
    const auto lastUnwrapped    = _UnwrapIterator(last);

    const auto difference       = IteratorsDifference(firstUnwrapped, lastUnwrapped);

    if constexpr (type_traits::IteratorCopyCategory<_InputIterator_, _OutputIterator_>::BitcopyAssignable) {
        auto destinationUnwrapped = _UnwrapIteratorOffset(destination, difference);

        auto firstAddress       = std::to_address(firstUnwrapped);
        const auto lastAddress  = std::to_address(lastUnwrapped);

        const auto byteLength   = ByteLength(firstAddress, lastAddress);
        auto destinationAddress = std::to_address(destinationUnwrapped);

        _MemcpyVectorized(destinationAddress, firstAddress, byteLength);
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

template <
    class _InputIterator_,
    class _OutputIterator_,
    class _Predicate_>
_Simd_inline_constexpr _OutputIterator_ copy_if(
    _InputIterator_     first,
    _InputIterator_     last,
    _OutputIterator_    destination,
    _Predicate_         predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_,
			type_traits::IteratorValueType<_InputIterator_>
		>)
{
    __verifyRange(first, last);

    auto firstUnwrapped         = _UnwrapIterator(first);
    const auto lastUnwrapped    = _UnwrapIterator(last);

    auto destinationUnwrapped   = _UnwrapUnverifiedIterator(destination);

    for (; firstUnwrapped != lastUnwrapped; ++firstUnwrapped) {
        if (predicate(*firstUnwrapped)) {
            *destinationUnwrapped = *firstUnwrapped;
            ++destinationUnwrapped;
        }
    }

    _SeekPossiblyWrappedIterator(destination, destinationUnwrapped);
    return destination;
}

template <
    class _ExecutionPolicy_,
    class _InputIterator_,
    class _OutputIterator_>
_OutputIterator_ copy(
    _ExecutionPolicy_&&,
    _InputIterator_     first,
    _InputIterator_     last,
    _OutputIterator_    destination) noexcept
{
    return simd_stl::algorithm::copy(first, last, destination);
}

template <
    class _ExecutionPolicy_,
    class _InputIterator_,
    class _OutputIterator_,
    class _Predicate_>
_OutputIterator_ copy_if(
    _ExecutionPolicy_&&,
    _InputIterator_     first,
    _InputIterator_     last,
    _OutputIterator_    destination,
    _Predicate_         predicate) noexcept(
        type_traits::is_nothrow_invocable_v<
        _Predicate_,
        type_traits::IteratorValueType<_InputIterator_>
        >)
{
    return simd_stl::algorithm::copy_if(first, last, destination, type_traits::passFunction(predicate));
}


__SIMD_STL_ALGORITHM_NAMESPACE_END