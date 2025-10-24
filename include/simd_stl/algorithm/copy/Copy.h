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
simd_stl_constexpr_cxx20 simd_stl_always_inline _OutputIterator_ copy(
    _InputIterator_     first,
    _InputIterator_     last,
    _OutputIterator_    destination) noexcept
{
    __verifyRange(first, last);

    auto firstUnwrapped         = __unwrapIterator(first);
    const auto lastUnwrapped    = __unwrapIterator(last);

    const auto difference       = IteratorsDifference(firstUnwrapped, lastUnwrapped);

    if constexpr (type_traits::IteratorCopyCategory<_InputIterator_, _OutputIterator_>::BitcopyAssignable) {
        auto destinationUnwrapped = __unwrapSizedIterator(destination, difference);

        auto firstAddress       = std::to_address(firstUnwrapped);
        const auto lastAddress  = std::to_address(lastUnwrapped);

        const auto byteLength   = ByteLength(firstAddress, lastAddress);
        auto destinationAddress = std::to_address(destinationUnwrapped);

        CopyVectorized(firstAddress, destinationAddress, byteLength);
        __seekWrappedIterator(destination, destination + difference);
    }
    else {
        auto destinationUnwrapped = __unwrapIterator(destination);

        for (; firstUnwrapped != lastUnwrapped; ++destinationUnwrapped, ++firstUnwrapped)
            *destinationUnwrapped = *firstUnwrapped;


        __seekWrappedIterator(destination, destinationUnwrapped);
    }

    return destination;
}

template <
    class _InputIterator_,
    class _OutputIterator_,
    class _Predicate_>
simd_stl_constexpr_cxx20 simd_stl_always_inline _OutputIterator_ copy_if(
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

    auto firstUnwrapped         = __unwrapIterator(first);
    const auto lastUnwrapped    = __unwrapIterator(last);

    auto destinationUnwrapped   = __unwrapUnverifiedIterator(destination);

    for (; firstUnwrapped != lastUnwrapped; ++firstUnwrapped) {
        if (predicate(*firstUnwrapped)) {
            *destinationUnwrapped = *firstUnwrapped;
            ++destinationUnwrapped;
        }
    }

    __seekWrappedIterator(destination, destinationUnwrapped);
    return destination;
}


__SIMD_STL_ALGORITHM_NAMESPACE_END