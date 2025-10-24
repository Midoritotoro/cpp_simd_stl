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
    class _SizeType_,
    class _OutputIterator_>
simd_stl_constexpr_cxx20 simd_stl_always_inline _OutputIterator_ copy_n(
    _InputIterator_     first,
    _SizeType_          elementsCount,
    _OutputIterator_    destination) noexcept
{
    using _ValueType_ = type_traits::IteratorValueType<_InputIterator_>;

    __verifyRange(first, last);

    auto firstUnwrapped         = __unwrapIterator(first);
    const auto bytes = sizeof(_ValueType_) * elementsCount;

    if constexpr (type_traits::IteratorCopyCategory<_InputIterator_, _OutputIterator_>::BitcopyAssignable) {
        auto destinationUnwrapped = __unwrapSizedIterator(destination, elementsCount);

        auto firstAddress       = std::to_address(firstUnwrapped);
        auto destinationAddress = std::to_address(destinationUnwrapped);

        CopyVectorized(firstAddress, destinationAddress, bytes);
        __seekWrappedIterator(destination, destination + elementsCount);
    }
    else {
        auto destinationUnwrapped = __unwrapIterator(destination);

        for (_SizeType_ current = 0; current < elementsCount; ++destinationUnwrapped, ++firstUnwrapped)
            *destinationUnwrapped = *firstUnwrapped;

        __seekWrappedIterator(destination, destinationUnwrapped);
    }

    return destination;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END