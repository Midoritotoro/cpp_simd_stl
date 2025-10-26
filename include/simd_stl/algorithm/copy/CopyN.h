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

    auto firstUnwrapped         = _UnwrapIterator(first);
    const auto bytes = sizeof(_ValueType_) * elementsCount;

    if constexpr (type_traits::IteratorCopyCategory<_InputIterator_, _OutputIterator_>::BitcopyAssignable) {
        auto destinationUnwrapped = _UnwrapIteratorOffset(destination, elementsCount);

        auto firstAddress       = std::to_address(firstUnwrapped);
        auto destinationAddress = std::to_address(destinationUnwrapped);

        _MemcpyVectorized(destinationAddress, firstAddress, bytes);
        _SeekPossiblyWrappedIterator(destination, destination + elementsCount);
    }
    else {
        auto destinationUnwrapped = _UnwrapIterator(destination);

        for (_SizeType_ current = 0; current < elementsCount; ++destinationUnwrapped, ++firstUnwrapped)
            *destinationUnwrapped = *firstUnwrapped;

        _SeekPossiblyWrappedIterator(destination, destinationUnwrapped);
    }

    return destination;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END