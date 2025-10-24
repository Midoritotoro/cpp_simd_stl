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
    using _ValueType_ = type_traits::IteratorValueType<_InputIterator_>;
    __verifyRange(first, last);

    auto firstUnwrapped         = __unwrapIterator(first);
    const auto lastUnwrapped    = __unwrapIterator(last);

    const auto difference       = IteratorsDifference(firstUnwrapped, lastUnwrapped);
    auto destinationUnwrapped   = __unwrapSizedIterator(destination, difference);

    const void* result = CopyVectorized(
        std::to_address(firstUnwrapped), 
        std::to_address(destinationUnwrapped), sizeof(_ValueType_) * difference);

    __seekWrappedIterator(destination, reinterpret_cast<const _ValueType_*>(result));
    return destination;
   // return copy_if(first, last, destination, type_traits::...<>{});
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