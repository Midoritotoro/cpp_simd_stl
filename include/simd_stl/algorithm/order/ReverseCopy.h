#pragma once 

#include <src/simd_stl/algorithm/vectorized/ReverseCopyVectorized.h>

#include <simd_stl/algorithm/swap/Swap.h>
#include <simd_stl/memory/Intersects.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    class _FirstBidirectionalIterator_,
    class _SecondBidirectionalIterator_>
simd_stl_nodiscard simd_stl_constexpr_cxx20 simd_stl_always_inline void reverse_copy(
    _FirstBidirectionalIterator_    first,
    _FirstBidirectionalIterator_    last,
    _SecondBidirectionalIterator_   destination) noexcept
{
    using _FirstBidirectionalUnwrappedIterator_     = unwrapped_iterator_type<_FirstBidirectionalIterator_>;
    using _SecondBidirectionalUnwrappedIterator_    = unwrapped_iterator_type<_SecondBidirectionalIterator_>;

    using _FirstBidirectionalIteratorValueType_     = type_traits::iterator_value_type<_FirstBidirectionalUnwrappedIterator_>;

    __verifyRange(first, last);
    memory::_CheckIntersection(first, last, destination);

    auto firstUnwrapped             = _UnwrapIterator(first);
    auto lastUnwrapped              = _UnwrapIterator(last);

    auto destinationUnwrapped       = _UnwrapIterator(destination);

    if constexpr (
        type_traits::is_iterator_contiguous_v<_FirstBidirectionalUnwrappedIterator_> &&
        type_traits::is_iterator_contiguous_v<_SecondBidirectionalUnwrappedIterator_> &&
        type_traits::__is_vector_type_supported_v<_FirstBidirectionalIteratorValueType_>)
    {
#if simd_stl_has_cxx20
        if (type_traits::is_constant_evaluated() == false)
#endif
        {
            return _ReverseCopyVectorized<_FirstBidirectionalIteratorValueType_>(
                std::to_address(firstUnwrapped), std::to_address(lastUnwrapped), std::to_address(destinationUnwrapped));
        }
    }

    for (; firstUnwrapped != lastUnwrapped; ++destinationUnwrapped)
        *destinationUnwrapped = std::move(*--lastUnwrapped);
}

template <
    class _ExecutionPolicy_,
    class _FirstBidirectionalIterator_,
    class _SecondBidirectionalIterator_>
simd_stl_nodiscard simd_stl_constexpr_cxx20 simd_stl_always_inline void reverse_copy(
    _ExecutionPolicy_&&,
    _FirstBidirectionalIterator_    first,
    _FirstBidirectionalIterator_    last,
    _SecondBidirectionalIterator_   destination) noexcept
{
    return simd_stl::algorithm::reverse_copy(first, last, destination);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
