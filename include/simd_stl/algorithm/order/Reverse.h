#pragma once 

#include <src/simd_stl/algorithm/vectorized/ReverseVectorized.h>
#include <simd_stl/algorithm/swap/Swap.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _BidirectionalIterator_>
simd_stl_nodiscard simd_stl_constexpr_cxx20 simd_stl_always_inline void reverse(
    _BidirectionalIterator_ first,
    _BidirectionalIterator_ last) noexcept
{
    __verifyRange(first, last);

    using _BidirectionalUnwrappedIterator_ = unwrapped_iterator_type<_BidirectionalIterator_>;

    auto firstUnwrapped = _UnwrapIterator(first);
    auto lastUnwrapped  = _UnwrapIterator(last);

    if constexpr (type_traits::is_iterator_random_ranges_v<_BidirectionalUnwrappedIterator_>) {
#if simd_stl_has_cxx20
        if (type_traits::is_constant_evaluated() == false)
#endif
        {
            _ReverseVectorized<type_traits::IteratorValueType<_BidirectionalUnwrappedIterator_>>(
                std::to_address(firstUnwrapped), std::to_address(lastUnwrapped));
        }
    }

    for (; firstUnwrapped != lastUnwrapped && firstUnwrapped != --lastUnwrapped; ++firstUnwrapped)
        simd_stl::algorithm::iter_swap(firstUnwrapped, lastUnwrapped);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
