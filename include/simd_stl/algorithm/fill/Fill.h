#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <simd_stl/compatibility/Nodiscard.h>
#include <simd_stl/compatibility/Inline.h>

#include <src/simd_stl/algorithm/vectorized/fill/FillVectorized.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>

#include <src/simd_stl/algorithm/AdvanceBytes.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    class _ForwardIterator_,
    class _Type_ = type_traits::iterator_value_type<_ForwardIterator_>>
__simd_inline_constexpr _ForwardIterator_ fill(
    _ForwardIterator_                                   first,
    _ForwardIterator_                                   last,
    const typename std::type_identity<_Type_>::type&    value) noexcept
{
    __verifyRange(first, last);

    auto firstUnwrapped         = _UnwrapIterator(first);
    const auto lastUnwrapped    = _UnwrapIterator(last);

    if constexpr (type_traits::is_vectorized_find_algorithm_safe_v<_ForwardIterator_, _Type_>) {
        const auto difference = __iterators_difference(firstUnwrapped, lastUnwrapped);
        _MemsetVectorized<_Type_>(std::to_address(firstUnwrapped), value, difference * sizeof(_Type_));
    }
    else
        for (; firstUnwrapped != lastUnwrapped; ++firstUnwrapped)
            *firstUnwrapped = value;

    return last;
}

template <
    class _ExecutionPolicy_,
    class _ForwardIterator_,
    class _Type_>
_ForwardIterator_ fill(
    _ExecutionPolicy_&&,
    _ForwardIterator_   first,
    _ForwardIterator_   last,
    const _Type_&       value) noexcept
{
    return simd_stl::algorithm::fill(first, last, value);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
