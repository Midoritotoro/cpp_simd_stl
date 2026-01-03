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
    class _SizeType_,
    class _Type_ = type_traits::IteratorValueType<_ForwardIterator_>>
__simd_inline_constexpr _ForwardIterator_ fill_n(
    _ForwardIterator_                                   first,
    _SizeType_                                          count,
    const typename std::type_identity<_Type_>::type&    value) noexcept
{
    auto firstUnwrapped         = _UnwrapIteratorOffset(first, count);

    if constexpr (type_traits::is_vectorized_find_algorithm_safe_v<_ForwardIterator_, _Type_>)
        _MemsetVectorized<_Type_>(std::to_address(firstUnwrapped), value, count * sizeof(_Type_));
    else
        for (_SizeType_ current = 0; current < count; ++current, ++firstUnwrapped)
            *firstUnwrapped = value;

    __seek_possibly_wrapped_iterator(first, first + count);
    return first;
}

template <
    class _ExecutionPolicy_,
    class _ForwardIterator_,
    class _SizeType_,
    class _Type_>
__simd_inline_constexpr _ForwardIterator_ fill_n(
    _ExecutionPolicy_&&,
    _ForwardIterator_   first,
    _SizeType_          count,
    const _Type_&       value) noexcept
{
    return simd_stl::algorithm::fill_n(first, count, value);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
