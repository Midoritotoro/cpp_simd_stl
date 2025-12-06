#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <simd_stl/compatibility/Nodiscard.h>
#include <simd_stl/compatibility/Inline.h>

#include <src/simd_stl/algorithm/vectorized/ReplaceVectorized.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>

#include <src/simd_stl/algorithm/AdvanceBytes.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    class _ForwardIterator_,
    class _Type_ = type_traits::IteratorValueType<_ForwardIterator_>>
simd_stl_constexpr_cxx20 simd_stl_always_inline void replace(
    _ForwardIterator_                                   first,
    _ForwardIterator_                                   last,
    const typename std::type_identity<_Type_>::type&    oldValue,
    const typename std::type_identity<_Type_>::type&    newValue) noexcept
{
    __verifyRange(first, last);

    auto firstUnwrapped         = _UnwrapIterator(first);
    const auto lastUnwrapped    = _UnwrapIterator(last);

    for (; firstUnwrapped != lastUnwrapped; ++firstUnwrapped)
        if (*firstUnwrapped == oldValue)
            *firstUnwrapped = newValue;
}

template <
    class _ForwardIterator_,
    class _UnaryPredicate_,
    class _Type_ = type_traits::IteratorValueType<_ForwardIterator_>>
simd_stl_constexpr_cxx20 simd_stl_always_inline void replace_if(
    _ForwardIterator_                                   first,
    _ForwardIterator_                                   last,
    _UnaryPredicate_                                    predicate,
    const typename std::type_identity<_Type_>::type&    newValue) noexcept(
        type_traits::is_nothrow_invocable_v<_UnaryPredicate_, _Type_>)
{
    __verifyRange(first, last);

    auto firstUnwrapped         = _UnwrapIterator(first);
    const auto lastUnwrapped    = _UnwrapIterator(last);

    for (; firstUnwrapped != lastUnwrapped; ++firstUnwrapped)
        if (predicate(*firstUnwrapped))
            *firstUnwrapped = newValue;
}


template <
    class _ExecutionPolicy_,
    class _ForwardIterator_,
    class _Type_ = type_traits::IteratorValueType<_ForwardIterator_>>
simd_stl_constexpr_cxx20 simd_stl_always_inline void replace(
    _ExecutionPolicy_&&,
    _ForwardIterator_                                   first,
    _ForwardIterator_                                   last,
    const typename std::type_identity<_Type_>::type&    oldValue,
    const typename std::type_identity<_Type_>::type&    newValue) noexcept
{
    return simd_stl::algorithm::replace(first, last, oldValue, newValue);
}

template <
    class _ExecutionPolicy_,
    class _ForwardIterator_,
    class _UnaryPredicate_,
    class _Type_ = type_traits::IteratorValueType<_ForwardIterator_>>
simd_stl_constexpr_cxx20 simd_stl_always_inline void replace_if(
    _ExecutionPolicy_&&,
    _ForwardIterator_                                   first,
    _ForwardIterator_                                   last,
    _UnaryPredicate_                                    predicate,
    const typename std::type_identity<_Type_>::type&    newValue) noexcept(
        type_traits::is_nothrow_invocable_v<_UnaryPredicate_, _Type_>)
{
    return simd_stl::algorithm::replace_if(first, last, type_traits::passFunction(predicate), newValue);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
