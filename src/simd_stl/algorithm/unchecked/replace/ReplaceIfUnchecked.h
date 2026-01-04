#pragma once 

#include <src/simd_stl/algorithm/AlgorithmDebug.h>
#include <src/simd_stl/type_traits/SimdAlgorithmSafety.h>

#include <src/simd_stl/algorithm/vectorized/replace/ReplaceVectorized.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    class _UnwrappedForwardIterator_,
    class _UnaryPredicate_,
    class _Type_ = type_traits::iterator_value_type<_UnwrappedForwardIterator_>>
__simd_inline_constexpr void __replace_if_unchecked(
    _UnwrappedForwardIterator_                          __first_unwrapped,
    _UnwrappedForwardIterator_                          __last_unwrapped,
    _UnaryPredicate_                                    __predicate,
    const typename std::type_identity<_Type_>::type&    __new_value) noexcept(
        type_traits::is_nothrow_invocable_v<_UnaryPredicate_, _Type_>)
{
    for (; __first_unwrapped != __last_unwrapped; ++__first_unwrapped)
        if (__predicate(*__first_unwrapped))
            *__first_unwrapped = __new_value;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
