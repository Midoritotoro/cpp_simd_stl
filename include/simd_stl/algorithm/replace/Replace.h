#pragma once 


#include <src/simd_stl/algorithm/unchecked/replace/ReplaceIfUnchecked.h>
#include <src/simd_stl/algorithm/unchecked/replace/ReplaceUnchecked.h>

#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>
#include <simd_stl/concurrency/Execution.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    class _ForwardIterator_,
    class _Type_ = type_traits::iterator_value_type<_ForwardIterator_>>
simd_stl_constexpr_cxx20 simd_stl_always_inline void replace(
    _ForwardIterator_                                   __first,
    _ForwardIterator_                                   __last,
    const typename std::type_identity<_Type_>::type&    __old_value,
    const typename std::type_identity<_Type_>::type&    __new_value) noexcept
{
    __verify_range(__first, __last);
    __replace_unchecked(__unwrap_iterator(__first), __unwrap_iterator(__last), __old_value, __new_value);
}

template <
    class _ForwardIterator_,
    class _UnaryPredicate_,
    class _Type_ = type_traits::iterator_value_type<_ForwardIterator_>>
simd_stl_constexpr_cxx20 simd_stl_always_inline void replace_if(
    _ForwardIterator_                                   __first,
    _ForwardIterator_                                   __last,
    _UnaryPredicate_                                    __predicate,
    const typename std::type_identity<_Type_>::type&    __new_value) noexcept(
        type_traits::is_nothrow_invocable_v<_UnaryPredicate_, _Type_>)
{
    __verify_range(__first, __last);
    __replace_if_unchecked(__unwrap_iterator(__first), __unwrap_iterator(__last),
        type_traits::__pass_function(__predicate), __new_value);
}


template <
    class _ExecutionPolicy_,
    class _ForwardIterator_,
    class _Type_ = type_traits::iterator_value_type<_ForwardIterator_>,
    concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
simd_stl_constexpr_cxx20 simd_stl_always_inline void replace(
    _ExecutionPolicy_&&,
    _ForwardIterator_                                   __first,
    _ForwardIterator_                                   __last,
    const typename std::type_identity<_Type_>::type&    __old_value,
    const typename std::type_identity<_Type_>::type&    __new_value) noexcept
{
    return simd_stl::algorithm::replace(__first, __last, __old_value, __new_value);
}

template <
    class _ExecutionPolicy_,
    class _ForwardIterator_,
    class _UnaryPredicate_,
    class _Type_ = type_traits::iterator_value_type<_ForwardIterator_>,
    concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
simd_stl_constexpr_cxx20 simd_stl_always_inline void replace_if(
    _ExecutionPolicy_&&,
    _ForwardIterator_                                   __first,
    _ForwardIterator_                                   __last,
    _UnaryPredicate_                                    __predicate,
    const typename std::type_identity<_Type_>::type&    __new_value) noexcept(
        type_traits::is_nothrow_invocable_v<_UnaryPredicate_, _Type_>)
{
    return simd_stl::algorithm::replace_if(__first, __last, type_traits::__pass_function(__predicate), __new_value);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
