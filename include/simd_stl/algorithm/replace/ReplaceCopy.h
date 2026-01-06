#pragma once 


#include <src/simd_stl/algorithm/unchecked/replace/ReplaceCopyIfUnchecked.h>
#include <src/simd_stl/algorithm/unchecked/replace/ReplaceCopyUnchecked.h>

#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>
#include <simd_stl/concurrency/Execution.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    class _InputIterator_,
    class _OutputIterator_,
    class _Type_ = type_traits::iterator_value_type<_InputIterator_>>
__simd_inline_constexpr void replace_copy(
        _InputIterator_                                     __first,
        _InputIterator_                                     __last,
        _OutputIterator_                                    __destination,
        const typename std::type_identity<_Type_>::type&    __old_value,
        const typename std::type_identity<_Type_>::type&    __new_value) noexcept
{
    __verify_range(__first, __last);
    __replace_copy_unchecked(__unwrap_iterator(__first), __unwrap_iterator(__last),
        __unwrap_iterator(__destination), __old_value, __new_value);
}

template <
    class _InputIterator_,
    class _OutputIterator_,
    class _UnaryPredicate_,
    class _Type_ = type_traits::iterator_value_type<_InputIterator_>>
__simd_inline_constexpr simd_stl_always_inline void replace_copy_if(
    _InputIterator_                                     __first,
    _InputIterator_                                     __last,
    _OutputIterator_                                    __destination,
    _UnaryPredicate_                                    __predicate,
    const typename std::type_identity<_Type_>::type&    __new_value) noexcept(
        type_traits::is_nothrow_invocable_v<_UnaryPredicate_, _Type_>)
{
    __verify_range(__first, __last);
    __replace_copy_if_unchecked(__unwrap_iterator(__first), __unwrap_iterator(__last),
        __unwrap_iterator(__destination), type_traits::__pass_function(__predicate), __new_value);
}


template <
    class _ExecutionPolicy_,
    class _SourceForwardIterator_,
    class _DestinationForwardIterator_,
    class _Type_ = type_traits::iterator_value_type<_SourceForwardIterator_>,
    concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
__simd_inline_constexpr void replace_copy(
    _ExecutionPolicy_&&,
    _SourceForwardIterator_                             __first,
    _SourceForwardIterator_                             __last,
    _DestinationForwardIterator_                        __destination,
    const typename std::type_identity<_Type_>::type&    __old_value,
    const typename std::type_identity<_Type_>::type&    __new_value) noexcept
{
    return simd_stl::algorithm::replace_copy(__first, __last, __destination, __old_value, __new_value);
}

template <
    class _ExecutionPolicy_,
    class _SourceForwardIterator_,
    class _DestinationForwardIterator_,
    class _UnaryPredicate_,
    class _Type_ = type_traits::iterator_value_type<_SourceForwardIterator_>,
    concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
__simd_inline_constexpr void replace_copy_if(
    _ExecutionPolicy_&&,
    _SourceForwardIterator_                             __first,
    _SourceForwardIterator_                             __last,
    _DestinationForwardIterator_                        __destination,
    _UnaryPredicate_                                    __predicate,
    const typename std::type_identity<_Type_>::type&    __new_value) noexcept(
        type_traits::is_nothrow_invocable_v<_UnaryPredicate_, _Type_>)
{
    return simd_stl::algorithm::replace_copy_if(__first, __last, __destination, type_traits::__pass_function(__predicate), __new_value);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
