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
    _ForwardIterator_                                   _First,
    _ForwardIterator_                                   _Last,
    const typename std::type_identity<_Type_>::type&    _OldValue,
    const typename std::type_identity<_Type_>::type&    _NewValue) noexcept
{
    __verifyRange(_First, _Last);
    _ReplaceUnchecked(_UnwrapIterator(_First), _UnwrapIterator(_Last), _OldValue, _NewValue);
}

template <
    class _ForwardIterator_,
    class _UnaryPredicate_,
    class _Type_ = type_traits::iterator_value_type<_ForwardIterator_>>
simd_stl_constexpr_cxx20 simd_stl_always_inline void replace_if(
    _ForwardIterator_                                   _First,
    _ForwardIterator_                                   _Last,
    _UnaryPredicate_                                    _Predicate,
    const typename std::type_identity<_Type_>::type&    _NewValue) noexcept(
        type_traits::is_nothrow_invocable_v<_UnaryPredicate_, _Type_>)
{
    __verifyRange(_First, _Last);
    _ReplaceIfUnchecked(_UnwrapIterator(_First), _UnwrapIterator(_Last),
        type_traits::passFunction(_Predicate), _NewValue);
}


template <
    class _ExecutionPolicy_,
    class _ForwardIterator_,
    class _Type_ = type_traits::iterator_value_type<_ForwardIterator_>,
    concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
simd_stl_constexpr_cxx20 simd_stl_always_inline void replace(
    _ExecutionPolicy_&&,
    _ForwardIterator_                                   _First,
    _ForwardIterator_                                   _Last,
    const typename std::type_identity<_Type_>::type&    _OldValue,
    const typename std::type_identity<_Type_>::type&    _NewValue) noexcept
{
    return simd_stl::algorithm::replace(_First, _Last, _OldValue, _NewValue);
}

template <
    class _ExecutionPolicy_,
    class _ForwardIterator_,
    class _UnaryPredicate_,
    class _Type_ = type_traits::iterator_value_type<_ForwardIterator_>,
    concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
simd_stl_constexpr_cxx20 simd_stl_always_inline void replace_if(
    _ExecutionPolicy_&&,
    _ForwardIterator_                                   _First,
    _ForwardIterator_                                   _Last,
    _UnaryPredicate_                                    _Predicate,
    const typename std::type_identity<_Type_>::type&    _NewValue) noexcept(
        type_traits::is_nothrow_invocable_v<_UnaryPredicate_, _Type_>)
{
    return simd_stl::algorithm::replace_if(_First, _Last, type_traits::passFunction(_Predicate), _NewValue);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
