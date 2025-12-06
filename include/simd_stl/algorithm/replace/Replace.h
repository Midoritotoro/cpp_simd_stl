#pragma once 


#include <src/simd_stl/algorithm/unchecked/replace/ReplaceIfUnchecked.h>
#include <src/simd_stl/algorithm/unchecked/replace/ReplaceUnchecked.h>

#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    class _ForwardIterator_,
    class _Type_ = type_traits::IteratorValueType<_ForwardIterator_>>
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
    class _Type_ = type_traits::IteratorValueType<_ForwardIterator_>>
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
