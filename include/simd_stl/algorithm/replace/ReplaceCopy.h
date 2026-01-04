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
        _InputIterator_                                     _First,
        _InputIterator_                                     _Last,
        _OutputIterator_                                    _Destination,
        const typename std::type_identity<_Type_>::type&    _OldValue,
        const typename std::type_identity<_Type_>::type&    _NewValue) noexcept
{
    __verifyRange(_First, _Last);
    _ReplaceCopyUnchecked(_UnwrapIterator(_First), _UnwrapIterator(_Last),
        _UnwrapIterator(_Destination), _OldValue, _NewValue);
}

template <
    class _InputIterator_,
    class _OutputIterator_,
    class _UnaryPredicate_,
    class _Type_ = type_traits::iterator_value_type<_InputIterator_>>
__simd_inline_constexpr simd_stl_always_inline void replace_copy_if(
    _InputIterator_                                     _First,
    _InputIterator_                                     _Last,
    _OutputIterator_                                    _Destination,
    _UnaryPredicate_                                    _Predicate,
    const typename std::type_identity<_Type_>::type&    _NewValue) noexcept(
        type_traits::is_nothrow_invocable_v<_UnaryPredicate_, _Type_>)
{
    __verifyRange(_First, _Last);
    _ReplaceIfUnchecked(_UnwrapIterator(_First), _UnwrapIterator(_Last),
        _UnwrapIterator(_Destination), type_traits::passFunction(_Predicate), _NewValue);
}


template <
    class _ExecutionPolicy_,
    class _SourceForwardIterator_,
    class _DestinationForwardIterator_,
    class _Type_ = type_traits::iterator_value_type<_SourceForwardIterator_>,
    concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
__simd_inline_constexpr void replace_copy(
    _ExecutionPolicy_&&,
    _SourceForwardIterator_                             _First,
    _SourceForwardIterator_                             _Last,
    _DestinationForwardIterator_                        _Destination,
    const typename std::type_identity<_Type_>::type&    _OldValue,
    const typename std::type_identity<_Type_>::type&    _NewValue) noexcept
{
    return simd_stl::algorithm::replace_copy(_First, _Last, _Destination, _OldValue, _NewValue);
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
    _SourceForwardIterator_                             _First,
    _SourceForwardIterator_                             _Last,
    _DestinationForwardIterator_                        _Destination,    
    _UnaryPredicate_                                    _Predicate,
    const typename std::type_identity<_Type_>::type&    _NewValue) noexcept(
        type_traits::is_nothrow_invocable_v<_UnaryPredicate_, _Type_>)
{
    return simd_stl::algorithm::replace_copy_if(_First, _Last, _Destination, type_traits::passFunction(_Predicate), _NewValue);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
