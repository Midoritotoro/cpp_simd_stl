#pragma once 

#include <src/simd_stl/algorithm/unchecked/copy/CopyUnchecked.h>
#include <src/simd_stl/algorithm/unchecked/copy/CopyIfUnchecked.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    class _InputIterator_,
    class _OutputIterator_>
__simd_inline_constexpr _OutputIterator_ copy(
    _InputIterator_     __first,
    _InputIterator_     _Last,
    _OutputIterator_    _Destination) noexcept
{
    __verifyRange(__first, _Last);

    __seek_possibly_wrapped_iterator(_Destination, _CopyUnchecked(_UnwrapIterator(__first),
        _UnwrapIterator(_Last), _UnwrapUnverifiedIterator(_Destination)));

    return _Destination;
}

template <
    class _InputIterator_,
    class _OutputIterator_,
    class _Predicate_>
__simd_inline_constexpr _OutputIterator_ copy_if(
    _InputIterator_     __first,
    _InputIterator_     _Last,
    _OutputIterator_    _Destination,
    _Predicate_         _Predicate) noexcept(
		type_traits::is_nothrow_invocable_v<
			_Predicate_, type_traits::IteratorValueType<_InputIterator_>>)
{
    __verifyRange(__first, _Last);

    __seek_possibly_wrapped_iterator(_Destination, _CopyIfUnchecked(_UnwrapIterator(__first),
        _UnwrapIterator(_Last), _UnwrapUnverifiedIterator(_Destination), type_traits::passFunction(_Predicate)));

    return _Destination;
}

template <
    class _ExecutionPolicy_,
    class _InputIterator_,
    class _OutputIterator_,
    concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
_OutputIterator_ copy(
    _ExecutionPolicy_&&,
    _InputIterator_     __first,
    _InputIterator_     _Last,
    _OutputIterator_    _Destination) noexcept
{
    return simd_stl::algorithm::copy(__first, _Last, _Destination);
}

template <
    class _ExecutionPolicy_,
    class _InputIterator_,
    class _OutputIterator_,
    class _Predicate_,
    concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
_OutputIterator_ copy_if(
    _ExecutionPolicy_&&,
    _InputIterator_     __first,
    _InputIterator_     _Last,
    _OutputIterator_    _Destination,
    _Predicate_         _Predicate) noexcept(
        type_traits::is_nothrow_invocable_v<
            _Predicate_, type_traits::IteratorValueType<_InputIterator_>>)
{
    return simd_stl::algorithm::copy_if(__first, _Last, _Destination, type_traits::passFunction(_Predicate));
}


__SIMD_STL_ALGORITHM_NAMESPACE_END