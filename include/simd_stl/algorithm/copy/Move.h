#pragma once 

#include <src/simd_stl/algorithm/unchecked/copy/MoveUnchecked.h>
#include <simd_stl/concurrency/Execution.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    class _InputIterator_,
    class _OutputIterator_>
__simd_inline_constexpr _OutputIterator_ move(
    _InputIterator_     _First,
    _InputIterator_     _Last,
    _OutputIterator_    _Destination) noexcept
{
    __verifyRange(_First, _Last);

    __seek_possibly_wrapped_iterator(_Destination, _MoveUnchecked(_UnwrapIterator(_First),
        _UnwrapIterator(_Last), _UnwrapUnverifiedIterator(_Destination)));

    return _Destination;
}

template <
    class _ExecutionPolicy_,
    class _InputIterator_,
    class _OutputIterator_,
    concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
_OutputIterator_ move(
    _ExecutionPolicy_&&,
    _InputIterator_     _First,
    _InputIterator_     _Last,
    _OutputIterator_    _Destination) noexcept
{
    return simd_stl::algorithm::move(_First, _Last, _Destination);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END