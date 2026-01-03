#pragma once 

#include <src/simd_stl/algorithm/unchecked/copy/CopyBackwardUnchecked.h>
#include <simd_stl/concurrency/Execution.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    class _BidirectionalFirstIterator_,
    class _BidirectionalSecondIterator_>
__simd_inline_constexpr _BidirectionalSecondIterator_ copy_backward(
    _BidirectionalFirstIterator_    __first,
    _BidirectionalFirstIterator_    _Last,
    _BidirectionalSecondIterator_   _DestinationLast) noexcept
{
    __verifyRange(__first, _Last);
    
    __seek_possibly_wrapped_iterator(_DestinationLast, _CopyBackwardUnchecked(_UnwrapIterator(__first),
        _UnwrapIterator(_Last), _UnwrapUnverifiedIterator(_DestinationLast)));

    return _DestinationLast;
}

template <
    class _ExecutionPolicy_,
    class _BidirectionalFirstIterator_,
    class _BidirectionalSecondIterator_,
    concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
_BidirectionalSecondIterator_ copy_backward(
    _ExecutionPolicy_&&,
    _BidirectionalFirstIterator_    __first,
    _BidirectionalFirstIterator_    _Last,
    _BidirectionalSecondIterator_   _DestinationLast) noexcept
{
    return simd_stl::algorithm::copy_backward(__first, _Last, _DestinationLast);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END