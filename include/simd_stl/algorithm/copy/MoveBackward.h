#pragma once 

#include <src/simd_stl/algorithm/unchecked/copy/MoveBackwardUnchecked.h>
#include <simd_stl/concurrency/Execution.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    class _BidirectionalFirstIterator_,
    class _BidirectionalSecondIterator_>
_Simd_inline_constexpr _BidirectionalSecondIterator_ move_backward(
    _BidirectionalFirstIterator_    _First,
    _BidirectionalFirstIterator_    _Last,
    _BidirectionalSecondIterator_   _DestinationLast) noexcept
{
    __verifyRange(_First, _Last);

    _SeekPossiblyWrappedIterator(_DestinationLast, _MoveBackwardUnchecked(_UnwrapIterator(_First),
        _UnwrapIterator(_Last), _UnwrapUnverifiedIterator(_DestinationLast)));

    return _DestinationLast;
}

template <
    class _ExecutionPolicy_,
    class _BidirectionalFirstIterator_,
    class _BidirectionalSecondIterator_,
    concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
_BidirectionalSecondIterator_ move_backward(
    _ExecutionPolicy_&&,
    _BidirectionalFirstIterator_    _First,
    _BidirectionalFirstIterator_    _Last,
    _BidirectionalSecondIterator_   _DestinationLast) noexcept
{
    return simd_stl::algorithm::move_backward(_First, _Last, _DestinationLast);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END