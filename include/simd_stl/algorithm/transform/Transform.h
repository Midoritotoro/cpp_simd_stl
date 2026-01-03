#pragma once 

#include <src/simd_stl/algorithm/unchecked/transform/TransformUnchecked.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    class _InputIterator_,
    class _OutputIterator_,
    class _UnaryPredicate_>
__simd_nodiscard_inline_constexpr _OutputIterator_ transform(
    _InputIterator_     _First,
    _InputIterator_     _Last,
    _OutputIterator_    _Destination,
    _UnaryPredicate_    _Predicate) noexcept
{
    __verifyRange(_First, _Last);
    __seek_possibly_wrapped_iterator(_Destination, _TransformUnchecked(_UnwrapIterator(_First), _UnwrapIterator(_Last),
        _UnwrapIterator(_Destination), type_traits::passFunction(_Predicate)));

    return _Destination;
}

template <
    class _FirstInputIterator_,
    class _SecondInputIterator_,
    class _OutputIterator_,
    class _UnaryPredicate_>
__simd_nodiscard_inline_constexpr _OutputIterator_ transform(
    _FirstInputIterator_    _First1,
    _FirstInputIterator_    _Last1,
    _SecondInputIterator_   _First2,
    _OutputIterator_        _Destination,
    _UnaryPredicate_        _Predicate) noexcept
{
    __verifyRange(_First1, _Last1);
    __seek_possibly_wrapped_iterator(_Destination, _TransformUnchecked(_UnwrapIterator(_First1), _UnwrapIterator(_Last1),
        _UnwrapIterator(_First2), _UnwrapIterator(_Destination), type_traits::passFunction(_Predicate)));

    return _Destination;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
