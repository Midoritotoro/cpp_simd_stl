#pragma once 

#include <simd_stl/algorithm/find/Find.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    class _InputIterator_, 
    class _Predicate_>
_Simd_nodiscard_inline_constexpr bool all_of(
    _InputIterator_ _First,
    _InputIterator_ _Last,
    _Predicate_     _Predicate) noexcept(
        type_traits::is_nothrow_invocable_v<
		    _Predicate_, type_traits::IteratorValueType<_InputIterator_>>)
{
    return (simd_stl::algorithm::find_if_not(_First, _Last, type_traits::passFunction(_Predicate)) == _Last);
}

template <
    class _ExecutionPolicy_,
    class _InputIterator_,
    class _Predicate_,
    concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
simd_stl_nodiscard bool all_of(
    _ExecutionPolicy_&&,
    _InputIterator_ _First,
    _InputIterator_ _Last,
    _Predicate_     _Predicate) noexcept(
        type_traits::is_nothrow_invocable_v<
        _Predicate_, type_traits::IteratorValueType<_InputIterator_>>)

{
    return simd_stl::algorithm::all_of(_First, _Last, type_traits::passFunction(_Predicate));
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
