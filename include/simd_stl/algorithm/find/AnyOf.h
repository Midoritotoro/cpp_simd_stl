#pragma once 

#include <simd_stl/algorithm/find/Find.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    class _InputIterator_, 
    class _Predicate_>
__simd_inline_constexpr bool any_of(
    _InputIterator_ __first,
    _InputIterator_ _Last,
    _Predicate_     _Predicate) noexcept(
        type_traits::is_nothrow_invocable_v<
		    _Predicate_, type_traits::IteratorValueType<_InputIterator_>>)
{
    return (simd_stl::algorithm::find_if(__first, _Last, type_traits::passFunction(_Predicate)) != _Last);
}

template <
    class _ExecutionPolicy_,
    class _InputIterator_,
    class _Predicate_,
    concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
simd_stl_nodiscard simd_stl_constexpr_cxx20 simd_stl_always_inline bool any_of(
    _ExecutionPolicy_&&,
    _InputIterator_ __first,
    _InputIterator_ _Last,
    _Predicate_     _Predicate) noexcept(
        type_traits::is_nothrow_invocable_v<
            _Predicate_, type_traits::IteratorValueType<_InputIterator_>>)
{
    return simd_stl::algorithm::any_of(__first, _Last, type_traits::passFunction(_Predicate));
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
