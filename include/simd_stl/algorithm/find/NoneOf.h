#pragma once 

#include <simd_stl/algorithm/find/Find.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    class _InputIterator_, 
    class _Predicate_>
simd_stl_nodiscard simd_stl_constexpr_cxx20 simd_stl_always_inline bool none_of(
    _InputIterator_ __first,
    _InputIterator_ __last,
    _Predicate_     __predicate) noexcept(
        type_traits::is_nothrow_invocable_v<
		    _Predicate_,
		    type_traits::iterator_value_type<_InputIterator_>>)
{
    return (simd_stl::algorithm::find_if(__first, __last, type_traits::__pass_function(__predicate)) == __last);
}


template <
    class _ExecutionPolicy_,
    class _InputIterator_, 
    class _Predicate_,
    concurrency::enable_if_execution_policy<_ExecutionPolicy_> = 0>
simd_stl_nodiscard simd_stl_constexpr_cxx20 simd_stl_always_inline bool none_of(
    _ExecutionPolicy_&&,
    _InputIterator_ __first,
    _InputIterator_ __last,
    _Predicate_     __predicate) noexcept(
        type_traits::is_nothrow_invocable_v<
            _Predicate_,
            type_traits::iterator_value_type<_InputIterator_>>)
{
    return simd_stl::algorithm::none_of(__first, __last, type_traits::__pass_function(__predicate));
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
