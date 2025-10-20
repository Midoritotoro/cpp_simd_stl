#pragma once 

#include <simd_stl/algorithm/find/Find.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    class _InputIterator_, 
    class _Predicate_>
simd_stl_nodiscard simd_stl_constexpr_cxx20 simd_stl_always_inline bool none_of(
    _InputIterator_ first,
    _InputIterator_ last,
    _Predicate_     predicate) noexcept(
        type_traits::is_nothrow_invocable_v<
		    _Predicate_,
		    type_traits::IteratorValueType<_InputIterator_>
        >
    )
{
    return (simd_stl::algorithm::find_if(first, last, type_traits::passFunction(predicate)) == last);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
