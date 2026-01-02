#pragma once 

#include <simd_stl/numeric/Simd.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    class _Function_,
    class _Simd_> 
constexpr inline bool _Is_predicate_vectorizable_v = type_traits::is_invocable_v<_Function_, _Simd_> 
    || type_traits::is_invocable_v<_Function_, _Simd_, _Simd_>;

__SIMD_STL_ALGORITHM_NAMESPACE_END