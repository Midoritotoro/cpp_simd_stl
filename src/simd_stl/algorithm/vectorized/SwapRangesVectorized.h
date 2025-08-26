#pragma once

#include <simd_stl/compatibility/SimdCompatibility.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

simd_stl_declare_const_function simd_stl_constexpr_cxx20 void simd_stl_cdecl SwapRangesVectorized(
    void* firstPointer1,
    void* lastPointer1,
    void* firstPointer2) noexcept;

__SIMD_STL_ALGORITHM_NAMESPACE_END
