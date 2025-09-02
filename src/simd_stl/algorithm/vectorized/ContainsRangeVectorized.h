#pragma once

#include <simd_stl/compatibility/SimdCompatibility.h>

#include <simd_stl/compatibility/Inline.h>
#include <simd_stl/compatibility/FunctionAttributes.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
simd_stl_declare_const_function simd_stl_constexpr_cxx20 const void* ContainsRangeVectorized(
    const void* firstMainRangePointer,
    const void* lastMainRangePointer,
    const void* firstSubRangePointer,
    const void* lastSubRangePointer) noexcept;

__SIMD_STL_ALGORITHM_NAMESPACE_END
