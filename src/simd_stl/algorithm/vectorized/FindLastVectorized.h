#pragma once

#include <simd_stl/compatibility/SimdCompatibility.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
simd_stl_declare_const_function simd_stl_constexpr_cxx20 const void* FindLastVectorized(
    const void*     firstPointer,
    const void*     lastPointer,
    const _Type_&   value) noexcept;

__SIMD_STL_ALGORITHM_NAMESPACE_END
