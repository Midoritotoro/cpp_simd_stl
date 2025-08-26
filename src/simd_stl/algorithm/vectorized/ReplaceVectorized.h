#pragma once

#include <simd_stl/compatibility/SimdCompatibility.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <typename _Type_>
simd_stl_declare_const_function void simd_stl_stdcall ReplaceVectorized(
    void*           firstPointer,
    void*           lastPointer,
    const _Type_&   oldValue,
    const _Type_&   newValue) noexcept;

__SIMD_STL_ALGORITHM_NAMESPACE_END

