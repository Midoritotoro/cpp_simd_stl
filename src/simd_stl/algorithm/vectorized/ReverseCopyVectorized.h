#pragma once

#include <simd_stl/compatibility/SimdCompatibility.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <size_t typeSizeInBytes>
simd_stl_declare_const_function simd_stl_constexpr_cxx20   const void* ReverseCopyVectorized(
    void* firstPointer,
    void* lastPointer,
    void* destinationPointer) noexcept;

__SIMD_STL_ALGORITHM_NAMESPACE_END
