#pragma once

#include <simd_stl/compatibility/SimdCompatibility.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <size_t typeSizeInbytes>
simd_stl_declare_const_function simd_stl_constexpr_cxx20 size_t MismatchVectorized(
    const void* const   firstRangeBegin,
    const void* const   secondRangeBegin,
    const size_t        length) noexcept;

__SIMD_STL_ALGORITHM_NAMESPACE_END
