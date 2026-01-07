#pragma once

#include <src/simd_stl/algorithm/AdvanceBytes.h>
#include <simd_stl/compatibility/SimdCompatibility.h>

#include <simd_stl/compatibility/FunctionAttributes.h>
#include <simd_stl/arch/ProcessorFeatures.h>

#include <src/simd_stl/type_traits/SimdTypeCheck.h>
#include <simd_stl/numeric/BasicSimd.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN


void* __memmove_vectorized(
    void*       __destination,
    const void* __source,
    sizetype    __bytes) noexcept
{
    return memmove(__destination, __source, __bytes);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
