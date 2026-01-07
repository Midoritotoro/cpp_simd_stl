#pragma once

#include <src/simd_stl/algorithm/vectorized/copy/MoveVectorized.h>

#include <simd_stl/memory/Intersects.h>
#include <simd_stl/memory/Alignment.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

void* __memcpy_vectorized(
    void*       __destination,
    const void* __source,
    sizetype    __bytes) noexcept
{
    return memcpy(__destination, __source, __bytes);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END

