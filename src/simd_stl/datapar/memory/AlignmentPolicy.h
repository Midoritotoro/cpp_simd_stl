#pragma once 

#include <simd_stl/SimdStlNamespace.h>


__SIMD_STL_DATAPAR_NAMESPACE_BEGIN

struct __aligned_policy {
    static constexpr bool __alignment = true;
};

struct __unaligned_policy {
    static constexpr bool __alignment = false;
};

__SIMD_STL_DATAPAR_NAMESPACE_END
