#pragma once 

#include <simd_stl/compatibility/SimdCompatibility.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

struct ZeroUpperOnDeleteAvx {
    ZeroUpperOnDeleteAvx() = default;

    ZeroUpperOnDeleteAvx(const ZeroUpperOnDeleteAvx&) = delete;
    ZeroUpperOnDeleteAvx& operator=(const ZeroUpperOnDeleteAvx&) = delete;

    ~ZeroUpperOnDeleteAvx() {
        _mm256_zeroupper();
    }
};

__SIMD_STL_ALGORITHM_NAMESPACE_END
