#pragma once 

#include <simd_stl/numeric/BasicSimd.h>

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN


template <typename _IntegralType_ = int>
using simd_sse2 = basic_simd<arch::CpuFeature::SSE2, _IntegralType_>;

template <typename _IntegralType_ = int>
using simd_sse3 = basic_simd<arch::CpuFeature::SSE3, _IntegralType_>;

template <typename _IntegralType_ = int>
using simd_ssse3 = basic_simd<arch::CpuFeature::SSSE3, _IntegralType_>;

template <typename _IntegralType_ = int>
using simd_sse41 = basic_simd<arch::CpuFeature::SSE41, _IntegralType_>;

template <typename _IntegralType_ = int>
using simd_sse42 = basic_simd<arch::CpuFeature::SSE42, _IntegralType_>;

template <typename _IntegralType_ = int>
using simd_avx2 = basic_simd<arch::CpuFeature::AVX2, _IntegralType_>;

template <typename _IntegralType_ = int>
using simd_avx512f = basic_simd<arch::CpuFeature::AVX512F, _IntegralType_>;

template <typename _IntegralType_ = int>
using simd_avx512bw = basic_simd<arch::CpuFeature::AVX512BW, _IntegralType_>;


__SIMD_STL_NUMERIC_NAMESPACE_END
