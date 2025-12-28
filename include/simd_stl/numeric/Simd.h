#pragma once 

#include <simd_stl/numeric/BasicSimd.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <typename _Type_>
using simd128_sse2			= simd<arch::CpuFeature::SSE2, _Type_, xmm128>;

template <typename _Type_>
using simd128_sse3			= simd<arch::CpuFeature::SSE3, _Type_, xmm128>;

template <typename _Type_>
using simd128_ssse3			= simd<arch::CpuFeature::SSSE3, _Type_, xmm128>;

template <typename _Type_>
using simd128_sse41			= simd<arch::CpuFeature::SSE41, _Type_, xmm128>;

template <typename _Type_>
using simd128_sse42			= simd<arch::CpuFeature::SSE42, _Type_, xmm128>;

template <typename _Type_>
using simd128_avx			= simd<arch::CpuFeature::AVX, _Type_, xmm128>;

template <typename _Type_>
using simd128_avx2			= simd<arch::CpuFeature::AVX2, _Type_, xmm128>;

template <typename _Type_>
using simd128_avx512bwvl	= simd<arch::CpuFeature::AVX512VLBW, _Type_, xmm128>;

template <typename _Type_>
using simd128_avx512vlf		= simd<arch::CpuFeature::AVX512VLF, _Type_, xmm128>;

template <typename _Type_>
using simd128_avx512vldq	= simd<arch::CpuFeature::AVX512VLDQ, _Type_, xmm128>;


template <typename _Type_>
using simd256_avx			= simd<arch::CpuFeature::AVX, _Type_, ymm256>;

template <typename _Type_>
using simd256_avx2			= simd<arch::CpuFeature::AVX2, _Type_, ymm256>;

template <typename _Type_>
using simd256_avx512vlbw	= simd<arch::CpuFeature::AVX512VLBW, _Type_, ymm256>;

template <typename _Type_>
using simd256_avx512vlf		= simd<arch::CpuFeature::AVX512VLF, _Type_, ymm256>;

template <typename _Type_>
using simd256_avx512vldq	= simd<arch::CpuFeature::AVX512VLDQ, _Type_, ymm256>;


template <typename _Type_>
using simd512_avx512f		= simd<arch::CpuFeature::AVX512F, _Type_, zmm512>;

template <typename _Type_>
using simd512_avx512bw		= simd<arch::CpuFeature::AVX512BW, _Type_, zmm512>;

template <typename _Type_>
using simd512_avx512dq		= simd<arch::CpuFeature::AVX512DQ, _Type_, zmm512>;


__SIMD_STL_NUMERIC_NAMESPACE_END
