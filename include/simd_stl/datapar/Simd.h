#pragma once 

#include <simd_stl/datapar/BasicSimd.h>
#include <simd_stl/datapar/SimdConfig.h>


__SIMD_STL_DATAPAR_NAMESPACE_BEGIN

template <typename _Type_>
using simd128_sse2			= simd<arch::ISA::SSE2, _Type_, xmm128>;

template <typename _Type_>
using simd128_sse3			= simd<arch::ISA::SSE3, _Type_, xmm128>;

template <typename _Type_>
using simd128_ssse3			= simd<arch::ISA::SSSE3, _Type_, xmm128>;

template <typename _Type_>
using simd128_sse41			= simd<arch::ISA::SSE41, _Type_, xmm128>;

template <typename _Type_>
using simd128_sse42			= simd<arch::ISA::SSE42, _Type_, xmm128>;

template <typename _Type_>
using simd128_avx512vlbw	= simd<arch::ISA::AVX512VLBW, _Type_, xmm128>;

template <typename _Type_>
using simd128_avx512vlf		= simd<arch::ISA::AVX512VLF, _Type_, xmm128>;

template <typename _Type_>
using simd128_avx512vldq	= simd<arch::ISA::AVX512VLDQ, _Type_, xmm128>;


template <typename _Type_>
using simd256_avx			= simd<arch::ISA::AVX, _Type_, ymm256>;

template <typename _Type_>
using simd256_avx2			= simd<arch::ISA::AVX2, _Type_, ymm256>;

template <typename _Type_>
using simd256_avx512vlbw	= simd<arch::ISA::AVX512VLBW, _Type_, ymm256>;

template <typename _Type_>
using simd256_avx512vlf		= simd<arch::ISA::AVX512VLF, _Type_, ymm256>;

template <typename _Type_>
using simd256_avx512vldq	= simd<arch::ISA::AVX512VLDQ, _Type_, ymm256>;

template <typename _Type_>
using simd256_avx512vlbwdq	= simd<arch::ISA::AVX512VLBWDQ, _Type_, ymm256>;


template <typename _Type_>
using simd512_avx512f		= simd<arch::ISA::AVX512F, _Type_, zmm512>;

template <typename _Type_>
using simd512_avx512bw		= simd<arch::ISA::AVX512BW, _Type_, zmm512>;

template <typename _Type_>
using simd512_avx512dq		= simd<arch::ISA::AVX512DQ, _Type_, zmm512>;

template <typename _Type_>
using simd512_avx512bwdq	= simd<arch::ISA::AVX512BWDQ, _Type_, zmm512>;


template <typename _Type_>
//#if defined(SIMD_STL_HAS_AVX512VBMI2_SUPPORT) 
//  using simd_native = simd<arch::CpuFeature::AVX512VBMI2, _Type_, zmm512>;
//
//#elif defined(SIMD_STL_HAS_AVX512VBMI_SUPPORT)
//  using simd_native = simd<arch::CpuFeature::AVX512VBMI, _Type_, zmm512>;

#if defined(SIMD_STL_HAS_AVX512DQ_SUPPORT) && defined(SIMD_STL_HAS_AVX512BW_SUPPORT)
  using simd_native = simd<arch::ISA::AVX512BWDQ, _Type_, zmm512>;

#elif defined(SIMD_STL_HAS_AVX512VL_SUPPORT) && defined(SIMD_STL_HAS_AVX512BW_SUPPORT) && defined(SIMD_STL_HAS_AVX512DQ_SUPPORT)
  using simd_native = simd<arch::CpuFeature::AVX512VLBWDQ, _Type_, ymm256>;

#elif defined(SIMD_STL_HAS_AVX512VL_SUPPORT) && defined(SIMD_STL_HAS_AVX512BW_SUPPORT)
  using simd_native = simd<arch::CpuFeature::AVX512VLBW, _Type_, ymm256>;

#elif defined(SIMD_STL_HAS_AVX512VL_SUPPORT) && defined(SIMD_STL_HAS_AVX512DQ_SUPPORT)
  using simd_native = simd<arch::CpuFeature::AVX512VLDQ, _Type_, ymm256>;

#elif defined(SIMD_STL_HAS_AVX512VL_SUPPORT) && defined(SIMD_STL_HAS_AVX512F_SUPPORT)
  using simd_native = simd<arch::CpuFeature::AVX512VLF, _Type_, ymm256>;

#elif defined(SIMD_STL_HAS_AVX512BW_SUPPORT)
  using simd_native = simd<arch::CpuFeature::AVX512BW, _Type_, zmm512>;

#elif defined(SIMD_STL_HAS_AVX512DQ_SUPPORT)
  using simd_native = simd<arch::CpuFeature::AVX512DQ, _Type_, zmm512>;

#elif defined(SIMD_STL_HAS_AVX512F_SUPPORT)
  using simd_native = simd<arch::CpuFeature::AVX512F, _Type_, zmm512>;

#elif defined(SIMD_STL_HAS_AVX2_SUPPORT)
  using simd_native = simd<arch::CpuFeature::AVX2, _Type_, ymm256>;

#elif defined(SIMD_STL_HAS_SSE42_SUPPORT)
  using simd_native = simd<arch::CpuFeature::SSE42, _Type_, xmm128>;

#elif defined(SIMD_STL_HAS_SSE41_SUPPORT)
  using simd_native = simd<arch::CpuFeature::SSE41, _Type_, xmm128>;

#elif defined(SIMD_STL_HAS_SSSE3_SUPPORT)
  using simd_native = simd<arch::CpuFeature::SSSE3, _Type_, xmm128>;
  
#elif defined(SIMD_STL_HAS_SSE3_SUPPORT)
  using simd_native = simd<arch::CpuFeature::SSE3, _Type_, xmm128>;

#elif defined(SIMD_STL_HAS_SSE2_SUPPORT)
  using simd_native = simd<arch::CpuFeature::SSE2, _Type_, xmm128>;

#else 
  #error "Unsupported architecture" 

#endif


//#if defined(SIMD_STL_HAS_AVX512VBMI2_SUPPORT) 
//  using simd_native = simd<arch::CpuFeature::AVX512VBMI2, _Type_, zmm512>;
//
//#elif defined(SIMD_STL_HAS_AVX512VBMI_SUPPORT)
//  using simd_native = simd<arch::CpuFeature::AVX512VBMI, _Type_, zmm512>;

#if defined(SIMD_STL_FORCE_AVX512DQ) && defined(SIMD_STL_FORCE_AVX512BW)
  template <typename _Type_>
  using simd_forced = simd<arch::CpuFeature::AVX512BWDQ, _Type_, zmm512>;

#elif defined(SIMD_STL_FORCE_AVX512VL) && defined(SIMD_STL_FORCE_AVX512BW) && defined(SIMD_STL_FORCE_AVX512DQ)
  template <typename _Type_>\
  using simd_forced = simd<arch::CpuFeature::AVX512VLBWDQ, _Type_, ymm256>;

#elif defined(SIMD_STL_FORCE_AVX512VL) && defined(SIMD_STL_FORCE_AVX512BW)
  template <typename _Type_>
  using simd_forced = simd<arch::CpuFeature::AVX512VLBW, _Type_, ymm256>;

#elif defined(SIMD_STL_FORCE_AVX512VL) && defined(SIMD_STL_FORCE_AVX512DQ)
  template <typename _Type_>
  using simd_forced = simd<arch::CpuFeature::AVX512VLDQ, _Type_, ymm256>;

#elif defined(SIMD_STL_FORCE_AVX512VL) && defined(SIMD_STL_FORCE_AVX512F)
  template <typename _Type_>
  using simd_forced = simd<arch::CpuFeature::AVX512VLF, _Type_, ymm256>;

#elif defined(SIMD_STL_FORCE_AVX512BW)
  template <typename _Type_>
  using simd_forced = simd<arch::CpuFeature::AVX512BW, _Type_, zmm512>;

#elif defined(SIMD_STL_FORCE_AVX512DQ)
  template <typename _Type_>
  using simd_forced = simd<arch::CpuFeature::AVX512DQ, _Type_, zmm512>;

#elif defined(SIMD_STL_FORCE_AVX512F)
  template <typename _Type_>
  using simd_forced = simd<arch::CpuFeature::AVX512F, _Type_, zmm512>;

#elif defined(SIMD_STL_FORCE_AVX2)
  template <typename _Type_>
  using simd_forced = simd<arch::CpuFeature::AVX2, _Type_, ymm256>;

#elif defined(SIMD_STL_FORCE_SSE42)
  template <typename _Type_>
  using simd_forced = simd<arch::CpuFeature::SSE42, _Type_, xmm128>;

#elif defined(SIMD_STL_FORCE_SSE41)
  template <typename _Type_>
  using simd_forced = simd<arch::CpuFeature::SSE41, _Type_, xmm128>;

#elif defined(SIMD_STL_FORCE_SSSE3)
  template <typename _Type_>
  using simd_forced = simd<arch::CpuFeature::SSSE3, _Type_, xmm128>;
  
#elif defined(SIMD_STL_FORCE_SSE3)
  template <typename _Type_>
  using simd_forced = simd<arch::CpuFeature::SSE3, _Type_, xmm128>;

#elif defined(SIMD_STL_FORCE_SSE2)
  template <typename _Type_>
  using simd_forced = simd<arch::CpuFeature::SSE2, _Type_, xmm128>;

#endif

__SIMD_STL_DATAPAR_NAMESPACE_END
