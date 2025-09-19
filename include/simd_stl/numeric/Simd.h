#pragma once 

#include <simd_stl/numeric/BasicSimd.h>

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

// ==============================================================
//								SSE2
// ==============================================================

using simd_sse2_i8 = basic_simd<arch::CpuFeature::SSE2, int8>;
using simd_sse2_u8 = basic_simd<arch::CpuFeature::SSE2, uint8>;

using simd_sse2_i16 = basic_simd<arch::CpuFeature::SSE2, int16>;
using simd_sse2_u16 = basic_simd<arch::CpuFeature::SSE2, uint16>;

using simd_sse2_i32 = basic_simd<arch::CpuFeature::SSE2, int32>;
using simd_sse2_u32 = basic_simd<arch::CpuFeature::SSE2, uint32>;

using simd_sse2_i64 = basic_simd<arch::CpuFeature::SSE2, int64>;
using simd_sse2_u64 = basic_simd<arch::CpuFeature::SSE2, uint64>;

using simd_sse2_f = basic_simd<arch::CpuFeature::SSE2, float>;
using simd_sse2_d = basic_simd<arch::CpuFeature::SSE2, double>;


// ==============================================================
//								SSE3
// ==============================================================


using simd_sse3_i8 = basic_simd<arch::CpuFeature::SSE3, int8>;
using simd_sse3_u8 = basic_simd<arch::CpuFeature::SSE3, uint8>;

using simd_sse3_i16 = basic_simd<arch::CpuFeature::SSE3, int16>;
using simd_sse3_u16 = basic_simd<arch::CpuFeature::SSE3, uint16>;

using simd_sse3_i32 = basic_simd<arch::CpuFeature::SSE3, int32>;
using simd_sse3_u32 = basic_simd<arch::CpuFeature::SSE3, uint32>;

using simd_sse3_i64 = basic_simd<arch::CpuFeature::SSE3, int64>;
using simd_sse3_u64 = basic_simd<arch::CpuFeature::SSE3, uint64>;

using simd_sse3_f = basic_simd<arch::CpuFeature::SSE3, float>;
using simd_sse3_d = basic_simd<arch::CpuFeature::SSE3, double>;


// ==============================================================
//								SSSE3
// ==============================================================


using simd_ssse3_i8 = basic_simd<arch::CpuFeature::SSSE3, int8>;
using simd_ssse3_u8 = basic_simd<arch::CpuFeature::SSSE3, uint8>;

using simd_ssse3_i16 = basic_simd<arch::CpuFeature::SSSE3, int16>;
using simd_ssse3_u16 = basic_simd<arch::CpuFeature::SSSE3, uint16>;

using simd_ssse3_i32 = basic_simd<arch::CpuFeature::SSSE3, int32>;
using simd_ssse3_u32 = basic_simd<arch::CpuFeature::SSSE3, uint32>;

using simd_ssse3_i64 = basic_simd<arch::CpuFeature::SSSE3, int64>;
using simd_ssse3_u64 = basic_simd<arch::CpuFeature::SSSE3, uint64>;

using simd_ssse3_f = basic_simd<arch::CpuFeature::SSSE3, float>;
using simd_ssse3_d = basic_simd<arch::CpuFeature::SSSE3, double>;


// ==============================================================
//								SSE4.1
// ==============================================================


using simd_sse41_i8 = basic_simd<arch::CpuFeature::SSE41, int8>;
using simd_sse41_u8 = basic_simd<arch::CpuFeature::SSE41, uint8>;

using simd_sse41_i16 = basic_simd<arch::CpuFeature::SSE41, int16>;
using simd_sse41_u16 = basic_simd<arch::CpuFeature::SSE41, uint16>;

using simd_sse41_i32 = basic_simd<arch::CpuFeature::SSE41, int32>;
using simd_sse41_u32 = basic_simd<arch::CpuFeature::SSE41, uint32>;

using simd_sse41_i64 = basic_simd<arch::CpuFeature::SSE41, int64>;
using simd_sse41_u64 = basic_simd<arch::CpuFeature::SSE41, uint64>;

using simd_sse41_f = basic_simd<arch::CpuFeature::SSE41, float>;
using simd_sse41_d = basic_simd<arch::CpuFeature::SSE41, double>;


// ==============================================================
//								SSE4.2
// ==============================================================


using simd_sse42_i8 = basic_simd<arch::CpuFeature::SSE42, int8>;
using simd_sse42_u8 = basic_simd<arch::CpuFeature::SSE42, uint8>;

using simd_sse42_i16 = basic_simd<arch::CpuFeature::SSE42, int16>;
using simd_sse42_u16 = basic_simd<arch::CpuFeature::SSE42, uint16>;

using simd_sse42_i32 = basic_simd<arch::CpuFeature::SSE42, int32>;
using simd_sse42_u32 = basic_simd<arch::CpuFeature::SSE42, uint32>;

using simd_sse42_i64 = basic_simd<arch::CpuFeature::SSE42, int64>;
using simd_sse42_u64 = basic_simd<arch::CpuFeature::SSE42, uint64>;

using simd_sse42_f = basic_simd<arch::CpuFeature::SSE42, float>;
using simd_sse42_d = basic_simd<arch::CpuFeature::SSE42, double>;


// ==============================================================
//								AVX
// ==============================================================


using simd_avx_i8 = basic_simd<arch::CpuFeature::AVX, int8>;
using simd_avx_u8 = basic_simd<arch::CpuFeature::AVX, uint8>;

using simd_avx_i16 = basic_simd<arch::CpuFeature::AVX, int16>;
using simd_avx_u16 = basic_simd<arch::CpuFeature::AVX, uint16>;

using simd_avx_i32 = basic_simd<arch::CpuFeature::AVX, int32>;
using simd_avx_u32 = basic_simd<arch::CpuFeature::AVX, uint32>;

using simd_avx_i64 = basic_simd<arch::CpuFeature::AVX, int64>;
using simd_avx_u64 = basic_simd<arch::CpuFeature::AVX, uint64>;

using simd_avx_f = basic_simd<arch::CpuFeature::AVX, float>;
using simd_avx_d = basic_simd<arch::CpuFeature::AVX, double>;


// ==============================================================
//								AVX2
// ==============================================================


using simd_avx2_i8 = basic_simd<arch::CpuFeature::AVX2, int8>;
using simd_avx2_u8 = basic_simd<arch::CpuFeature::AVX2, uint8>;

using simd_avx2_i16 = basic_simd<arch::CpuFeature::AVX2, int16>;
using simd_avx2_u16 = basic_simd<arch::CpuFeature::AVX2, uint16>;

using simd_avx2_i32 = basic_simd<arch::CpuFeature::AVX2, int32>;
using simd_avx2_u32 = basic_simd<arch::CpuFeature::AVX2, uint32>;

using simd_avx2_i64 = basic_simd<arch::CpuFeature::AVX2, int64>;
using simd_avx2_u64 = basic_simd<arch::CpuFeature::AVX2, uint64>;

using simd_avx2_f = basic_simd<arch::CpuFeature::AVX2, float>;
using simd_avx2_d = basic_simd<arch::CpuFeature::AVX2, double>;


// ==============================================================
//							AVX512F
// ==============================================================


using simd_avx512f_i8 = basic_simd<arch::CpuFeature::AVX512F, int8>;
using simd_avx512f_u8 = basic_simd<arch::CpuFeature::AVX512F, uint8>;

using simd_avx512f_i16 = basic_simd<arch::CpuFeature::AVX512F, int16>;
using simd_avx512f_u16 = basic_simd<arch::CpuFeature::AVX512F, uint16>;

using simd_avx512f_i32 = basic_simd<arch::CpuFeature::AVX512F, int32>;
using simd_avx512f_u32 = basic_simd<arch::CpuFeature::AVX512F, uint32>;

using simd_avx512f_i64 = basic_simd<arch::CpuFeature::AVX512F, int64>;
using simd_avx512f_u64 = basic_simd<arch::CpuFeature::AVX512F, uint64>;

using simd_avx512f_f = basic_simd<arch::CpuFeature::AVX512F, float>;
using simd_avx512f_d = basic_simd<arch::CpuFeature::AVX512F, double>;


// ==============================================================
//							AVX512BW
// ==============================================================


using simd_avx512bw_i8 = basic_simd<arch::CpuFeature::AVX512BW, int8>;
using simd_avx512bw_u8 = basic_simd<arch::CpuFeature::AVX512BW, uint8>;

using simd_avx512bw_i16 = basic_simd<arch::CpuFeature::AVX512BW, int16>;
using simd_avx512bw_u16 = basic_simd<arch::CpuFeature::AVX512BW, uint16>;

using simd_avx512bw_i32 = basic_simd<arch::CpuFeature::AVX512BW, int32>;
using simd_avx512bw_u32 = basic_simd<arch::CpuFeature::AVX512BW, uint32>;

using simd_avx512bw_i64 = basic_simd<arch::CpuFeature::AVX512BW, int64>;
using simd_avx512bw_u64 = basic_simd<arch::CpuFeature::AVX512BW, uint64>;

using simd_avx512bw_f = basic_simd<arch::CpuFeature::AVX512BW, float>;
using simd_avx512bw_d = basic_simd<arch::CpuFeature::AVX512BW, double>;


__SIMD_STL_NUMERIC_NAMESPACE_END
