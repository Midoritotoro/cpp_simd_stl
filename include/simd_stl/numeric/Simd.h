#pragma once 

#include <simd_stl/numeric/BasicSimd.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
	typename	_Type_, 
	class		_RegisterPolicy_ = xmm128>
using simd_sse2 = basic_simd<arch::CpuFeature::SSE2, _Type_, _RegisterPolicy_>;

template <
	typename	_Type_,
	class		_RegisterPolicy_ = xmm128>
using simd_sse3 = basic_simd<arch::CpuFeature::SSE3, _Type_, _RegisterPolicy_>;

template <
	typename	_Type_,
	class		_RegisterPolicy_ = xmm128>
using simd_ssse3 = basic_simd<arch::CpuFeature::SSSE3, _Type_, _RegisterPolicy_>;

template <
	typename	_Type_,
	class		_RegisterPolicy_ = xmm128>
using simd_sse41 = basic_simd<arch::CpuFeature::SSE41, _Type_, _RegisterPolicy_>;

template <
	typename	_Type_,
	class		_RegisterPolicy_ = xmm128>
using simd_sse42 = basic_simd<arch::CpuFeature::SSE42, _Type_, _RegisterPolicy_>;

template <
	typename	_Type_,
	class		_RegisterPolicy_ = ymm256>
using simd_avx2 = basic_simd<arch::CpuFeature::AVX2, _Type_, _RegisterPolicy_>;

template <
	typename	_Type_,
	class		_RegisterPolicy_ = zmm512>
using simd_avx512f = basic_simd<arch::CpuFeature::AVX512F, _Type_, _RegisterPolicy_>;

template <
	typename	_Type_,
	class		_RegisterPolicy_ = zmm512>
using simd_avx512bw = basic_simd<arch::CpuFeature::AVX512BW, _Type_, _RegisterPolicy_>;

template <
	typename	_Type_,
	class		_RegisterPolicy_ = zmm512>
using simd_avx512dq = basic_simd<arch::CpuFeature::AVX512DQ, _Type_, _RegisterPolicy_>;

__SIMD_STL_NUMERIC_NAMESPACE_END
