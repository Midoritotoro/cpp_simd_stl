#pragma once 

#include <simd_stl/Types.h>


__SIMD_STL_ARCH_NAMESPACE_BEGIN

enum class CpuFeature : simd_stl::uchar {
	None,
	SSE,
	SSE2,
	SSE3,
	SSSE3,
	SSE41,
	SSE42,
	AVX,
	AVX2,
	AVX512F,
	AVX512BW,
	AVX512DQ,
	AVX512BWDQ,			// AVX512BW + AVX512DQ
	AVX512VLBWDQ,		// AVX512VL + AVX512BW + AVX512DQ
	AVX512VLDQ,			// AVX512VL + AVX512DQ
	AVX512VLBW,			// AVX512VL + AVX512BW
	AVX512VLF,			// AVX512VL + AVX512F
	AVX512VBMI,			// AVX512VBMI + AVX512BW
	AVX512VBMI2,		// AVX512VBMI2 + AVX512BW
	AVX512VBMIVL,		// AVX512VBMI + AVX512VL
	AVX512VBMI2VL,		// AVX512VBMI2 + AVX512VL
	AVX512VBMIDQ,		// AVX512VBMI + AVX512BW + AVX512DQ
	AVX512VBMI2DQ,		// AVX512VBMI2 + AVX512BW + AVX512DQ
	AVX512VBMIVLDQ,		// AVX512VBMI + AVX512BW + AVX512DQ + AVX512VL
	AVX512VBMI2VLDQ,	// AVX512VBMI2 + AVX512BW + AVX512DQ + AVX512VL
};

template <
	CpuFeature	_Feature_,
	CpuFeature	Candidate,
	typename	Enable = void>
struct __is_in_list_helper:
	std::false_type
{};

template <
	CpuFeature _Feature_,
	CpuFeature _Candidate_>
struct __is_in_list_helper<
	_Feature_, _Candidate_,
	std::enable_if_t<(_Feature_ == _Candidate_)>>:
		std::true_type
{};

template <
	CpuFeature		_Feature_,
	CpuFeature ...	_List_>
struct __contains {
	static constexpr bool value = (__is_in_list_helper<_Feature_, _List_>::value || ...);
};

#define __xmm_features arch::CpuFeature::SSE, arch::CpuFeature::SSE2, arch::CpuFeature::SSE3, arch::CpuFeature::SSSE3, arch::CpuFeature::SSE41, arch::CpuFeature::SSE42
#define __ymm_features arch::CpuFeature::AVX, arch::CpuFeature::AVX2
#define __zmm_features arch::CpuFeature::AVX512F, arch::CpuFeature::AVX512BW, arch::CpuFeature::AVX512BWDQ, \
	arch::CpuFeature::AVX512DQ, arch::CpuFeature::AVX512VLDQ, arch::CpuFeature::AVX512VLBW, arch::CpuFeature::AVX512VLF, \
	arch::CpuFeature::AVX512VLBWDQ, arch::CpuFeature::AVX512VBMI, arch::CpuFeature::AVX512VBMI2, arch::CpuFeature::AVX512VBMIDQ,\
	arch::CpuFeature::AVX512VBMI2DQ, arch::CpuFeature::AVX512VBMIVL, arch::CpuFeature::AVX512VBMI2VL, arch::CpuFeature::AVX512VBMIVLDQ, arch::CpuFeature::AVX512VBMI2VLDQ

template <arch::CpuFeature _SimdGeneration_> 
constexpr inline bool __is_xmm_v = __contains<_SimdGeneration_, __xmm_features>::value;

template <arch::CpuFeature _SimdGeneration_>
constexpr inline bool __is_ymm_v = __contains<_SimdGeneration_, __ymm_features>::value;

template <arch::CpuFeature _SimdGeneration_>
constexpr inline bool __is_zmm_v = __contains<_SimdGeneration_, __zmm_features>::value;

__SIMD_STL_ARCH_NAMESPACE_END
