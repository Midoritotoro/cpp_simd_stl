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
	AVX512CD,
	AVX512ER,
	AVX512PF,
	AVX512VL
};

static inline auto constexpr __supportedFeatures = {
	CpuFeature::SSE,
	CpuFeature::SSE2,
	CpuFeature::SSE3,
	CpuFeature::SSSE3,
	CpuFeature::SSE41,
	CpuFeature::SSE42,
	CpuFeature::AVX,
	CpuFeature::AVX2,
	CpuFeature::AVX512F,
	CpuFeature::AVX512BW,
	CpuFeature::AVX512CD,
	CpuFeature::AVX512ER,
	CpuFeature::AVX512PF,
	CpuFeature::AVX512VL
};

template <
	CpuFeature	Feature,
	CpuFeature	Candidate,
	typename	Enable = void>
struct IsInListHelper :
	std::false_type
{
};

template <
	CpuFeature Feature,
	CpuFeature Candidate>
struct IsInListHelper<
	Feature, Candidate,
	std::enable_if_t<(Feature == Candidate)>> :
	std::true_type
{};

template <
	CpuFeature		Feature,
	CpuFeature ...	List>
struct Contains {
	static constexpr bool value = (IsInListHelper<Feature, List>::value || ...);
};


template <arch::CpuFeature _SimdGeneration_> 
constexpr inline bool __is_xmm_v = Contains<_SimdGeneration_, {	
			CpuFeature::SSE, CpuFeature::SSE2, CpuFeature::SSE3,
			CpuFeature::SSSE3, CpuFeature::SSE41, CpuFeature::SSE42
		}
	>::value;

template <arch::CpuFeature _SimdGeneration_>
constexpr inline bool __is_ymm_v = Contains < _SimdGeneration_, {
			CpuFeature::AVX, CpuFeature::AVX2,
		}
	>::value;

template <arch::CpuFeature _SimdGeneration_>
constexpr inline bool __is_zmm_v = Contains<_SimdGeneration_, {
				CpuFeature::AVX512F, CpuFeature::AVX512BW, CpuFeature::AVX512CD,
				CpuFeature::AVX512ER, CpuFeature::AVX512PF, CpuFeature::AVX512VL
			}
	>::value;

#ifndef SIMD_STL_STATIC_VERIFY_CPU_FEATURE
#define SIMD_STL_STATIC_VERIFY_CPU_FEATURE(current, failureLogPrefix, ...)                      \
    static_assert(																				\
        simd_stl::arch::Contains<current, __VA_ARGS__>::value,                                  \
        failureLogPrefix": " #current " not found in supported features (" #__VA_ARGS__ ")")
#endif // SIMD_STL_STATIC_VERIFY_CPU_FEATURE


#if !defined(SIMD_STL_DECLARE_CPU_FEATURE_GUARDED_FUNCTION)
#  define SIMD_STL_DECLARE_CPU_FEATURE_GUARDED_FUNCTION(functionDeclaration, featureVariableName, failureLogPrefix, returnValue, ...)		\
     functionDeclaration { SIMD_STL_STATIC_VERIFY_CPU_FEATURE(featureVariableName, failureLogPrefix, __VA_ARGS__); return returnValue; }
#endif // !defined(SIMD_STL_DECLARE_CPU_FEATURE_GUARDED_FUNCTION)

#ifndef SIMD_STL_DECLARE_CPU_FEATURE_GUARDED_CLASS
#  define SIMD_STL_DECLARE_CPU_FEATURE_GUARDED_CLASS(_class, featureVariableName, failureLogPrefix, ...)	\
     _class { SIMD_STL_STATIC_VERIFY_CPU_FEATURE(featureVariableName, failureLogPrefix, __VA_ARGS__); }
#endif // SIMD_STL_DECLARE_CPU_FEATURE_GUARDED_CLASS

__SIMD_STL_ARCH_NAMESPACE_END
