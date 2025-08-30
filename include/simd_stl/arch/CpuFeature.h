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
	AVX512VL,
};

template <CpuFeature feature>
constexpr const char* featureStringify() {
    if		constexpr (feature == CpuFeature::None)		return "None";
    else if constexpr (feature == CpuFeature::SSE)      return "SSE";
    else if constexpr (feature == CpuFeature::SSE2)     return "SSE2";
    else if constexpr (feature == CpuFeature::SSE3)     return "SSE3";
    else if constexpr (feature == CpuFeature::SSSE3)    return "SSSE3";
    else if constexpr (feature == CpuFeature::SSE41)    return "SSE41";
    else if constexpr (feature == CpuFeature::SSE42)    return "SSE42";
    else if constexpr (feature == CpuFeature::AVX)      return "AVX";
    else if constexpr (feature == CpuFeature::AVX2)     return "AVX2";
    else if constexpr (feature == CpuFeature::AVX512F)  return "AVX512F";
    else if constexpr (feature == CpuFeature::AVX512BW) return "AVX512BW";
    else if constexpr (feature == CpuFeature::AVX512CD) return "AVX512CD";
    else if constexpr (feature == CpuFeature::AVX512ER) return "AVX512ER";
    else if constexpr (feature == CpuFeature::AVX512PF) return "AVX512PF";
    else if constexpr (feature == CpuFeature::AVX512VL) return "AVX512VL";
    else {
        static_assert(false, "");
        return "Unknown";
    }
}


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
