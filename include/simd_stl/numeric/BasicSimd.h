#pragma once 

#include <simd_stl/compatibility/SimdCompatibility.h>
#include <src/simd_stl/type_traits/TypeTraits.h>

#include <simd_stl/arch/CpuFeature.h>



__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <arch::CpuFeature _SimdGeneration_>
constexpr inline bool __is_generation_supported_v = arch::Contains<_SimdGeneration_, arch::__supportedFeatures>::value;

template <arch::CpuFeature _SimdGeneration_>
using __deduce_simd_vector_type = std::conditional_t<arch::__is_xmm_v<_SimdGeneration_>, void>


template <
	typename			_Element_,
	arch::CpuFeature	_SimdGeneration_>
class basic_simd {
	static_assert(__is_generation_supported_v<_SimdGeneration_>);

public:
	using 
};


	


__SIMD_STL_NUMERIC_NAMESPACE_END
