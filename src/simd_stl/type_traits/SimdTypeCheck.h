#pragma once 

#include <simd_stl/arch/CpuFeature.h>
#include <simd_stl/compatibility/SimdCompatibility.h>

#include <src/simd_stl/type_traits/IntegralProperties.h>
#include <src/simd_stl/type_traits/TypeCheck.h>


__SIMD_STL_TYPE_TRAITS_NAMESPACE_BEGIN

template <arch::CpuFeature _SimdGeneration_>
constexpr inline bool __is_generation_supported_v = arch::Contains<_SimdGeneration_, __ymm_features, __xmm_features, __zmm_features>::value;

template <typename _VectorElementType_>
constexpr inline bool __is_vector_type_supported_v = std::is_arithmetic_v<_VectorElementType_>;

template <>
constexpr inline bool __is_vector_type_supported_v<bool> = false;

template <
    arch::CpuFeature  _SimdGeneration_,
    typename          _VectorElementType_>
using __deduce_simd_vector_type = std::conditional_t <
    arch::__is_zmm_v<_SimdGeneration_>,
        std::conditional_t<
            type_traits::is_any_of_v<_VectorElementType_, double, long double>, __m512d,
                std::conditional_t<
                    std::is_same_v<_VectorElementType_, float>, __m512,
                        std::conditional_t<
                            type_traits::is_nonbool_integral_v<_VectorElementType_>, __m512i, void>>>,
    std::conditional_t<
        arch::__is_ymm_v<_SimdGeneration_>,
            std::conditional_t<
                type_traits::is_any_of_v<_VectorElementType_, double, long double>, __m256d,
                    std::conditional_t<
                         std::is_same_v<_VectorElementType_, float>, __m256,
                            std::conditional_t<
                                type_traits::is_nonbool_integral_v<_VectorElementType_>, __m256i, void>>>,
    std::conditional_t<
        arch::__is_xmm_v<_SimdGeneration_>,
            std::conditional_t<
                type_traits::is_any_of_v<_VectorElementType_, double, long double>, __m128d,
                    std::conditional_t<
                        std::is_same_v<_VectorElementType_, float>, __m128,
                            std::conditional_t<
                                type_traits::is_nonbool_integral_v<_VectorElementType_>, __m128i, void>>>,
    void>>>;


template <uint8 size>
using __deduce_simd_mask_type_helper =
	std::conditional_t<size <= 1, uint8,
		std::conditional_t<size <= 2, uint16,
			std::conditional_t<size <= 4, uint32,
				std::conditional_t<size <= 8, uint64, void>>>>;

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_>
using __deduce_simd_mask_type =  __deduce_simd_mask_type_helper<(sizeof(type_traits::__deduce_simd_vector_type<_SimdGeneration_, _Element_>) / sizeof(_Element_))>;


template <sizetype _VectorLength_>
using __deduce_simd_shuffle_mask_type = std::conditional_t<size == 2, uint8,
		std::conditional_t<size == 2, uint8,
			std::conditional_t<size == 8, uint32,
				std::conditional_t<size <= 16, uint64, void>>>>;


__SIMD_STL_TYPE_TRAITS_NAMESPACE_END