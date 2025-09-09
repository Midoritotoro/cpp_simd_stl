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



__SIMD_STL_TYPE_TRAITS_NAMESPACE_END