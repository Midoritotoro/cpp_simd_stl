#pragma once 

#include <simd_stl/arch/CpuFeature.h>
#include <simd_stl/compatibility/SimdCompatibility.h>

#include <src/simd_stl/type_traits/IntegralProperties.h>
#include <src/simd_stl/type_traits/TypeCheck.h>

#include <src/simd_stl/numeric/SimdIntegralTypesCheck.h>


__SIMD_STL_TYPE_TRAITS_NAMESPACE_BEGIN

template <arch::CpuFeature _SimdGeneration_>
constexpr inline bool __is_generation_supported_v =
    arch::Contains<_SimdGeneration_, __ymm_features, __xmm_features, __zmm_features>::value;

template <typename _VectorElementType_>
constexpr inline bool __is_pointer_decay_v = std::is_pointer_v<std::decay_t<_VectorElementType_>>;

template <typename _VectorElementType_>
constexpr inline bool __is_vector_type_supported_v =
    std::is_arithmetic_v<std::decay_t<_VectorElementType_>> ||
    __is_pointer_decay_v<_VectorElementType_> || 
    std::is_same_v<std::decay_t<_VectorElementType_>, std::nullptr_t>;

template <>
constexpr inline bool __is_vector_type_supported_v<bool> = false; // запрет для bool как скалярного элемента

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _VectorElementType_,
    class               _RegisterPolicy_ = numeric::_DefaultRegisterPolicy<_SimdGeneration_>>
struct __deduce_simd_vector_type__ {
private:
    using T = std::decay_t<_VectorElementType_>;

    static constexpr bool is_fp64 = type_traits::is_any_of_v<T, double, long double> || (std::is_same_v<T, std::nullptr_t> && sizeof(std::nullptr_t) == 8);
    static constexpr bool is_fp32 = std::is_same_v<T, float>;
    static constexpr bool is_int  = type_traits::is_nonbool_integral_v<T> || (std::is_same_v<T, std::nullptr_t> && sizeof(std::nullptr_t) == 4);
    static constexpr bool is_ptr  = __is_pointer_decay_v<_VectorElementType_>;
    static constexpr bool use_i   = is_int || is_ptr;

public:
    using type =
        std::conditional_t<
            std::is_same_v<_RegisterPolicy_, numeric::zmm512>,
                std::conditional_t<
                    is_fp64, __m512d,
                    std::conditional_t<
                        is_fp32, __m512,
                        std::conditional_t<
                            use_i,   __m512i,
                                     void>>>,
        std::conditional_t<
            std::is_same_v<_RegisterPolicy_, numeric::ymm256>,
                std::conditional_t<
                    is_fp64, __m256d,
                    std::conditional_t<
                        is_fp32, __m256,
                        std::conditional_t<
                            use_i,   __m256i,
                                     void>>>,
        std::conditional_t<
            std::is_same_v<_RegisterPolicy_, numeric::xmm128>,
                std::conditional_t<
                    is_fp64, __m128d,
                    std::conditional_t<
                        is_fp32, __m128,
                        std::conditional_t<
                            use_i,   __m128i,
                                     void>>>,
        void>>>;
};

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _VectorElementType_,
    class               _RegisterPolicy_ = numeric::_DefaultRegisterPolicy<_SimdGeneration_>>
using __deduce_simd_vector_type =
    typename __deduce_simd_vector_type__<_SimdGeneration_, _VectorElementType_, _RegisterPolicy_>::type;

template <sizetype size>
using __deduce_simd_shuffle_mask_type_helper = std::conditional_t<size <= 2, uint8,
		std::conditional_t<size <= 4, uint8,
			std::conditional_t<size <= 8, uint32,
				std::conditional_t<size <= 16, uint64, void>>>>;


template <sizetype size>
using __deduce_simd_mask_type_helper = std::conditional_t<size <= 8, uint8,
		std::conditional_t<size <= 16, uint16,
			std::conditional_t<size <= 32, uint32,
				std::conditional_t<size <= 64, uint64, void>>>>;

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
    class               _RegisterPolicy_ = numeric::_DefaultRegisterPolicy<_SimdGeneration_>>
using __deduce_simd_mask_type = __deduce_simd_mask_type_helper<(sizeof(type_traits::__deduce_simd_vector_type<_SimdGeneration_, _Element_, _RegisterPolicy_>) / sizeof(_Element_))>;

template <
	arch::CpuFeature	_SimdGeneration_,
	typename			_Element_,
    class               _RegisterPolicy_ = numeric::_DefaultRegisterPolicy<_SimdGeneration_>>
using __deduce_simd_shuffle_mask_type = __deduce_simd_shuffle_mask_type_helper<(sizeof(type_traits::__deduce_simd_vector_type<_SimdGeneration_, _Element_, _RegisterPolicy_>) / sizeof(_Element_))>;


template <arch::CpuFeature _SimdGeneration_>
constexpr bool is_native_mask_load_supported_v = std::conjunction_v<
    !arch::__is_xmm_v<_SimdGeneration_>,
    arch::__is_ymm_v<_SimdGeneration_>,
    arch::__is_zmm_v<_SimdGeneration_>
>;

template <arch::CpuFeature _SimdGeneration_>
constexpr bool is_native_mask_store_supported_v = is_native_mask_load_supported_v<_SimdGeneration_>;

template <arch::CpuFeature _SimdGeneration_> 
constexpr bool is_streaming_load_supported_v = 
    (static_cast<int8>(_SimdGeneration_) == static_cast<int8>(arch::CpuFeature::SSE41) ||
        static_cast<int8>(_SimdGeneration_) == static_cast<int8>(arch::CpuFeature::SSE42)) || 
    (static_cast<int8>(_SimdGeneration_) == static_cast<int8>(arch::CpuFeature::AVX2)) || 
    (static_cast<int8>(_SimdGeneration_) >= static_cast<int8>(arch::CpuFeature::AVX512F));

template <arch::CpuFeature _SimdGeneration_>
constexpr bool is_streaming_store_supported_v =
    (static_cast<int8>(_SimdGeneration_) >= static_cast<int8>(arch::CpuFeature::SSE2)       &&
        static_cast<int8>(_SimdGeneration_) <= static_cast<int8>(arch::CpuFeature::SSE42))  ||
    (static_cast<int8>(_SimdGeneration_) == static_cast<int8>(arch::CpuFeature::AVX)        ||
        static_cast<int8>(_SimdGeneration_) == static_cast<int8>(arch::CpuFeature::AVX2))   ||
    (static_cast<int8>(_SimdGeneration_) >= static_cast<int8>(arch::CpuFeature::AVX512F));

template <arch::CpuFeature _SimdGeneration_> 
constexpr bool is_streaming_supported_v = 
    is_streaming_load_supported_v<_SimdGeneration_> &&
    is_streaming_store_supported_v<_SimdGeneration_>;

template <arch::CpuFeature _SimdGeneration_> 
constexpr bool is_zeroupper_required_v =
    static_cast<int8>(_SimdGeneration_) == static_cast<int8>(arch::CpuFeature::AVX2) ||
    static_cast<int8>(_SimdGeneration_) == static_cast<int8>(arch::CpuFeature::AVX);

template <
    arch::CpuFeature _SimdGenerationFirst_,
    arch::CpuFeature _SimdGenerationSecond_>
constexpr bool is_simd_feature_superior_v = (static_cast<uint8>(_SimdGenerationFirst_) > static_cast<uint8>(_SimdGenerationSecond_));


template <
    class _BasicSimdFrom_,
    class _BasicSimdTo_>
using deduce_superior_basic_simd_type = std::conditional_t<
        is_simd_feature_superior_v<
            _BasicSimdFrom_::_Generation,
            _BasicSimdTo_::_Generation>,
        _BasicSimdFrom_,
        _BasicSimdTo_
    >;

__SIMD_STL_TYPE_TRAITS_NAMESPACE_END