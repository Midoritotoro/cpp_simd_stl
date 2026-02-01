#pragma once 

#include <simd_stl/arch/CpuFeature.h>
#include <src/simd_stl/type_traits/TypeCheck.h>
#include <src/simd_stl/type_traits/IsVirtualBaseOf.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_,
    class               _RegisterPolicy_>
class simd;

constexpr auto __xmm_width = sizeof(__m128);
constexpr auto __ymm_width = sizeof(__m256);
constexpr auto __zmm_width = sizeof(__m512);

struct xmm128 { 
	static constexpr auto __width = __xmm_width;
};

struct ymm256 {
	static constexpr auto __width = __ymm_width;
};

struct zmm512 {
	static constexpr auto __width = __zmm_width;
};


template <arch::CpuFeature _SimdGeneration_>
using __default_register_policy = std::conditional_t<
	arch::__is_xmm_v<_SimdGeneration_>,
	numeric::xmm128,
	std::conditional_t<
	arch::__is_ymm_v<_SimdGeneration_>,
	numeric::ymm256,
	std::conditional_t<
	arch::__is_zmm_v<_SimdGeneration_>,
	numeric::zmm512, void
	>
	>
>;

template <
	arch::CpuFeature    _SimdGeneration_,
	typename            _VectorElementType_>
constexpr int __vector_default_size = __default_register_policy<_SimdGeneration_>::__width;

template <
	arch::CpuFeature	_SimdGeneration_, 
	class				_RegisterPolicy_> 
constexpr bool __is_register_policy_for_generation_v = __vector_default_size<_SimdGeneration_, int> >= _RegisterPolicy_::__width;

template <class _Type_>
constexpr bool __is_intrin_type_v = type_traits::is_any_of_v<std::remove_cvref_t<_Type_>,
	__m128, __m128i, __m128d, __m256, __m256i, __m256d, __m512, __m512i, __m512d>;

template <typename _Element_>
constexpr bool __is_epi64_v =  
	((std::is_signed_v<_Element_> && !std::is_floating_point_v<_Element_>) 
		|| std::is_pointer_v<_Element_>
		|| std::is_same_v<_Element_, std::nullptr_t>) && sizeof(_Element_) == 8;

template <typename _Element_>
constexpr bool __is_epu64_v = 
	((std::is_unsigned_v<_Element_> && !std::is_floating_point_v<_Element_>) 
		|| std::is_pointer_v<_Element_> 
		|| std::is_same_v<_Element_, std::nullptr_t>) && sizeof(_Element_) == 8;

template <typename _Element_>
constexpr bool __is_epi32_v = 
	((std::is_signed_v<_Element_> && !std::is_floating_point_v<_Element_>) 
		|| std::is_pointer_v<_Element_>
		|| std::is_same_v<_Element_, std::nullptr_t>) && sizeof(_Element_) == 4;

template <typename _Element_>
constexpr bool __is_epu32_v = 
	((std::is_unsigned_v<_Element_> && !std::is_floating_point_v<_Element_>)
		|| std::is_pointer_v<_Element_>
		|| std::is_same_v<_Element_, std::nullptr_t>) && sizeof(_Element_) == 4;

template <typename _Element_>
constexpr bool __is_epi16_v = sizeof(_Element_) == 2 && std::is_signed_v<_Element_> && !std::is_floating_point_v<_Element_>;

template <typename _Element_>
constexpr bool __is_epu16_v = sizeof(_Element_) == 2 && std::is_unsigned_v<_Element_> && !std::is_floating_point_v<_Element_>;

template <typename _Element_>
constexpr bool __is_epi8_v  = sizeof(_Element_) == 1 && std::is_signed_v<_Element_> && !std::is_floating_point_v<_Element_>;

template <typename _Element_>
constexpr bool __is_epu8_v  = sizeof(_Element_) == 1 && std::is_unsigned_v<_Element_> && !std::is_floating_point_v<_Element_>;

template <typename _Element_>
constexpr bool __is_pd_v    = sizeof(_Element_) == 8 && type_traits::is_any_of_v<_Element_, double, long double>;

template <typename _Element_>
constexpr bool __is_ps_v    = sizeof(_Element_) == 4 && std::is_same_v<_Element_, float>;

#if !defined(__verify_register_policy) 
#  define __verify_register_policy(_Generation_, _Policy_) \
	static_assert(simd_stl::numeric::__is_register_policy_for_generation_v<_Generation_, _Policy_>, "Simd generation does not support register policy. ");
#endif // !defined(_VerifyRegisterPolicy)

template <
	class _BasicSimd_, 
	class = void>
struct __is_valid_basic_simd: 
	std::false_type
{};

template <class _BasicSimd_>
struct __is_valid_basic_simd<
    _BasicSimd_,
    std::void_t<simd<
        _BasicSimd_::__generation,
        typename _BasicSimd_::value_type,
        typename _BasicSimd_::policy_type>>> 
    : std::bool_constant<
        type_traits::is_virtual_base_of_v<
            simd<_BasicSimd_::__generation,
                       typename _BasicSimd_::value_type,
                       typename _BasicSimd_::policy_type>,
            _BasicSimd_> ||
        std::is_same_v<
            simd<_BasicSimd_::__generation,
                       typename _BasicSimd_::value_type,
                       typename _BasicSimd_::policy_type>,
            _BasicSimd_>> 
{};

template <class _BasicSimd_>
constexpr bool __is_valid_basic_simd_v = __is_valid_basic_simd<_BasicSimd_>::value;

template <
    class _VectorType_,
    bool _IsBasicSimd_	= __is_valid_basic_simd_v<_VectorType_>,
    bool _IsIntrin_		= __is_intrin_type_v<_VectorType_>>
struct __vector_element_t {
    using type = void;
};

template <class _VectorType_>
struct __vector_element_t<_VectorType_, false, true> {
    using type = std::conditional_t<type_traits::is_any_of_v<_VectorType_, __m128i, __m256i, __m512i>, int, 
		std::conditional_t<type_traits::is_any_of_v<_VectorType_, __m128d, __m256d, __m512d>, double, 
			std::conditional_t<type_traits::is_any_of_v<_VectorType_, __m128, __m256, __m512>, float, void>>>;
};

template <class _VectorType_>
struct __vector_element_t<_VectorType_, true, false> {
    using type = typename _VectorType_::value_type;
};

template <class _VectorType_>
using __vector_element_type = typename __vector_element_t<_VectorType_>::type;

template <
	class _VectorType_,
	bool _IsIntrin_		= __is_intrin_type_v<_VectorType_>,
	bool _IsBasicSimd_	= __is_valid_basic_simd_v<_VectorType_>>
struct __unwrapped_vector_t {
	using type = _VectorType_;
};

template <class _VectorType_>
struct __unwrapped_vector_t<_VectorType_, false, true> {
	using type = typename _VectorType_::vector_type;
};

template <class _VectorType_>
struct __unwrapped_vector_t<_VectorType_, true, false> {
	using type = _VectorType_;
};

template <class _VectorType_>
using __unwrapped_vector_type = typename __unwrapped_vector_t<_VectorType_>::type;

template <
	class		_BasicSimd_,
	typename	_ReturnType_,
	typename	_DesiredType_>
using __native_compare_return_type_helper = std::conditional_t<__is_intrin_type_v<_ReturnType_>,
	simd<_BasicSimd_::__generation, _DesiredType_, typename _BasicSimd_::policy_type>, _ReturnType_>;

template <
    class _RebindType_,
    class _VectorType_,
    bool _IsBasicSimd_  = __is_valid_basic_simd_v<_VectorType_>,
    bool _IsIntrin_     = __is_intrin_type_v<_VectorType_>>
struct __rebind_vector_element_t;

template <
    arch::CpuFeature	_ToSimdGeneration_,
    class               _RebindType_,
    class               _VectorType_,
    bool                _IsBasicSimd_ = __is_valid_basic_simd_v<_VectorType_>,
    bool                _IsIntrin_ = __is_intrin_type_v<_VectorType_>>
struct __rebind_vector_generation_t;

template <
    class _RebindType_,
    class _VectorType_>
using __rebind_vector_element_type = typename __rebind_vector_element_t<_RebindType_, _VectorType_>::type;

template <
    arch::CpuFeature	_ToSimdGeneration_,
    class               _RebindType_,
    class               _VectorType_>
using __rebind_vector_generation_type = typename __rebind_vector_generation_t<_ToSimdGeneration_, _RebindType_, _VectorType_>::type;


template <arch::CpuFeature _SimdGeneration_> 
constexpr inline bool __has_avx512f_support_v = static_cast<int>(_SimdGeneration_) >= static_cast<int>(arch::CpuFeature::AVX512F);

template <arch::CpuFeature _SimdGeneration_>
constexpr inline bool __has_avx512bw_support_v = static_cast<int>(_SimdGeneration_) == static_cast<int>(arch::CpuFeature::AVX512BW)
	|| static_cast<int>(_SimdGeneration_) == static_cast<int>(arch::CpuFeature::AVX512BWDQ)
	|| static_cast<int>(_SimdGeneration_) == static_cast<int>(arch::CpuFeature::AVX512VLBWDQ)
	|| static_cast<int>(_SimdGeneration_) == static_cast<int>(arch::CpuFeature::AVX512VLBW);

template <arch::CpuFeature _SimdGeneration_>
constexpr inline bool __has_avx512dq_support_v = static_cast<int>(_SimdGeneration_) == static_cast<int>(arch::CpuFeature::AVX512DQ)
	|| static_cast<int>(_SimdGeneration_) == static_cast<int>(arch::CpuFeature::AVX512BWDQ)
	|| static_cast<int>(_SimdGeneration_) == static_cast<int>(arch::CpuFeature::AVX512VLBWDQ)
	|| static_cast<int>(_SimdGeneration_) == static_cast<int>(arch::CpuFeature::AVX512VLDQ);

__SIMD_STL_NUMERIC_NAMESPACE_END
