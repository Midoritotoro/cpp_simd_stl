#pragma once 

#include <simd_stl/arch/CpuFeature.h>
#include <src/simd_stl/type_traits/TypeCheck.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

constexpr auto _XmmWidth = sizeof(__m128);
constexpr auto _YmmWidth = sizeof(__m256);
constexpr auto _ZmmWidth = sizeof(__m512);

struct xmm128 { 
	static constexpr auto _Width = _XmmWidth;
};

struct ymm256 {
	static constexpr auto _Width = _YmmWidth;
};

struct zmm512 {
	static constexpr auto _Width = _ZmmWidth;
};

template <arch::CpuFeature _SimdGeneration_>
using _DefaultRegisterPolicy = std::conditional_t<
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
constexpr int _Vector_default_size = numeric::_DefaultRegisterPolicy<_SimdGeneration_>::_Width;

template <
	arch::CpuFeature	_SimdGeneration_, 
	class				_RegisterPolicy_> 
constexpr bool _Is_register_policy_for_generation_v = _Vector_default_size<_SimdGeneration_, int> >= _RegisterPolicy_::_Width;

template <class _Type_>
constexpr bool _Is_intrin_type_v = type_traits::is_any_of_v<std::remove_cvref_t<_Type_>,
	__m128, __m128i, __m128d, __m256, __m256i, __m256d, __m512, __m512i, __m512d>;

template <typename _Element_>
constexpr bool _Is_epi64_v =  
	((std::is_signed_v<_Element_> && !std::is_floating_point_v<_Element_>) 
		|| std::is_pointer_v<_Element_>
		|| std::is_same_v<_Element_, std::nullptr_t>) && sizeof(_Element_) == 8;

template <typename _Element_>
constexpr bool _Is_epu64_v = 
	((std::is_unsigned_v<_Element_> && !std::is_floating_point_v<_Element_>) 
		|| std::is_pointer_v<_Element_> 
		|| std::is_same_v<_Element_, std::nullptr_t>) && sizeof(_Element_) == 8;

template <typename _Element_>
constexpr bool _Is_epi32_v = 
	((std::is_signed_v<_Element_> && !std::is_floating_point_v<_Element_>) 
		|| std::is_pointer_v<_Element_>
		|| std::is_same_v<_Element_, std::nullptr_t>) && sizeof(_Element_) == 4;

template <typename _Element_>
constexpr bool _Is_epu32_v = 
	((std::is_unsigned_v<_Element_> && !std::is_floating_point_v<_Element_>)
		|| std::is_pointer_v<_Element_>
		|| std::is_same_v<_Element_, std::nullptr_t>) && sizeof(_Element_) == 4;

template <typename _Element_>
constexpr bool _Is_epi16_v = sizeof(_Element_) == 2 && std::is_signed_v<_Element_> && !std::is_floating_point_v<_Element_>;

template <typename _Element_>
constexpr bool _Is_epu16_v = sizeof(_Element_) == 2 && std::is_unsigned_v<_Element_> && !std::is_floating_point_v<_Element_>;

template <typename _Element_>
constexpr bool _Is_epi8_v  = sizeof(_Element_) == 1 && std::is_signed_v<_Element_> && !std::is_floating_point_v<_Element_>;

template <typename _Element_>
constexpr bool _Is_epu8_v  = sizeof(_Element_) == 1 && std::is_unsigned_v<_Element_> && !std::is_floating_point_v<_Element_>;

template <typename _Element_>
constexpr bool _Is_pd_v    = sizeof(_Element_) == 8 && type_traits::is_any_of_v<_Element_, double, long double>;

template <typename _Element_>
constexpr bool _Is_ps_v    = sizeof(_Element_) == 4 && std::is_same_v<_Element_, float>;

#if !defined(_VerifyRegisterPolicy) 
#  define _VerifyRegisterPolicy(_Generation_, _Policy_) \
	static_assert(simd_stl::numeric::_Is_register_policy_for_generation_v<_Generation_, _Policy_>, "Simd generation does not support register policy. ");
#endif // !defined(_VerifyRegisterPolicy)

__SIMD_STL_NUMERIC_NAMESPACE_END
