#pragma once 

#include <simd_stl/arch/CpuFeature.h>
#include <src/simd_stl/type_traits/TypeTraits.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

struct xmm128 {};
struct ymm256 {};
struct zmm512 {};

template <typename _Element_>
constexpr bool is_epi64_v =  
	((std::is_signed_v<_Element_> && !std::is_floating_point_v<_Element_>) 
		|| std::is_pointer_v<_Element_>
		|| std::is_same_v<_Element_, std::nullptr_t>) && sizeof(_Element_) == 8;

template <typename _Element_>
constexpr bool is_epu64_v = 
	((std::is_unsigned_v<_Element_> && !std::is_floating_point_v<_Element_>) 
		|| std::is_pointer_v<_Element_> 
		|| std::is_same_v<_Element_, std::nullptr_t>) && sizeof(_Element_) == 8;

template <typename _Element_>
constexpr bool is_epi32_v = 
	((std::is_signed_v<_Element_> && !std::is_floating_point_v<_Element_>) 
		|| std::is_pointer_v<_Element_>
		|| std::is_same_v<_Element_, std::nullptr_t>) && sizeof(_Element_) == 4;

template <typename _Element_>
constexpr bool is_epu32_v = 
	((std::is_unsigned_v<_Element_> && !std::is_floating_point_v<_Element_>)
		|| std::is_pointer_v<_Element_>
		|| std::is_same_v<_Element_, std::nullptr_t>) && sizeof(_Element_) == 4;

template <typename _Element_>
constexpr bool is_epi16_v = sizeof(_Element_) == 2 && std::is_signed_v<_Element_> && !std::is_floating_point_v<_Element_>;

template <typename _Element_>
constexpr bool is_epu16_v = sizeof(_Element_) == 2 && std::is_unsigned_v<_Element_> && !std::is_floating_point_v<_Element_>;

template <typename _Element_>
constexpr bool is_epi8_v  = sizeof(_Element_) == 1 && std::is_signed_v<_Element_> && !std::is_floating_point_v<_Element_>;

template <typename _Element_>
constexpr bool is_epu8_v  = sizeof(_Element_) == 1 && std::is_unsigned_v<_Element_> && !std::is_floating_point_v<_Element_>;

template <typename _Element_>
constexpr bool is_pd_v    = sizeof(_Element_) == 8 && type_traits::is_any_of_v<_Element_, double, long double>;

template <typename _Element_>
constexpr bool is_ps_v    = sizeof(_Element_) == 4 && std::is_same_v<_Element_, float>;


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

__SIMD_STL_NUMERIC_NAMESPACE_END
