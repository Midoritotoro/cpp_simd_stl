#pragma once 

#include <simd_stl/arch/CpuFeature.h>
#include <src/simd_stl/type_traits/TypeTraits.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <arch::CpuFeature _SimdGeneration_>
class BasicSimdImplementation {};

template <typename _Element_>
constexpr bool is_epi64_v = sizeof(_Element_) == 8 && std::is_signed_v<_Element_>;

template <typename _Element_>
constexpr bool is_epu64_v = sizeof(_Element_) == 8 && std::is_unsigned_v<_Element_>;

template <typename _Element_>
constexpr bool is_epi32_v = sizeof(_Element_) == 4 && std::is_signed_v<_Element_>;

template <typename _Element_>
constexpr bool is_epu32_v = sizeof(_Element_) == 4 && std::is_unsigned_v<_Element_>;

template <typename _Element_>
constexpr bool is_epi16_v = sizeof(_Element_) == 2 && std::is_signed_v<_Element_>;

template <typename _Element_>
constexpr bool is_epu16_v = sizeof(_Element_) == 2 && std::is_unsigned_v<_Element_>;

template <typename _Element_>
constexpr bool is_epi8_v  = sizeof(_Element_) == 1 && std::is_signed_v<_Element_>;

template <typename _Element_>
constexpr bool is_epu8_v  = sizeof(_Element_) == 1 && std::is_unsigned_v<_Element_>;

template <typename _Element_>
constexpr bool is_pd_v    = sizeof(_Element_) == 8 && type_traits::is_any_of_v<_Element_, double, long double>;

template <typename _Element_>
constexpr bool is_ps_v    = sizeof(_Element_) == 4 && std::is_same_v<_Element_, float>;

__SIMD_STL_NUMERIC_NAMESPACE_END
