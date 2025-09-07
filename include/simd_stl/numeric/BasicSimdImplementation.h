#pragma once 

#include <src/simd_stl/type_traits/TypeTraits.h>
#include <simd_stl/compatibility/Inline.h>

#include <xstring>

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

#if !defined(__simd_imlp)
#  define __simd_imlp template <arch::CpuFeature _SimdGeneration_, typename _Element_>
#endif // __simd_imlp

#if !defined(__simd_imlp_t)
#  define __simd_imlp_t BasicSimdImplementation<_SimdGeneration_, _Element_>
#endif // __simd_imlp_t


template <
    arch::CpuFeature	_SimdGeneration_,
    typename			_Element_>
class BasicSimdImplementation {};


template <typename _Element_>
constexpr bool __is_epi64_v = (sizeof(_Element_) == 8) && (std::is_integral_v<_Element_>);

template <typename _Element_>
constexpr bool __is_epu64_v = (sizeof(_Element_) == 8) && (std::is_integral_v<_Element_>) && (std::is_unsigned_v<_Element_>);

template <typename _Element_>
constexpr bool __is_epi32_v = (sizeof(_Element_) == 4) && (std::is_integral_v<_Element_>);

template <typename _Element_>
constexpr bool __is_epu32_v = (sizeof(_Element_) == 4) && (std::is_integral_v<_Element_>) && (std::is_unsigned_v<_Element_>);

template <typename _Element_>
constexpr bool __is_epi16_v = (sizeof(_Element_) == 2) && (std::is_integral_v<_Element_>);

template <typename _Element_>
constexpr bool __is_epu16_v = (sizeof(_Element_) == 2) && (std::is_integral_v<_Element_>) && (std::is_unsigned_v<_Element_>);

template <typename _Element_>
constexpr bool __is_epi8_v  = (sizeof(_Element_) == 1) && (std::is_integral_v<_Element_>);

template <typename _Element_>
constexpr bool __is_epu8_v  = (sizeof(_Element_) == 1) && (std::is_integral_v<_Element_>) && (std::is_unsigned_v<_Element_>);

template <typename _Element_>
constexpr bool __is_pd_v    = (sizeof(_Element_) == sizeof(double)) && (type_traits::is_any_of_v<_Element_, double, long double>);

template <typename _Element_>
constexpr bool __is_ps_v    = (sizeof(_Element_) == sizeof(float)) && (std::is_same_v<_Element_, float>);


template <typename _Element_>
class BasicSimdImplementation<arch::CpuFeature::SSE, _Element_> {
    using value_type    = _Element_;
    using vector_type   = __m128; // SSE работает только с float-векторами 

    using size_type = unsigned short;

    static simd_stl_constexpr_cxx20 simd_stl_always_inline vector_type constructZero() noexcept {
        return _mm_setzero_ps();
    }

    static simd_stl_constexpr_cxx20 simd_stl_always_inline vector_type broadcast(const value_type value) noexcept {
        return _mm_set1_ps(value);
    }

    static simd_stl_constexpr_cxx20 simd_stl_always_inline vector_type add(
        const vector_type& left, 
        const vector_type& right) noexcept
    {
        return _mm_add_ps(left, right);
    }

    static simd_stl_constexpr_cxx20 simd_stl_always_inline vector_type sub(
        const vector_type& left, 
        const vector_type& right) noexcept 
    {
        return _mm_sub_ps(left, right);
    }

    static simd_stl_constexpr_cxx20 simd_stl_always_inline vector_type mul(
        const vector_type& left, 
        const vector_type& right) noexcept
    {
        return _mm_mul_ps(left, right);
    }

    static simd_stl_constexpr_cxx20 simd_stl_always_inline vector_type div(
        const vector_type& left,
        const vector_type& right) noexcept
    {
        return _mm_div_ps(left, right);
    }

    static simd_stl_constexpr_cxx20 simd_stl_always_inline vector_type bitwiseNot(const vector_type& vector) noexcept {
        return _mm_xor_ps(vector, _mm_cmpeq_ps(vector, vector)); 
    }

    static simd_stl_constexpr_cxx20 simd_stl_always_inline vector_type bitwiseXor(
        const vector_type& left, 
        const vector_type& right) noexcept 
    {
        return _mm_xor_ps(left, right);
    }

    static simd_stl_constexpr_cxx20 simd_stl_always_inline vector_type bitwiseAnd(
        const vector_type& left, 
        const vector_type& right) noexcept 
    {
        return _mm_and_ps(left, right);
    }

    static simd_stl_constexpr_cxx20 simd_stl_always_inline vector_type bitwiseOr(
        const vector_type& left, 
        const vector_type& right) noexcept
    {
        return _mm_or_ps(left, right);
    }
};

template <typename _Element_>
class BasicSimdImplementation<arch::CpuFeature::SSE2, _Element_> {
    using value_type = _Element_;
    using vector_type = type_traits::__deduce_simd_vector_type<arch::CpuFeature::SSE2, _Element_>;

    using size_type = unsigned short;

    static simd_stl_constexpr_cxx20 simd_stl_always_inline vector_type constructZero() noexcept {
        if      constexpr (std::is_same_v<vector_type, __m128d>)
            return _mm_setzero_pd();
        else if constexpr (std::is_same_v<vector_type, __m128i>)
            return _mm_setzero_si128();
        else if constexpr (std::is_same_v<vector_type, __m128>)
            return _mm_setzero_ps();
    }

    static simd_stl_constexpr_cxx20 simd_stl_always_inline vector_type broadcast(const value_type value) noexcept {
        if constexpr (__is_epi64_v<value_type> || __is_epu64_v<value_type>)
            return _mm_set1_epi64x(value);
        else if constexpr (__is_epi32_v<value_type> || __is_epu32_v<value_type>)
            return _mm_set1_epi32(value)
        else if constexpr (__is_epi16_v<value_type> || __is_epu16_v<value_type>)
            return _mm_set1_epi16(value);
        else if constexpr (__is_epi8_v<value_type> || __is_epu8_v<value_type>)
            return _mm_set1_epi8(value);
        else if constexpr (__is_ps_v<value_type>)
            return _mm_set1_ps(value);
        else if constexpr (__is_pd_v<value_type>)
            return _mm_set1_pd(value);
    }

    static simd_stl_constexpr_cxx20 simd_stl_always_inline vector_type add(
        const vector_type& left,
        const vector_type& right) noexcept
    {
        if constexpr (__is_epi64_v<value_type> || __is_epu64_v<value_type>)
            return _mm_add_epi64(left, right);
        else if constexpr (__is_epi32_v<value_type> || __is_epu32_v<value_type>)
            return _mm_add_epi32(left, right)
        else if constexpr (__is_epi16_v<value_type> || __is_epu16_v<value_type>)
            return _mm_add_epi16(left, right);
        else if constexpr (__is_epi8_v<value_type> || __is_epu8_v<value_type>)
            return _mm_add_epi8(left, right);
        else if constexpr (__is_ps_v<value_type>)
            return _mm_add_ps(left, right);
        else if constexpr (__is_pd_v<value_type>)
            return _mm_add_pd(left, right);
    }

    static simd_stl_constexpr_cxx20 simd_stl_always_inline vector_type sub(
        const vector_type& left,
        const vector_type& right) noexcept
    {
        if constexpr (__is_epi64_v<value_type> || __is_epu64_v<value_type>)
            return _mm_sub_epi64(left, right);
        else if constexpr (__is_epi32_v<value_type> || __is_epu32_v<value_type>)
            return _mm_sub_epi32(left, right)
        else if constexpr (__is_epi16_v<value_type> || __is_epu16_v<value_type>)
            return _mm_sub_epi16(left, right);
        else if constexpr (__is_epi8_v<value_type> || __is_epu8_v<value_type>)
            return _mm_sub_epi8(left, right);
        else if constexpr (__is_ps_v<value_type>)
            return _mm_sub_ps(left, right);
        else if constexpr (__is_pd_v<value_type>)
            return _mm_sub_pd(left, right);
    }

    static simd_stl_constexpr_cxx20 simd_stl_always_inline vector_type mul(
        const vector_type& left,
        const vector_type& right) noexcept
    {
        if      constexpr (__is_epi64_v<value_type>)
            return _mm_mul_epi64(left, right);
        else if constexpr (__is_epu64_v<value_type>)
            return _mm_mul_epu64(left, right);
        else if constexpr (__is_epi32_v<value_type>)
            return _mm_mul_epi32(left, right)
        else if constexpr (__is_epu32_v<value_type>)
            return _mm_mul_epu32(left, right);
        else if constexpr (__is_epi16_v<value_type>)
            return _mm_mul_epi16(left, right);
        else if constexpr (__is_epu16_v<value_type>)
            return _mm_mul_epu16(left, right);
        else if constexpr (__is_epi8_v<value_type>)
            return _mm_mul_epi8(left, right);
        else if constexpr (__is_epu8_v<value_type>)
            return _mm_mul_epi8(left, right);
        else if constexpr (__is_ps_v<value_type>)
            return _mm_mul_ps(left, right);
        else if constexpr (__is_pd_v<value_type>)
            return _mm_mul_pd(left, right);
    }

    static simd_stl_constexpr_cxx20 simd_stl_always_inline vector_type div(
        const vector_type& left,
        const vector_type& right) noexcept
    {
        if      constexpr (__is_epi64_v<value_type>)
            return _mm_div_epi64(left, right);
        else if constexpr (__is_epu64_v<value_type>)
            return _mm_div_epu64(left, right);
        else if constexpr (__is_epi32_v<value_type>)
            return _mm_div_epi32(left, right)
        else if constexpr (__is_epu32_v<value_type>)
            return _mm_div_epu32(left, right);
        else if constexpr (__is_epi16_v<value_type>)
            return _mm_div_epi16(left, right);
        else if constexpr (__is_epu16_v<value_type>)
            return _mm_div_epu16(left, right);
        else if constexpr (__is_epi8_v<value_type>)
            return _mm_div_epi8(left, right);
        else if constexpr (__is_epu8_v<value_type>)
            return _mm_div_epi8(left, right);
        else if constexpr (__is_ps_v<value_type>)
            return _mm_div_ps(left, right);
        else if constexpr (__is_pd_v<value_type>)
            return _mm_div_pd(left, right);
    }

    static simd_stl_constexpr_cxx20 simd_stl_always_inline vector_type bitwiseNot(const vector_type& vector) noexcept {
        if      constexpr (std::is_same_v<vector_type, __m128d>)
            return _mm_xor_pd(vector, _mm_cmpeq_pd(vector, vector));
        else if constexpr (std::is_same_v<vector_type, __m128i>)
            return _mm_xor_si128(vector, _mm_cmpeq_epi32(vector, vector));
        else if constexpr (std::is_same_v<vector_type, __m128>)
            return _mm_xor_ps(vector, _mm_cmpeq_ps(vector, vector));
    }

    static simd_stl_constexpr_cxx20 simd_stl_always_inline vector_type bitwiseXor(
        const vector_type& left,
        const vector_type& right) noexcept
    {
        if      constexpr (std::is_same_v<vector_type, __m128d>)
            return _mm_xor_pd(vector);
        else if constexpr (std::is_same_v<vector_type, __m128i>)
            return _mm_xor_si128(vector);
        else if constexpr (std::is_same_v<vector_type, __m128>)
            return _mm_xor_ps(vector);
    }

    static simd_stl_constexpr_cxx20 simd_stl_always_inline vector_type bitwiseAnd(
        const vector_type& left,
        const vector_type& right) noexcept
    {
        if      constexpr (std::is_same_v<vector_type, __m128d>)
            return _mm_and_pd(vector);
        else if constexpr (std::is_same_v<vector_type, __m128i>)
            return _mm_and_si128(vector);
        else if constexpr (std::is_same_v<vector_type, __m128>)
            return _mm_and_ps(vector);
    }

    static simd_stl_constexpr_cxx20 simd_stl_always_inline vector_type bitwiseOr(
        const vector_type& left,
        const vector_type& right) noexcept
    {
        if      constexpr (std::is_same_v<vector_type, __m128d>)
            return _mm_or_pd(vector);
        else if constexpr (std::is_same_v<vector_type, __m128i>)
            return _mm_or_si128(vector);
        else if constexpr (std::is_same_v<vector_type, __m128>)
            return _mm_or_ps(vector);
    }
};

template <typename _Element_>
class BasicSimdImplementation<arch::CpuFeature::SSE3, _Element_> {

};

template <typename _Element_>
class BasicSimdImplementation<arch::CpuFeature::SSSE3, _Element_> {

};

template <typename _Element_>
class BasicSimdImplementation<arch::CpuFeature::SSE41, _Element_> {

};

template <typename _Element_>
class BasicSimdImplementation<arch::CpuFeature::SSE42, _Element_> {

};

template <typename _Element_>
class BasicSimdImplementation<arch::CpuFeature::AVX, _Element_> {

};

template <typename _Element_>
class BasicSimdImplementation<arch::CpuFeature::AVX2, _Element_> {

};

template <typename _Element_>
class BasicSimdImplementation<arch::CpuFeature::AVX512F, _Element_> {

};

template <typename _Element_>
class BasicSimdImplementation<arch::CpuFeature::AVX512BW, _Element_> {

};
