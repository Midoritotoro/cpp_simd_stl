#pragma once 

#include <src/simd_stl/type_traits/TypeTraits.h>
#include <simd_stl/compatibility/Inline.h>

#include <xstring>

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

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
public:
    using value_type    = _Element_;
    using vector_type   = __m128;

    using size_type = unsigned short;

    static simd_stl_constexpr_cxx20 simd_stl_always_inline value_type extract(
        const vector_type&  vector,
        const size_type     where) noexcept
    {
        return _mm_cvtss_f32(_mm_shuffle_ps(vector, vector, _MM_SHUFFLE(where, where, where, where)));
    }

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
public:
    using value_type = _Element_;
    using vector_type = type_traits::__deduce_simd_vector_type<arch::CpuFeature::SSE2, _Element_>;

    using size_type = unsigned short;

    static constexpr size_type vectorElementsCount = sizeof(vector_type) / sizeof(value_type);

    static simd_stl_constexpr_cxx20 simd_stl_always_inline value_type extract(
        const vector_type&  vector,
        const size_type     where) noexcept
    {
        if      constexpr (__is_pd_v<value_type>)
            return _mm_cvtsd_f64(_mm_shuffle_pd(vector, vector, _MM_SHUFFLE(where, where, where, where)));
        else if constexpr (__is_ps_v<value_type>)
            return _mm_cvtss_f32(_mm_shuffle_ps(vector, vector, _MM_SHUFFLE(where, where, where, where)));
        else if constexpr (__is_epi64_v<value_type> || __is_epu64_v<value_type>)
            return _mm_cvtsi128_si64(_mm_shuffle_epi32(vector, vector, _MM_SHUFFLE(where, where, where, where)));
        else if constexpr (__is_epi32_v<value_type> || __is_epu32_v<value_type>)
            return _mm_cvtsi128_si32(_mm_shuffle_epi32(vector, vector, _MM_SHUFFLE(where, where, where, where)));
        else if constexpr (__is_epi16_v<value_type> || __is_epu16_v<value_type>)
            return _mm_extract_epi16(vector, where);
        else if constexpr (__is_epi8_v<value_type> || __is_epu8_v<value_type>) {
            return ((where >> 1) < vectorElementsCount) 
                ? (_mm_cvtsi128_si64(vector) >> (where << 3)) & 0xff;
                : return (_mm_cvtsi128_si64(_mm_shuffle_epi32(vector, vector, 
                    _MM_SHUFFLE(where, where, where, where))) >> (where << 3)) & 0xff;
        }
    }

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
            return _mm_set1_epi32(value);
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
            return _mm_add_epi32(left, right);
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
            return _mm_sub_epi32(left, right);
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
            return _mm_mul_epi32(left, right);
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
            return _mm_div_epi32(left, right);
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
            return _mm_xor_pd(left, right);
        else if constexpr (std::is_same_v<vector_type, __m128i>)
            return _mm_xor_si128(left, right);
        else if constexpr (std::is_same_v<vector_type, __m128>)
            return _mm_xor_ps(left, right);
    }

    static simd_stl_constexpr_cxx20 simd_stl_always_inline vector_type bitwiseAnd(
        const vector_type& left,
        const vector_type& right) noexcept
    {
        if      constexpr (std::is_same_v<vector_type, __m128d>)
            return _mm_and_pd(left, right);
        else if constexpr (std::is_same_v<vector_type, __m128i>)
            return _mm_and_si128(left, right);
        else if constexpr (std::is_same_v<vector_type, __m128>)
            return _mm_and_ps(left, right);
    }

    static simd_stl_constexpr_cxx20 simd_stl_always_inline vector_type bitwiseOr(
        const vector_type& left,
        const vector_type& right) noexcept
    {
        if      constexpr (std::is_same_v<vector_type, __m128d>)
            return _mm_or_pd(left, right);
        else if constexpr (std::is_same_v<vector_type, __m128i>)
            return _mm_or_si128(left, right);
        else if constexpr (std::is_same_v<vector_type, __m128>)
            return _mm_or_ps(left, right);
    }
};

template <typename _Element_>
class BasicSimdImplementation<arch::CpuFeature::SSE3, _Element_>: 
    public BasicSimdImplementation<arch::CpuFeature::SSE2, _Element_> 
{

};

template <typename _Element_>
class BasicSimdImplementation<arch::CpuFeature::SSSE3, _Element_>:
    public BasicSimdImplementation<arch::CpuFeature::SSE3, _Element_> 
{};

template <typename _Element_>
class BasicSimdImplementation<arch::CpuFeature::SSE41, _Element_>: 
    public BasicSimdImplementation<arch::CpuFeature::SSSE3, _Element_> 
{};

template <typename _Element_>
class BasicSimdImplementation<arch::CpuFeature::SSE42, _Element_>: 
    public BasicSimdImplementation<arch::CpuFeature::SSE41, _Element_>
{};

template <typename _Element_>
class BasicSimdImplementation<arch::CpuFeature::AVX, _Element_> {
public:
    using value_type = _Element_;
    using vector_type = type_traits::__deduce_simd_vector_type<arch::CpuFeature::AVX, _Element_>;

    using size_type = unsigned short;

    static constexpr size_type vectorElementsCount = sizeof(vector_type) / sizeof(value_type);


    static simd_stl_constexpr_cxx20 simd_stl_always_inline value_type extract(
        const vector_type&  vector,
        const size_type     where) noexcept
    {
        if      constexpr (__is_pd_v<value_type>)
            return _mm256_cvtsd_f64(_mm_shuffle_pd(vector, vector, _MM_SHUFFLE(where, where, where, where)));
        else if constexpr (__is_ps_v<value_type>)
            return _mm_cvtss_f32(_mm_shuffle_ps(vector, vector, _MM_SHUFFLE(where, where, where, where)));
        else if constexpr (__is_epi64_v<value_type> || __is_epu64_v<value_type>)
            return _mm_cvtsi128_si64(_mm_shuffle_epi32(vector, vector, _MM_SHUFFLE(where, where, where, where)));
        else if constexpr (__is_epi32_v<value_type> || __is_epu32_v<value_type>)
            return _mm_cvtsi128_si32(_mm_shuffle_epi32(vector, vector, _MM_SHUFFLE(where, where, where, where)));
        else if constexpr (__is_epi16_v<value_type> || __is_epu16_v<value_type>)
            return _mm_extract_epi16(vector, where);
        else if constexpr (__is_epi8_v<value_type> || __is_epu8_v<value_type>) {
            return ((where >> 1) < vectorElementsCount)
                ? (_mm_cvtsi128_si64(vector) >> (where << 3)) & 0xff;
            : return (_mm_cvtsi128_si64(_mm_shuffle_epi32(vector, vector,
                _MM_SHUFFLE(where, where, where, where))) >> (where << 3)) & 0xff;
        }
    }

    static simd_stl_constexpr_cxx20 simd_stl_always_inline vector_type constructZero() noexcept {
        if      constexpr (std::is_same_v<vector_type, __m256d>)
            return _mm256_setzero_pd();
        else if constexpr (std::is_same_v<vector_type, __m256i>)
            return _mm256_setzero_si256();
        else if constexpr (std::is_same_v<vector_type, __m256>)
            return _mm256_setzero_ps();
    }

    static simd_stl_constexpr_cxx20 simd_stl_always_inline vector_type broadcast(const value_type value) noexcept {
        if constexpr (__is_epi64_v<value_type> || __is_epu64_v<value_type>)
            return _mm256_set1_epi64x(value);
        else if constexpr (__is_epi32_v<value_type> || __is_epu32_v<value_type>)
            return _mm256_set1_epi32(value);
        else if constexpr (__is_epi16_v<value_type> || __is_epu16_v<value_type>)
            return _mm256_set1_epi16(value);
        else if constexpr (__is_epi8_v<value_type> || __is_epu8_v<value_type>)
            return _mm256_set1_epi8(value);
        else if constexpr (__is_ps_v<value_type>)
            return _mm256_set1_ps(value);
        else if constexpr (__is_pd_v<value_type>)
            return _mm256_set1_pd(value);
    }

    static simd_stl_constexpr_cxx20 simd_stl_always_inline vector_type add(
        const vector_type& left,
        const vector_type& right) noexcept
    {
        if constexpr (__is_epi64_v<value_type> || __is_epu64_v<value_type>)
            return _mm256_add_epi64(left, right);
        else if constexpr (__is_epi32_v<value_type> || __is_epu32_v<value_type>)
            return _mm256_add_epi32(left, right);
        else if constexpr (__is_epi16_v<value_type> || __is_epu16_v<value_type>)
            return _mm256_add_epi16(left, right);
        else if constexpr (__is_epi8_v<value_type> || __is_epu8_v<value_type>)
            return _mm256_add_epi8(left, right);
        else if constexpr (__is_ps_v<value_type>)
            return _mm256_add_ps(left, right);
        else if constexpr (__is_pd_v<value_type>)
            return _mm256_add_pd(left, right);
    }

    static simd_stl_constexpr_cxx20 simd_stl_always_inline vector_type sub(
        const vector_type& left,
        const vector_type& right) noexcept
    {
        if constexpr (__is_epi64_v<value_type> || __is_epu64_v<value_type>)
            return _mm256_sub_epi64(left, right);
        else if constexpr (__is_epi32_v<value_type> || __is_epu32_v<value_type>)
            return _mm256_sub_epi32(left, right);
        else if constexpr (__is_epi16_v<value_type> || __is_epu16_v<value_type>)
            return _mm256_sub_epi16(left, right);
        else if constexpr (__is_epi8_v<value_type> || __is_epu8_v<value_type>)
            return _mm256_sub_epi8(left, right);
        else if constexpr (__is_ps_v<value_type>)
            return _mm256_sub_ps(left, right);
        else if constexpr (__is_pd_v<value_type>)
            return _mm256_sub_pd(left, right);
    }

    static simd_stl_constexpr_cxx20 simd_stl_always_inline vector_type mul(
        const vector_type& left,
        const vector_type& right) noexcept
    {
        if constexpr (__is_epi64_v<value_type> || __is_epu64_v<value_type>) {
            auto ymm2 = _mm256_mul_epu32(_mm256_srli_epi64(right, 32), left);
            auto ymm3 = _mm256_mul_epu32(_mm256_srli_epi64(left, 32), right);

            ymm2 = _mm256_slli_epi64(_mm256_add_epi64(ymm3, ymm2), 32);
            return _mm256_add_epi64(_mm256_mul_epu32(right, left), ymm2);
        }
        else if constexpr (__is_epi32_v<value_type>)
            return _mm256_mul_epi32(left, right);
        else if constexpr (__is_epu32_v<value_type>)
            return _mm256_mul_epu32(left, right);
        else if constexpr (__is_epi16_v<value_type> || __is_epu16_v<value_type>)
            return _mm256_mullo_epi16(left, right);
        else if constexpr (__is_epi8_v<value_type> || __is_epu8_v<value_type>) {
            auto ymm3 = _mm256_unpacklo_epi8(right, right);
            auto ymm2 = _mm256_unpacklo_epi8(left, left);

            left = _mm256_unpackhi_epi8(left, left);
            right = _mm256_unpackhi_epi8(right, right);

            ymm2 = _mm256_mullo_epi16(ymm2, ymm3);
            left = _mm256_mullo_epi16(left, right);

            ymm2 = _mm256_shuffle_epi8(ymm2, ymm2);
            left = _mm256_shuffle_epi8(left, left);

            return _mm256_blend_epi32(left, ymm2, 0x33);
        }
        else if constexpr (__is_ps_v<value_type>)
            return _mm256_mul_ps(left, right);
        else if constexpr (__is_pd_v<value_type>)
            return _mm256_mul_pd(left, right);
    }

    static simd_stl_constexpr_cxx20 simd_stl_always_inline vector_type div(
        const vector_type& left,
        const vector_type& right) noexcept
    {
        if      constexpr (__is_epi64_v<value_type>)
            return _mm256_castpd_si256(_mm256_div_pd(left, right));
        else if constexpr (__is_epu64_v<value_type>)
            return _mm256_castpd_si256(_mm256_div_pd(left, right));
        else if constexpr (__is_epi32_v<value_type> || __is_epu32_v<value_type>)
            return _mm256_castps_si256(_mm256_div_ps(left, right));
        else if constexpr (__is_epi16_v<value_type> || __is_epu16_v<value_type>)
            return div_u16(left, right);
        else if constexpr (__is_epi8_v<value_type> || __is_epu8_v<value_type>)
            return div_u8(left, right);
        else if constexpr (__is_ps_v<value_type>)
            return _mm256_div_ps(left, right);
        else if constexpr (__is_pd_v<value_type>)
            return _mm256_div_pd(left, right);
    }

    static simd_stl_constexpr_cxx20 simd_stl_always_inline vector_type bitwiseNot(const vector_type& vector) noexcept {
        if      constexpr (std::is_same_v<vector_type, __m256d>)
            return _mm256_xor_pd(vector, _mm256_cmp_pd(vector, vector, _CMP_EQ_OQ));
        else if constexpr (std::is_same_v<vector_type, __m256>)
            return _mm256_xor_ps(vector, _mm256_cmp_ps(vector, vector, _CMP_EQ_OQ));
    }

    static simd_stl_constexpr_cxx20 simd_stl_always_inline vector_type bitwiseXor(
        const vector_type& left,
        const vector_type& right) noexcept
    {
        if      constexpr (std::is_same_v<vector_type, __m256d>)
            return _mm256_xor_pd(left, right);
        else if constexpr (std::is_same_v<vector_type, __m256>)
            return _mm256_xor_ps(left, right);
    }

    static simd_stl_constexpr_cxx20 simd_stl_always_inline vector_type bitwiseAnd(
        const vector_type& left,
        const vector_type& right) noexcept
    {
        if      constexpr (std::is_same_v<vector_type, __m256d>)
            return _mm256_and_pd(left, right);
        else if constexpr (std::is_same_v<vector_type, __m256>)
            return _mm256_and_ps(left, right);
    }

    static simd_stl_constexpr_cxx20 simd_stl_always_inline vector_type bitwiseOr(
        const vector_type& left,
        const vector_type& right) noexcept
    {
        if      constexpr (std::is_same_v<vector_type, __m256d>)
            return _mm256_or_pd(left, right);
        else if constexpr (std::is_same_v<vector_type, __m256>)
            return _mm256_or_ps(left, right);
    }

private:
    static simd_stl_constexpr_cxx20 simd_stl_always_inline __m256i divLow_u8_i32x8(
        __m256i left,
        __m256i right, 
        float mul) noexcept 
    {
        const auto af = _mm256_cvtepi32_ps(left);
        const auto bf = _mm256_cvtepi32_ps(right);

        const auto m1 = _mm256_mul_ps(af, _mm256_set1_ps(1.001f * mul));
        const auto m2 = _mm256_rcp_ps(bf);

        return _mm256_cvttps_epi32(_mm256_mul_ps(m1, m2));
    }

    static simd_stl_constexpr_cxx20 simd_stl_always_inline __m256i div_u8(
        __m256i left,
        __m256i right) noexcept 
    {
        const auto m0 = _mm256_set1_epi32(0x000000ff);
        const auto m1 = _mm256_set1_epi32(0x0000ff00);
        const auto m2 = _mm256_set1_epi32(0x00ff0000);

        const auto r0 = divLow_u8_i32x8(_mm256_and_si256(left, m0), _mm256_and_si256(right, m0), 1);
        auto r1 = divLow_u8_i32x8(_mm256_and_si256(left, m1), _mm256_and_si256(right, m1), 1);

        r1 = _mm256_slli_epi32(r1, 8);

        const auto r2 = divLow_u8_i32x8(_mm256_and_si256(left, m2), _mm256_and_si256(right, m2), 1 << 16);
        auto r3 = divLow_u8_i32x8(_mm256_srli_epi32(left, 24), _mm256_srli_epi32(right, 24), 1);

        r3 = _mm256_slli_epi32(r3, 24);

        const auto r01 = _mm256_or_si256(r0, r1);
        const auto r23 = _mm256_or_si256(r2, r3);

        return _mm256_blend_epi16(r01, r23, 0xAA);
    }

    static simd_stl_constexpr_cxx20 simd_stl_always_inline __m256i div_u16(
        const __m256i left, 
        const __m256i right) noexcept
    {
        const auto mask_lo = _mm256_set1_epi32(0x0000ffff);

        const auto a_lo_u32 = _mm256_and_si256(left, mask_lo);
        const auto b_lo_u32 = _mm256_and_si256(right, mask_lo);

        const auto a_hi_u32 = _mm256_srli_epi32(left, 16);
        const auto b_hi_u32 = _mm256_srli_epi32(right, 16);

        const auto  a_lo_f32 = _mm256_cvtepi32_ps(a_lo_u32);
        const auto  a_hi_f32 = _mm256_cvtepi32_ps(a_hi_u32);
        const auto  b_lo_f32 = _mm256_cvtepi32_ps(b_lo_u32);
        const auto  b_hi_f32 = _mm256_cvtepi32_ps(b_hi_u32);

        const auto  c_lo_f32 = _mm256_div_ps(a_lo_f32, b_lo_f32);
        const auto  c_hi_f32 = _mm256_div_ps(a_hi_f32, b_hi_f32);

        const auto c_lo_i32 = _mm256_cvttps_epi32(c_lo_f32); // values in the u16 range
        const auto c_hi_i32_0 = _mm256_cvttps_epi32(c_hi_f32); // values in the u16 range
        const auto c_hi_i32 = _mm256_slli_epi32(c_hi_i32_0, 16);

        return _mm256_or_si256(c_lo_i32, c_hi_i32);
    }

    static simd_stl_constexpr_cxx20 simd_stl_always_inline __m256i div_u64(
        const __m256i ymm0,
        const __m256i ymm1) noexcept
    {
        /*auto ymm3 = _mm256_set1_epi64x(1);
        const auto ymm2 = bitwiseXor(ymm0, ymm1);

        const auto ymm4 = _mm256_abs_epi32(ymm0);
        const auto ymm5 = _mm256_abs_epi32(ymm1);

        auto ymm6 = constructZero();
        auto ymm7 = _mm256_cmpeq_epi32(ymm6, ymm6);

        auto k2 = _mm256_movemask_pd(vectorsXor);

        ymm0 = _mm256_castsi256_pd(ymm4);
        ymm1 = _mm256_castsi256_pd(ymm5);

        ymm1 = _mm256_div_pd(ymm3, ymm1);
        ymm0 = _mm256_mul_pd(ymm0, ymm1);

        ymm3 = _mm256_cvtepi32_pd(ymm0);
        ymm2 = _mm256_mul_pd(ymm3, ymm5);

        ymm4 = _mm256_sub_epi64(ymm4, ymm2);
        ymm0 = _mm256_castsi256_pd(ymm4);

        ymm0 = _mm256_mul_pd(ymm0, ymm1);
        ymm2 = _mm256_castsi256_pd(ymm0);

        ymm3 = _mm256_add_epi64(ymm3, ymm2);
        ymm2 = _mm256_mul_pd(ymm2, ymm5);

        ymm4 = _mm256_sub_epi64(ymm4, ymm2);
        auto k1 = _mm256_movemask_epi8(_mm256_cmpgt_epi64(ymm4, ymm5));

        ymm3 = _mm256_blend_epi8(_mm256_sub_epi64(ymm3, ymm7), k1);
        ymm3 = _mm256_blend_epi8(_mm256_sub_epi64(ymm6, ymm3), k2);

        return ymm3;*/
    }

};

template <typename _Element_>
class BasicSimdImplementation<arch::CpuFeature::AVX2, _Element_> {
public:
    using value_type = _Element_;
    using vector_type = type_traits::__deduce_simd_vector_type<arch::CpuFeature::AVX, _Element_>;

    using size_type = unsigned short;
};

template <typename _Element_>
class BasicSimdImplementation<arch::CpuFeature::AVX512F, _Element_> {

};

template <typename _Element_>
class BasicSimdImplementation<arch::CpuFeature::AVX512BW, _Element_> {

};

__SIMD_STL_NUMERIC_NAMESPACE_END