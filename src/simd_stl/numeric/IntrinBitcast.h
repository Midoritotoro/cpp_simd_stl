#pragma once 

#include <src/simd_stl/numeric/SimdIntegralTypesCheck.h>
#include <src/simd_stl/type_traits/SimdTypeCheck.h>

#include <simd_stl/compatibility/Inline.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    class _ToVector_,
    class _FromVector_>
static simd_stl_always_inline _ToVector_ _IntrinBitcast(_FromVector_ from) noexcept {
    static_assert(_Is_intrin_type_v<_ToVector_> && _Is_intrin_type_v<_FromVector_>, "SimdCast: unsupported non-SIMD type. ");

    if constexpr (std::is_same_v<_FromVector_, _ToVector_>)
        return from;

    // Xmm casts 

    else if constexpr (std::is_same_v<_FromVector_, __m128> && std::is_same_v<_ToVector_, __m128i>)
        return _mm_castps_si128(from);

    else if constexpr (std::is_same_v<_FromVector_, __m128> && std::is_same_v<_ToVector_, __m128d>)
        return _mm_castps_pd(from);

    else if constexpr (std::is_same_v<_FromVector_, __m128d> && std::is_same_v<_ToVector_, __m128>)
        return _mm_castpd_ps(from);

    else if constexpr (std::is_same_v<_FromVector_, __m128d> && std::is_same_v<_ToVector_, __m128i>)
        return _mm_castpd_si128(from);

    else if constexpr (std::is_same_v<_FromVector_, __m128i> && std::is_same_v<_ToVector_, __m128>)
        return _mm_castsi128_ps(from);

    else if constexpr (std::is_same_v<_FromVector_, __m128i> && std::is_same_v<_ToVector_, __m128d>)
        return _mm_castsi128_pd(from);

    // Ymm casts

    else if constexpr (std::is_same_v<_FromVector_, __m256> && std::is_same_v<_ToVector_, __m256i>)
        return _mm256_castps_si256(from);

    else if constexpr (std::is_same_v<_FromVector_, __m256> && std::is_same_v<_ToVector_, __m256d>)
        return _mm256_castps_pd(from);

    else if constexpr (std::is_same_v<_FromVector_, __m256d> && std::is_same_v<_ToVector_, __m256>)
        return _mm256_castpd_ps(from);

    else if constexpr (std::is_same_v<_FromVector_, __m256d> && std::is_same_v<_ToVector_, __m256i>)
        return _mm256_castpd_si256(from);

    else if constexpr (std::is_same_v<_FromVector_, __m256i> && std::is_same_v<_ToVector_, __m256>)
        return _mm256_castsi256_ps(from);

    else if constexpr (std::is_same_v<_FromVector_, __m256i> && std::is_same_v<_ToVector_, __m256d>)
        return _mm256_castsi256_pd(from);

    // Xmm to ymm casts

    else if constexpr (std::is_same_v<_FromVector_, __m128> && std::is_same_v<_ToVector_, __m256>)
        return _mm256_zextps128_ps256(from);

    else if constexpr (std::is_same_v<_FromVector_, __m128i> && std::is_same_v<_ToVector_, __m256>)
        return _mm256_castsi256_ps(_mm256_zextsi128_si256(from));

    else if constexpr (std::is_same_v<_FromVector_, __m128d> && std::is_same_v<_ToVector_, __m256>)
        return _mm256_castpd_ps(_mm256_zextpd128_pd256(from));

    else if constexpr (std::is_same_v<_FromVector_, __m128> && std::is_same_v<_ToVector_, __m256d>)
        return _mm256_castps_pd(_mm256_zextps128_ps256(from));

    else if constexpr (std::is_same_v<_FromVector_, __m128i> && std::is_same_v<_ToVector_, __m256d>)
        return _mm256_castsi256_pd(_mm256_zextsi128_si256(from));

    else if constexpr (std::is_same_v<_FromVector_, __m128d> && std::is_same_v<_ToVector_, __m256d>)
        return _mm256_zextpd128_pd256(from);

    else if constexpr (std::is_same_v<_FromVector_, __m128> && std::is_same_v<_ToVector_, __m256i>)
        return _mm256_castps_si256(_mm256_zextps128_ps256(from));

    else if constexpr (std::is_same_v<_FromVector_, __m128i> && std::is_same_v<_ToVector_, __m256i>)
        return _mm256_zextsi128_si256(from);

    else if constexpr (std::is_same_v<_FromVector_, __m128d> && std::is_same_v<_ToVector_, __m256i>)
        return _mm256_castpd_si256(_mm256_zextpd128_pd256(from));

    // Ymm to xmm casts

    else if constexpr (std::is_same_v<_FromVector_, __m256> && std::is_same_v<_ToVector_, __m128>)
        return _mm256_zextps256_ps128(from);

    else if constexpr (std::is_same_v<_FromVector_, __m256i> && std::is_same_v<_ToVector_, __m128>)
        return _mm_castsi128_ps(_mm256_castsi256_si128(from));

    else if constexpr (std::is_same_v<_FromVector_, __m256d> && std::is_same_v<_ToVector_, __m128>)
        return _mm_castpd_ps(_mm256_castpd256_pd128(from));

    else if constexpr (std::is_same_v<_FromVector_, __m256> && std::is_same_v<_ToVector_, __m128d>)
        return _mm_castps_pd(_mm256_castps256_ps128(from));

    else if constexpr (std::is_same_v<_FromVector_, __m256i> && std::is_same_v<_ToVector_, __m128d>)
        return _mm_castsi128_pd(_mm256_castsi256_si128(from));

    else if constexpr (std::is_same_v<_FromVector_, __m256d> && std::is_same_v<_ToVector_, __m128d>)
        return _mm256_castpd256_pd128(from);

    else if constexpr (std::is_same_v<_FromVector_, __m256> && std::is_same_v<_ToVector_, __m128i>)
        return _mm_castps_si128(_mm256_castps256_ps128(from));

    else if constexpr (std::is_same_v<_FromVector_, __m256i> && std::is_same_v<_ToVector_, __m128i>)
        return _mm256_castsi256_si128(from);

    else if constexpr (std::is_same_v<_FromVector_, __m256d> && std::is_same_v<_ToVector_, __m128i>)
        return _mm_castpd_si128(_mm256_castpd256_pd128(from));

    // Zmm casts 

    else if constexpr (std::is_same_v<_FromVector_, __m512> && std::is_same_v<_ToVector_, __m512i>)
        return _mm512_castps_si512(from);

    else if constexpr (std::is_same_v<_FromVector_, __m512> && std::is_same_v<_ToVector_, __m512d>)
        return _mm512_castps_pd(from);

    else if constexpr (std::is_same_v<_FromVector_, __m512d> && std::is_same_v<_ToVector_, __m512>)
        return _mm512_castpd_ps(from);

    else if constexpr (std::is_same_v<_FromVector_, __m512d> && std::is_same_v<_ToVector_, __m512i>)
        return _mm512_castpd_si512(from);

    else if constexpr (std::is_same_v<_FromVector_, __m512i> && std::is_same_v<_ToVector_, __m512>)
        return _mm512_castsi512_ps(from);

    else if constexpr (std::is_same_v<_FromVector_, __m512i> && std::is_same_v<_ToVector_, __m512d>)
        return _mm512_castsi512_pd(from);

    // Zmm to ymm casts

    else if constexpr (std::is_same_v<_FromVector_, __m512> && std::is_same_v<_ToVector_, __m256i>)
        return _mm256_castps_si256(_mm512_castps512_ps256(from));

    else if constexpr (std::is_same_v<_FromVector_, __m512> && std::is_same_v<_ToVector_, __m256>)
        return _mm512_castps512_ps256(from);
    
    else if constexpr (std::is_same_v<_FromVector_, __m512> && std::is_same_v<_ToVector_, __m256d>)
        return _mm256_castps_pd(_mm512_castps512_ps256(from));

    else if constexpr (std::is_same_v<_FromVector_, __m512i> && std::is_same_v<_ToVector_, __m256i>)
        return _mm512_castsi512_si256(from);

    else if constexpr (std::is_same_v<_FromVector_, __m512i> && std::is_same_v<_ToVector_, __m256>)
        return _mm256_castsi256_ps(_mm512_castsi512_si256(from));

    else if constexpr (std::is_same_v<_FromVector_, __m512i> && std::is_same_v<_ToVector_, __m256d>)
        return _mm256_castsi256_pd(_mm512_castsi512_si256(from));

    else if constexpr (std::is_same_v<_FromVector_, __m512d> && std::is_same_v<_ToVector_, __m256i>)
        return _mm256_castpd_si256(_mm512_castpd512_pd256(from));

    else if constexpr (std::is_same_v<_FromVector_, __m512d> && std::is_same_v<_ToVector_, __m256>)
        return _mm256_castpd_ps(_mm512_castpd512_pd256(from));

    else if constexpr (std::is_same_v<_FromVector_, __m512d> && std::is_same_v<_ToVector_, __m256d>)
        return _mm512_castpd512_pd256(from);

    // Zmm to xmm casts 

    else if constexpr (std::is_same_v<_FromVector_, __m512> && std::is_same_v<_ToVector_, __m128i>)
        return _mm_castps_si128(_mm512_castps512_ps128(from));

    else if constexpr (std::is_same_v<_FromVector_, __m512> && std::is_same_v<_ToVector_, __m128>)
        return _mm512_castps512_ps128(from);

    else if constexpr (std::is_same_v<_FromVector_, __m512> && std::is_same_v<_ToVector_, __m128d>)
        return _mm_castps_pd(_mm512_castps512_ps128(from));

    else if constexpr (std::is_same_v<_FromVector_, __m512i> && std::is_same_v<_ToVector_, __m128i>)
        return _mm512_castsi512_si128(from);

    else if constexpr (std::is_same_v<_FromVector_, __m512i> && std::is_same_v<_ToVector_, __m128>)
        return _mm_castsi128_ps(_mm512_castsi512_si128(from));

    else if constexpr (std::is_same_v<_FromVector_, __m512i> && std::is_same_v<_ToVector_, __m128d>)
        return _mm_castsi128_pd(_mm512_castsi512_si128(from));

    else if constexpr (std::is_same_v<_FromVector_, __m512d> && std::is_same_v<_ToVector_, __m128i>)
        return _mm_castpd_si128(_mm512_castpd512_pd128(from));

    else if constexpr (std::is_same_v<_FromVector_, __m512d> && std::is_same_v<_ToVector_, __m128>)
        return _mm_castpd_ps(_mm512_castpd512_pd128(from));

    else if constexpr (std::is_same_v<_FromVector_, __m512d> && std::is_same_v<_ToVector_, __m128d>)
        return _mm512_castpd512_pd128(from);

    // Ymm to zmm casts

    else if constexpr (std::is_same_v<_FromVector_, __m256> && std::is_same_v<_ToVector_, __m512i>)
        return _mm512_castps_si512(_mm512_zextps256_ps512(from));

    else if constexpr (std::is_same_v<_FromVector_, __m256d> && std::is_same_v<_ToVector_, __m512i>)
        return _mm512_castpd_si512(_mm512_zextpd256_pd512(from));

    else if constexpr (std::is_same_v<_FromVector_, __m256i> && std::is_same_v<_ToVector_, __m512i>)
        return _mm512_zextsi256_si512(from);

    else if constexpr (std::is_same_v<_FromVector_, __m256> && std::is_same_v<_ToVector_, __m512d>)
        return _mm512_castps_pd(_mm512_zextps256_ps512(from));

    else if constexpr (std::is_same_v<_FromVector_, __m256d> && std::is_same_v<_ToVector_, __m512d>)
        return _mm512_zextpd256_pd512(from);

    else if constexpr (std::is_same_v<_FromVector_, __m256i> && std::is_same_v<_ToVector_, __m512d>)
        return _mm512_castsi512_pd(_mm512_zextsi256_si512(from));

    else if constexpr (std::is_same_v<_FromVector_, __m256> && std::is_same_v<_ToVector_, __m512>)
        return _mm512_zextps256_ps512(from);

    else if constexpr (std::is_same_v<_FromVector_, __m256d> && std::is_same_v<_ToVector_, __m512>)
        return _mm512_castpd_ps(_mm512_zextpd256_pd512(from));

    else if constexpr (std::is_same_v<_FromVector_, __m256i> && std::is_same_v<_ToVector_, __m512>)
        return _mm512_castsi512_ps(_mm512_zextsi256_si512(from));

    // Xmm to zmm casts

    else if constexpr (std::is_same_v<_FromVector_, __m128> && std::is_same_v<_ToVector_, __m512i>)
        return _mm512_castps_si512(_mm512_zextps128_ps512(from));

    else if constexpr (std::is_same_v<_FromVector_, __m128d> && std::is_same_v<_ToVector_, __m512i>)
        return _mm512_castpd_si512(_mm512_zextpd128_pd512(from));

    else if constexpr (std::is_same_v<_FromVector_, __m128i> && std::is_same_v<_ToVector_, __m512i>)
        return _mm512_zextsi128_si512(from);

    else if constexpr (std::is_same_v<_FromVector_, __m128> && std::is_same_v<_ToVector_, __m512d>)
        return _mm512_castps_pd(_mm512_zextps128_ps512(from));

    else if constexpr (std::is_same_v<_FromVector_, __m128d> && std::is_same_v<_ToVector_, __m512d>)
        return _mm512_zextpd128_pd512(from);

    else if constexpr (std::is_same_v<_FromVector_, __m128i> && std::is_same_v<_ToVector_, __m512d>)
        return _mm512_castsi512_pd(_mm512_zextsi128_si512(from));

    else if constexpr (std::is_same_v<_FromVector_, __m128> && std::is_same_v<_ToVector_, __m512>)
        return _mm512_zextps128_ps512(from);

    else if constexpr (std::is_same_v<_FromVector_, __m128d> && std::is_same_v<_ToVector_, __m512>)
        return _mm512_castpd_ps(_mm512_zextpd128_pd512(from));

    else if constexpr (std::is_same_v<_FromVector_, __m128i> && std::is_same_v<_ToVector_, __m512>)
        return _mm512_castsi512_ps(_mm512_zextsi128_si512(from));
}


__SIMD_STL_NUMERIC_NAMESPACE_END
