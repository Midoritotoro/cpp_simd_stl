#pragma once 

#include <src/simd_stl/numeric/SimdIntegralTypesCheck.h>
#include <src/simd_stl/numeric/SimdConvert.h>

#include <simd_stl/numeric/BasicSimdShuffleMask.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <arch::CpuFeature _SimdGeneration_>
class SimdElementWise;

template <>
class SimdElementWise<arch::CpuFeature::SSE2> {
    static constexpr auto _Feature = arch::CpuFeature::SSE2;

    using _Cast_    = SimdCast<_Feature>;
    using _Convert_ = SimdConvert<_Feature>;
public:
    template <
        typename    _DesiredType_,
        uint8 ...   _Indices_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ permute(_VectorType_ vector) noexcept
    {
        if      constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_> || is_pd_v<_DesiredType_>)
            return _Cast_::template cast<__m128d, _VectorType_>(_mm_shuffle_pd(
                _Cast_::template cast<_VectorType_, __m128d>(vector),
                _Cast_::template cast<_VectorType_, __m128d>(vector),
                basic_simd_permute_mask<_Indices_...>::template unwrap<_Feature, _DesiredType_>())
            );
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_> || is_ps_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_shuffle_epi32(
                _Cast_::template cast<_VectorType_, __m128i>(vector),
                basic_simd_permute_mask<_Indices_...>::template unwrap<_Feature, _DesiredType_>())
            );
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>) {
            _DesiredType_ sourceVector[8], result[8];

            _mm_storeu_si128(
                reinterpret_cast<__m128*>(sourceVector),
                _Cast_::template cast<_VectorType_, __m128i>(vector));

            auto index = basic_simd_permute_mask<_Indices_...>::template unwrap<_Feature, _DesiredType_>();

            for (int j = 0; j < 8; j++) 
                result[j] = sourceVector[index[j] & 0x07];

            return _Cast_::template cast<__m128i, _VectorType_>(_mm_loadu_si128(reinterpret_cast<const __m128i*>(result)));
        }
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>) {
            _DesiredType_ sourceVector[16], result[16];

            _mm_storeu_si128(
                reinterpret_cast<__m128*>(sourceVector), 
                _Cast_::template cast<_VectorType_, __m128i>(vector));

            auto index = basic_simd_permute_mask<_Indices_...>::template unwrap<_Feature, _DesiredType_>();

            for (int32 j = 0; j < 16; j++) 
                result[j] = sourceVector[index[j] & 0x0F];

            return _Cast_::template cast<__m128i, _VectorType_>(_mm_loadu_si128(reinterpret_cast<const __m128i*>(result)));
        }
    }

    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ shuffle(
        _VectorType_                                                            vector,
        type_traits::__deduce_simd_shuffle_mask_type<_Feature, _DesiredType_>   mask) noexcept
    {
        if      constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_> || is_pd_v<_DesiredType_>)
            return _Cast_::template cast<__m128d, _VectorType_>(_mm_shuffle_pd(
                _Cast_::template cast<_VectorType_, __m128d>(vector),
                _Cast_::template cast<_VectorType_, __m128d>(vector), mask)
            );
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_> || is_ps_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_shuffle_epi32(
                _Cast_::template cast<_VectorType_, __m128i>(vector), mask)
            );
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>) {
            _DesiredType_ sourceVector[8], result[8];

            _mm_storeu_si128(
                reinterpret_cast<__m128*>(sourceVector),
                _Cast_::template cast<_VectorType_, __m128i>(vector));

            for (int j = 0; j < 8; j++)
                result[j] = sourceVector[(mask >> (j * 3)) & 0x07];

            return _Cast_::template cast<__m128i, _VectorType_>(_mm_loadu_si128(reinterpret_cast<const __m128i*>(result)));
        }
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>) {
            _DesiredType_ sourceVector[16], result[16];

            _mm_storeu_si128(
                reinterpret_cast<__m128*>(sourceVector), 
                _Cast_::template cast<_VectorType_, __m128i>(vector));

            for (int32 j = 0; j < 16; j++) 
                result[j] = sourceVector[(mask >> (j * 4)) & 0x0F];

            return _Cast_::template cast<__m128i, _VectorType_>(_mm_loadu_si128(reinterpret_cast<const __m128i*>(result)));
        }
    }

    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ blend(
        _VectorType_                                                    firstVector,
        _VectorType_                                                    secondVector,
        type_traits::__deduce_simd_mask_type<_Feature, _DesiredType_>   mask) noexcept
    {
        if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_> || is_pd_v<_DesiredType_>) {
            return _Cast_::cast<__m128i, _VectorType_>(_mm_shuffle_pd(
                _Cast_::cast<_VectorType_, __m128d>(secondVector),
                _Cast_::cast<_VectorType_, __m128d>(firstVector), ~mask)
            );
        }
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_> || is_ps_v<_DesiredType_>) {
            const __m128i shuffle = _mm_setr_epi32(0, 0, 0x01010101, 0x01010101);
            __m128i v = _mm_shuffle_epi8(_mm_cvtsi32_si128(mask), shuffle);  // SSSE3 pshufb

            const __m128i bitselect = _mm_setr_epi8(
                1, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6, 1U << 7,
                1, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6, 1U << 7);
            v = _mm_and_si128(v, bitselect);
            v = _mm_min_epu8(v, _mm_set1_epi8(1));
        }
    }
};

template <>
class SimdElementWise<arch::CpuFeature::SSE3> :
    public SimdElementWise<arch::CpuFeature::SSE2>
{};

template <>
class SimdElementWise<arch::CpuFeature::SSSE3> :
    public SimdElementWise<arch::CpuFeature::SSE3>
{};

template <>
class SimdElementWise<arch::CpuFeature::SSE41> :
    public SimdElementWise<arch::CpuFeature::SSSE3>
{};

template <>
class SimdElementWise<arch::CpuFeature::SSE42> :
    public SimdElementWise<arch::CpuFeature::SSE41>
{};

__SIMD_STL_NUMERIC_NAMESPACE_END
