#pragma once 

#include <src/simd_stl/numeric/SimdCast.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_>
class _SimdConvert;

template <class _RegisterPolicy_>
class _SimdConvert<arch::CpuFeature::SSE2, _RegisterPolicy_> {
    using _Cast_ = _SimdCast<arch::CpuFeature::SSE2, _RegisterPolicy_>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline uint32 convertToMask(_VectorType_ vector) noexcept {
        if      constexpr (is_pd_v<_DesiredType_> || is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>)
            return _mm_movemask_pd(_Cast_::template cast<_VectorType_, __m128d>(vector));
        else if constexpr (is_ps_v<_DesiredType_> || is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>)
            return _mm_movemask_ps(_Cast_::template cast<_VectorType_, __m128>(vector));
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>)
            return _mm_movemask_epi8(_mm_packs_epi16(_Cast_::template cast<_VectorType_, __m128i>(vector), _mm_setzero_si128()));
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>)
            return _mm_movemask_epi8(_Cast_::template cast<_VectorType_, __m128i>(vector));
    }

    template <
        typename _VectorType_,
        typename _MaskType_>
    static simd_stl_always_inline _VectorType_ convertFromMask(_MaskType_ mask) noexcept {
        if (is_epi32_v<_MaskType_> || is_epu32_v<_MaskType_>) {
            const auto shuffle = _mm_setr_epi32(0, 0, 0x01010101, 0x01010101);
            _MaskType_ sourceVector[8], result[8];

            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(sourceVector),
                _Cast_::template cast<_VectorType_, __m128i>(shuffle));

            for (int j = 0; j < 8; ++j)
                result[j] = sourceVector[(mask >> (j * 4)) & 0x07];

            const auto bitSelect = _mm_setr_epi8(
                1, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6, 1U << 7,
                1, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6, 1U << 7);

            auto value = _mm_loadu_si128(static_cast<const __m128i*>(result));

            value = _mm_and_si128(value, bitSelect);
            value = _mm_min_epu8(value, _mm_set1_epi8(1));

            return _Cast_::template cast<__m128i, _VectorType_>(value);
        }
    }
};

template <class _RegisterPolicy_>
class _SimdConvert<arch::CpuFeature::SSE3, _RegisterPolicy_> :
    public _SimdConvert<arch::CpuFeature::SSE2, _RegisterPolicy_>
{};

template <class _RegisterPolicy_>
class _SimdConvert<arch::CpuFeature::SSSE3, _RegisterPolicy_> :
    public _SimdConvert<arch::CpuFeature::SSE3, _RegisterPolicy_>
{
    using _Cast_ = _SimdCast<arch::CpuFeature::SSSE3, _RegisterPolicy_>;
public:
    template <
        typename _VectorType_,
        typename _MaskType_>
    static simd_stl_always_inline _VectorType_ convertFromMask(_MaskType_ mask) noexcept {
        if (is_epi32_v<_MaskType_> || is_epu32_v<_MaskType_>) {
            const auto shuffle = _mm_setr_epi32(0, 0, 0x01010101, 0x01010101);
            auto value = _mm_shuffle_epi8(_mm_cvtsi32_si128(mask), shuffle);

            const auto bitSelect = _mm_setr_epi8(
                1, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6, 1U << 7,
                1, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6, 1U << 7);

            value = _mm_and_si128(value, bitSelect);
            value = _mm_min_epu8(value, _mm_set1_epi8(1));

            return _Cast_::template cast<__m128i, _VectorType_>(value);
        }
    }
};

template <class _RegisterPolicy_>
class _SimdConvert<arch::CpuFeature::SSE41, _RegisterPolicy_> :
    public _SimdConvert<arch::CpuFeature::SSSE3, _RegisterPolicy_>
{};

template <class _RegisterPolicy_>
class _SimdConvert<arch::CpuFeature::SSE42, _RegisterPolicy_> :
    public _SimdConvert<arch::CpuFeature::SSE41, _RegisterPolicy_>
{};

__SIMD_STL_NUMERIC_NAMESPACE_END
