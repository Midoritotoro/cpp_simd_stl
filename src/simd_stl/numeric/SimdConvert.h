#pragma once 

#include <src/simd_stl/numeric/SimdElementWise.h>
#include <simd_stl/numeric/BasicSimdShuffleMask.h>

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <arch::CpuFeature _SimdGeneration_>
class SimdConvert;

template <>
class SimdConvert<arch::CpuFeature::SSE2> {
    using _Cast_        = SimdCast<arch::CpuFeature::SSE2>;
    using _ElementWise_ = SimdElementWise<arch::CpuFeature::SSE2>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline uint32 convertToMask(_VectorType_ vector) noexcept {
        if      constexpr (is_pd_v<_DesiredType_> || is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>)
            return _mm_movemask_pd(_Cast_::template cast<_VectorType_, __m128d>(vector));
        else if constexpr (is_ps_v<_DesiredType_> || is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>)
            return _mm_movemask_ps(_Cast_::template cast<_VectorType_, __m128>(vector));
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>) {
            const auto mask     = _mm_set1_epi32(0b00000000000000010000000000000001);

            const auto bitAnd   = _mm_and_si128(mask, _Cast_::template cast<_VectorType_, __m128i>(vector));

            // { 1, 0, 1, 0, 1, 0, 1, 0 }
            const auto toMask   = _mm_shufflelo_epi16(bitAnd, basic_simd_shuffle_mask<0, >);

        }
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>) {
            return _mm_movemask_epi8(_Cast_::template cast<_VectorType_, __m128i>(vector));
        }
    }
};

template <>
class SimdConvert<arch::CpuFeature::SSE3> :
    public SimdConvert<arch::CpuFeature::SSE2>
{};

template <>
class SimdConvert<arch::CpuFeature::SSSE3> :
    public SimdConvert<arch::CpuFeature::SSE3>
{};

template <>
class SimdConvert<arch::CpuFeature::SSE41> :
    public SimdConvert<arch::CpuFeature::SSSE3>
{};

template <>
class SimdConvert<arch::CpuFeature::SSE42> :
    public SimdConvert<arch::CpuFeature::SSE41>
{};

__SIMD_STL_NUMERIC_NAMESPACE_END
