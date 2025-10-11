#pragma once 

#include <src/simd_stl/numeric/SimdIntegralTypesCheck.h>
#include <src/simd_stl/numeric/SimdCast.h>

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <arch::CpuFeature _SimdGeneration_>
class SimdElementWise;

template <>
class SimdElementWise<arch::CpuFeature::SSE2> {
    using _Cast_            = SimdCast<arch::CpuFeature::SSE2>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ shuffle(
        _VectorType_                                        vector,
        uint8 shuffleMask) noexcept
    {
        return shuffle<_DesiredType_>(vector, vector, shuffleMask);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ shuffle(
        _VectorType_                                            vector,
        _VectorType_                                            secondVector,
        uint8       shuffleMask) noexcept
    {
        //if      constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_> || is_pd_v<_DesiredType_>)
        //    return _Cast_::template cast<__m128d, _VectorType_>(_mm_shuffle_pd(
        //        _Cast_::template cast<_VectorType_, __m128d>(vector),
        //        _Cast_::template cast<_VectorType_, __m128d>(secondVector),
        //        shuffleMask)
        //    );
        //else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_> || is_ps_v<_DesiredType_>)
        //    return _Cast_::template cast<__m128i, _VectorType_>(_mm_shuffle_epi32(
        //        _Cast_::template cast<_VectorType_, __m128i>(vector),
        //        shuffleMask)
        //    );
        //else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>) {

        //}
        //else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>) {
        //    uint8 ii[16];
        //    int8  sourceVector[16], rr[16];

        //    _mm_storeu_si128(reinterpret_cast<__m128* vector);
        //    index.store(ii);

        //    for (int32 j = 0; j < 16; j++) 
        //        rr[j] = tt[ii[j] & 0x0F];

        //    return Vec16c().load(rr);
        //}
        return vector;
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
