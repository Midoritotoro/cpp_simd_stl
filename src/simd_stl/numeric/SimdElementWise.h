#pragma once 

#include <src/simd_stl/numeric/BasicSimdImplementationUnspecialized.h>
#include <src/simd_stl/numeric/xmm/SimdDivisors.h>

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <arch::CpuFeature _SimdGeneration_>
class SimdElementWise;

template <>
class SimdElementWise<arch::CpuFeature::SSE2> {
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ shuffle(
        _VectorType_                                        vector,
        type_traits::__deduce_simd_shuffle_mask_type<
        sizeof(_VectorType_) / sizeof(_DesiredType_)>   shuffleMask) noexcept
    {
        return shuffle<_DesiredType_>(vector, vector, shuffleMask);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ shuffle(
        _VectorType_                                            vector,
        _VectorType_                                            secondVector,
        type_traits::__deduce_simd_shuffle_mask_type<
        sizeof(_VectorType_) / sizeof(_DesiredType_)>       shuffleMask) noexcept
    {
        if      constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_> || is_pd_v<_DesiredType_>)
            return cast<__m128d, _VectorType_>(_mm_shuffle_pd(
                cast<_VectorType_, __m128d>(vector),
                cast<_VectorType_, __m128d>(secondVector),
                shuffleMask)
            );
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_> || is_ps_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_shuffle_epi32(
                cast<_VectorType_, __m128i>(vector),
                shuffleMask)
            );
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>) {

        }
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>) {

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

__SIMD_STL_NUMERIC_NANESPACE_END
