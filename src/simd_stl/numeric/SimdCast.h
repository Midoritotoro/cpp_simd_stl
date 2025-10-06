#pragma once 

#include <src/simd_stl/numeric/SimdIntegralTypesCheck.h>
#include <src/simd_stl/type_traits/SimdTypeCheck.h>

#include <simd_stl/compatibility/Inline.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <arch::CpuFeature _SimdGeneration_>
class SimdCast;

template <>
class SimdCast<arch::CpuFeature::SSE2> {
public:
    template <
        typename    _FromVector_,
        typename    _ToVector_,
        bool        _SafeCast_ = false>
    static simd_stl_always_inline _ToVector_ cast(_FromVector_ from) noexcept {
        if constexpr (std::is_same_v<_FromVector_, _ToVector_>)
            return from;

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
    }
};

template <>
class SimdCast<arch::CpuFeature::SSE3>:
    public SimdCast<arch::CpuFeature::SSE2>
{};

template <>
class SimdCast<arch::CpuFeature::SSSE3> :
    public SimdCast<arch::CpuFeature::SSE3>
{};

template <>
class SimdCast<arch::CpuFeature::SSE41> :
    public SimdCast<arch::CpuFeature::SSSE3>
{};

template <>
class SimdCast<arch::CpuFeature::SSE42> :
    public SimdCast<arch::CpuFeature::SSE41>
{};

__SIMD_STL_NUMERIC_NAMESPACE_END
