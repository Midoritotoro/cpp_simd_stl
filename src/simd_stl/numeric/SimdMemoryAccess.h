#pragma once 

#include <src/simd_stl/numeric/SimdElementWise.h>

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <arch::CpuFeature _SimdGeneration_>
class SimdMemoryAccess;

template <>
class SimdMemoryAccess<arch::CpuFeature::SSE2> {
    using _ElementWise_ = SimdElementWise<arch::CpuFeature::SSE2>;
public:
    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ loadUnaligned(const _DesiredType_* where) noexcept {
        if      constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_loadu_si128(reinterpret_cast<const __m128i*>(where));
        else if constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_loadu_pd(reinterpret_cast<const double*>(where));
        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_loadu_ps(reinterpret_cast<const float*>(where));
    }

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ loadAligned(const _DesiredType_* where) noexcept {
        if      constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_load_si128(reinterpret_cast<const __m128i*>(where));
        else if constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_load_pd(reinterpret_cast<const double*>(where));
        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_load_ps(reinterpret_cast<const float*>(where));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void storeUnaligned(
        _DesiredType_* where,
        const _VectorType_      vector) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_storeu_si128(reinterpret_cast<__m128i*>(where), vector);
        else if constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_storeu_pd(reinterpret_cast<double*>(where), vector);
        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_storeu_ps(reinterpret_cast<float*>(where), vector);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void storeAligned(
        _DesiredType_* where,
        const _VectorType_      vector) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_store_si128(reinterpret_cast<__m128i*>(where), vector);
        else if constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_store_pd(reinterpret_cast<double*>(where), vector);
        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_store_ps(reinterpret_cast<float*>(where), vector);
    }


    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void maskStoreUnaligned(
        _DesiredType_*                                      where,
        const type_traits::__deduce_simd_shuffle_mask_type<arch::CpuFeature::SSE2, _DesiredType_>   mask,
        const _VectorType_                                  vector) noexcept
    {
        storeUnaligned(where, _ElementWise_::template shuffle<_DesiredType_>(
            loadUnaligned<_VectorType_>(where), vector, mask));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void maskStoreAligned(
        _DesiredType_*                                      where,
        const type_traits::__deduce_simd_shuffle_mask_type<arch::CpuFeature::SSE2, _DesiredType_>   mask,
        const _VectorType_                                  vector) noexcept
    {
        storeAligned(where, _ElementWise_::template shuffle<_DesiredType_>(
            loadAligned<_VectorType_>(where), vector, mask));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ maskLoadUnaligned(
        const _DesiredType_*                                where,
        const type_traits::__deduce_simd_shuffle_mask_type<arch::CpuFeature::SSE2, _DesiredType_>   mask,
        const _VectorType_                                  vector) noexcept
    {
        return _ElementWise_::template shuffle<_DesiredType_>(
            loadUnaligned<_VectorType_>(where), vector, mask);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void maskLoadAligned(
        const _DesiredType_*                                where,
        const type_traits::__deduce_simd_shuffle_mask_type<arch::CpuFeature::SSE2, _DesiredType_>  mask,
        _VectorType_                                        vector) noexcept
    {
        return _ElementWise_::template shuffle<_DesiredType_>(
            loadAligned<_VectorType_>(where), vector, mask);
    }
};

template <>
class SimdMemoryAccess<arch::CpuFeature::SSE3> :
    public SimdMemoryAccess<arch::CpuFeature::SSE2>
{};

template <>
class SimdMemoryAccess<arch::CpuFeature::SSSE3> :
    public SimdMemoryAccess<arch::CpuFeature::SSE3>
{};

template <>
class SimdMemoryAccess<arch::CpuFeature::SSE41> :
    public SimdMemoryAccess<arch::CpuFeature::SSSE3>
{};

template <>
class SimdMemoryAccess<arch::CpuFeature::SSE42> :
    public SimdMemoryAccess<arch::CpuFeature::SSE41>
{};

__SIMD_STL_NUMERIC_NAMESPACE_END
