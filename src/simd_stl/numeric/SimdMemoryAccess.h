#pragma once 

#include <src/simd_stl/numeric/SimdElementWise.h>

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <arch::CpuFeature _SimdGeneration_>
class SimdMemoryAccess;

template <>
class SimdMemoryAccess<arch::CpuFeature::SSE2> {
    template <typename _IntegralSimdElementsMask_> 
    using shuffle_mask_type = std::conditional_t<
        sizeof(_IntegralSimdElementsMask_) == 1, unsigned int,
            std::conditional_t<sizeof(_IntegralSimdElementsMask_) == 2, unsigned long long, void>>;

    template <
        size_t      _ElementsCount_,
        typename    _IntegralSimdElementsMask_>
    inline shuffle_mask_type<_IntegralSimdElementsMask_> toShuffleMask(typename std::type_identity<_IntegralSimdElementsMask_>::type mask) noexcept { 
        using _ResultType_ = shuffle_mask_type<_IntegralSimdElementsMask_>;

        auto result = _ResultType_(0);
        auto destinationOffset = 0;

        constexpr auto step = math::CountTrailingZeroBits(_ElementsCount_);

        for (auto current = 0; current < _ElementsCount_; ++current) {
            if ((mask >> current) & 1) {
                result |= static_cast<_ResultType_>(current & (_ElementsCount_ - 1)) << destinationOffset;
                destinationOffset += step;
            }
        }

        return result;
    }

    static constexpr auto _Feature = arch::CpuFeature::SSE2;

    using _ElementWise_ = SimdElementWise<_Feature>;
    using _SimdCast_    = SimdCast<_Feature>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void compressStoreUnaligned(
        void*                                                           where,
        type_traits::__deduce_simd_mask_type<_Feature, _DesiredType_>   mask,
        _VectorType_                                                    vector) noexcept
    {
        
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ nonTemporalLoad(const void* where) noexcept {
        return loadAligned<_VectorType_, void>(where);
    }

    template <typename _VectorType_>
    static simd_stl_always_inline void nonTemporalStore(
        void*           where,
        _VectorType_    vector) noexcept 
    {
        _mm_stream_si128(static_cast<__m128i*>(where), _SimdCast_::cast<_VectorType_, __m128i>(vector));
    }

    static simd_stl_always_inline void streamingFence() noexcept {
        return _mm_sfence();
    }

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
        _DesiredType_*      where,
        const _VectorType_  vector) noexcept
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
        _DesiredType_*                                                      where,
        const type_traits::__deduce_simd_mask_type<_Feature, _DesiredType_> mask,
        const _VectorType_                                                  vector) noexcept
    {
        const auto loaded   = loadUnaligned(where);
        const auto blended  = _ElementWise_::template blend<_DesiredType_>(vector, loaded, mask);

        storeUnaligned(where, blended);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void maskStoreAligned(
        _DesiredType_*                                                      where,
        const type_traits::__deduce_simd_mask_type<_Feature, _DesiredType_> mask,
        const _VectorType_                                                  vector) noexcept
    {
        
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ maskLoadUnaligned(
        const _DesiredType_*                                                where,
        const type_traits::__deduce_simd_mask_type<_Feature, _DesiredType_> mask,
        const _VectorType_                                                  vector) noexcept
    {
        
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void maskLoadAligned(
        const _DesiredType_*                                                where,
        const type_traits::__deduce_simd_mask_type<_Feature, _DesiredType_> mask,
        _VectorType_                                                        vector) noexcept
    {

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
{
public:
    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ nonTemporalLoad(const void* where) noexcept {
        return _SimdCast_::cast<__m128i, _VectorType_>(_mm_stream_load_si128(static_cast<const __m128i*>(where)));
    }
};

template <>
class SimdMemoryAccess<arch::CpuFeature::SSE42> :
    public SimdMemoryAccess<arch::CpuFeature::SSE41>
{};

__SIMD_STL_NUMERIC_NAMESPACE_END
