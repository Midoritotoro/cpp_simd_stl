#pragma once 

#include <src/simd_stl/numeric/SimdElementWise.h>
#include <simd_stl/numeric/BasicSimdMask.h>

#include <src/simd_stl/algorithm/AdvanceBytes.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <arch::CpuFeature _SimdGeneration_>
class SimdMemoryAccess;

namespace detail::SSE {
    template <
        sizetype _VerticalSize_,
        sizetype _HorizontalSize_>
    struct ShuffleTables {
        uint8 shuffle[_VerticalSize_][_HorizontalSize_];
        uint8 size[_VerticalSize_];
    };

    template <
        sizetype _VerticalSize_,
        sizetype _HorizontalSize_>
    constexpr auto makeShuffleTables(
        const uint32 multiplier, 
        const uint32 elementGroupStride) noexcept
    {
        ShuffleTables<_VerticalSize_, _HorizontalSize_> result;

        for (uint32 verticalIndex = 0; verticalIndex != _VerticalSize_; ++verticalIndex) {
            uint32 activeGroupCount = 0;

            for (uint32 horizontalIndex = 0; horizontalIndex != _HorizontalSize_ / elementGroupStride; ++horizontalIndex) {
                if ((verticalIndex & (1 << horizontalIndex)) == 0) {
                    for (uint32 elementOffset = 0; elementOffset != elementGroupStride; ++elementOffset)
                        result.shuffle[verticalIndex][activeGroupCount * elementGroupStride + elementOffset] = 
                            static_cast<uint8>(horizontalIndex * elementGroupStride + elementOffset);
                    
                    ++activeGroupCount;
                }
            }

            result.size[verticalIndex] = static_cast<uint8>(activeGroupCount * multiplier);

         
            for (; activeGroupCount != _HorizontalSize_ / elementGroupStride; ++activeGroupCount)
                for (uint32 elementOffset = 0; elementOffset != elementGroupStride; ++elementOffset)
                    result.shuffle[verticalIndex][activeGroupCount * elementGroupStride + elementOffset] = 
                        static_cast<uint8>(activeGroupCount * elementGroupStride + elementOffset);
        }

        return result;
    }

    constexpr auto tables8Bit   = makeShuffleTables<256, 8>(1, 1);
    constexpr auto tables16Bit  = makeShuffleTables<256, 16>(2, 2);
    constexpr auto tables32Bit  = makeShuffleTables<16, 16>(4, 4);
    constexpr auto tables64Bit  = makeShuffleTables<4, 16>(8, 8);
} // namespace detail


template <>
class SimdMemoryAccess<arch::CpuFeature::SSE2> {
    static constexpr auto _Feature = arch::CpuFeature::SSE2;

    using _ElementWise_ = SimdElementWise<_Feature>;
    using _SimdCast_    = SimdCast<_Feature>;
    using _SimdConvert_ = SimdConvert<_Feature>;
public:
    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ loadUpperHalf(const void* where) noexcept {
        return _SimdCast_::template cast<__m128d, _VectorType_>(_mm_loadh_pd(_mm_setzero_pd(), static_cast<const double*>(where)));
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ loadLowerHalf(const void* where) noexcept {
        return _SimdCast_::template cast<__m128d, _VectorType_>(_mm_loadl_pd(_mm_setzero_pd(), static_cast<const double*>(where)));
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
        const auto loaded   = loadUnaligned<_VectorType_>(where);
        const auto blended  = _ElementWise_::template blend<_DesiredType_>(loaded, vector, mask);

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
        const auto loaded   = loadAligned<_VectorType_>(where);
        const auto blended  = _ElementWise_::template blend<_DesiredType_>(loaded, vector, mask);

        storeAligned(where, blended);
    }

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ maskLoadUnaligned(
        const _DesiredType_*                                                where,
        const type_traits::__deduce_simd_mask_type<_Feature, _DesiredType_> mask) noexcept
    {
        const auto zeros    = _mm_setzero_si128();
        const auto loaded   = _mm_loadu_si128(reinterpret_cast<const __m128i*>(where));

        const auto blended  = _ElementWise_::template blend<_DesiredType_>(
            _SimdCast_::template cast<__m128i, _VectorType_>(zeros), 
            _SimdCast_::template cast<__m128i, _VectorType_>(loaded), mask);

        return blended;
    }

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ maskLoadAligned(
        const _DesiredType_*                                                where,
        const type_traits::__deduce_simd_mask_type<_Feature, _DesiredType_> mask) noexcept
    {
        const auto zeros    = _mm_setzero_si128();
        const auto loaded   = _mm_load_si128(reinterpret_cast<const __m128i*>(where));

        const auto blended  = _ElementWise_::template blend<_DesiredType_>(
            _SimdCast_::template cast<__m128i, _VectorType_>(zeros), 
            _SimdCast_::template cast<__m128i, _VectorType_>(loaded), mask);

        return blended;
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* compressStoreUnaligned(
        _DesiredType_*                                                      where,
        type_traits::__deduce_simd_mask_type<_Feature, _DesiredType_>       mask,
        const _VectorType_                                                  vector) noexcept
    {
        __m128i shuffle;

        
        return where;
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* compressStoreAligned(
        _DesiredType_*                                                      where,
        const type_traits::__deduce_simd_mask_type<_Feature, _DesiredType_> mask,
        const _VectorType_                                                  vector) noexcept
    {
        __m128i shuffle;

        if      constexpr (sizeof(_DesiredType_) == 8)
            shuffle = _mm_loadu_si128(reinterpret_cast<const __m128i*>(detail::SSE::tables64Bit.shuffle[mask]));
        else if constexpr (sizeof(_DesiredType_) == 4)
            shuffle = _mm_loadu_si128(reinterpret_cast<const __m128i*>(detail::SSE::tables32Bit.shuffle[mask]));
        else if constexpr (sizeof(_DesiredType_) == 2)
            shuffle = _mm_loadu_si128(reinterpret_cast<const __m128i*>(detail::SSE::tables16Bit.shuffle[mask]));
        else if constexpr (sizeof(_DesiredType_) == 1)
            shuffle = _mm_loadu_si128(reinterpret_cast<const __m128i*>(detail::SSE::tables8Bit.shuffle[mask]));

        const auto destination  = _ElementWise_::template shuffle<uint8>(
            _SimdCast_::template cast<_VectorType_, __m128i>(vector),
            _SimdConvert_::template convertToMask<_DesiredType_>(shuffle));

        _mm_store_si128(reinterpret_cast<__m128i*>(where), destination);

        if      constexpr (sizeof(_DesiredType_) == 8)
            algorithm::AdvanceBytes(where, detail::SSE::tables64Bit.size[mask]);
        else if constexpr (sizeof(_DesiredType_) == 4)
            algorithm::AdvanceBytes(where, detail::SSE::tables32Bit.size[mask]);
        else if constexpr (sizeof(_DesiredType_) == 2)
            algorithm::AdvanceBytes(where, detail::SSE::tables16Bit.size[mask]);
        else if constexpr (sizeof(_DesiredType_) == 1) 
            algorithm::AdvanceBytes(where, detail::SSE::tables8Bit.size[mask]);

        return where;
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* compressStoreMergeUnaligned(
        _DesiredType_* where,
        const type_traits::__deduce_simd_mask_type<_Feature, _DesiredType_> mask,
        const _VectorType_                                                  vector,
        const _VectorType_                                                  sourceVector) noexcept
    {
        basic_simd_mask<_Feature, _DesiredType_> simdMask = mask;

        if (simdMask.allOf() == false)
            storeUnaligned(where, sourceVector);

        return compressStoreUnaligned(where, mask, vector);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* compressStoreMergeAligned(
        _DesiredType_* where,
        const type_traits::__deduce_simd_mask_type<_Feature, _DesiredType_> mask,
        const _VectorType_                                                  vector,
        const _VectorType_                                                  sourceVector) noexcept
    {
        basic_simd_mask<_Feature, _DesiredType_> simdMask = mask;

        if (simdMask.allOf() == false)
            storeAligned(where, sourceVector);

        return compressStoreAligned(where, mask, vector);
    }
};

template <>
class SimdMemoryAccess<arch::CpuFeature::SSE3> :
    public SimdMemoryAccess<arch::CpuFeature::SSE2>
{};

template <>
class SimdMemoryAccess<arch::CpuFeature::SSSE3> :
    public SimdMemoryAccess<arch::CpuFeature::SSE3>
{
    static constexpr auto _Feature = arch::CpuFeature::SSSE3;
    
    using _Cast_        = SimdCast<_Feature>;
    using _ElementWise_ = SimdElementWise<_Feature>;
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* compressStoreLowerHalf(
        _DesiredType_*                                                      where,
        const type_traits::__deduce_simd_mask_type<_Feature, _DesiredType_> mask,
        const _VectorType_                                                  vector) noexcept
    {
        static_assert(sizeof(_DesiredType_) != 8);

        __m128i shuffle;

        if constexpr (sizeof(_DesiredType_) == 4)
            shuffle = loadLowerHalf<__m128i>(detail::SSE::tables32Bit.shuffle[mask]);
        else if constexpr (sizeof(_DesiredType_) == 2)
            shuffle = loadLowerHalf<__m128i>(detail::SSE::tables16Bit.shuffle[mask]);
        else if constexpr (sizeof(_DesiredType_) == 1)
            shuffle = loadLowerHalf<__m128i>(detail::SSE::tables8Bit.shuffle[mask]);

        const auto destination  = _mm_shuffle_epi8(_Cast_::template cast<_VectorType_, __m128i>(vector), shuffle);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(where), destination);

        if constexpr (sizeof(_DesiredType_) == 4)
            algorithm::AdvanceBytes(where, detail::SSE::tables32Bit.size[mask]);
        else if constexpr (sizeof(_DesiredType_) == 2)
            algorithm::AdvanceBytes(where, detail::SSE::tables16Bit.size[mask]);
        else if constexpr (sizeof(_DesiredType_) == 1)
            algorithm::AdvanceBytes(where, detail::SSE::tables8Bit.size[mask]);

        return where;
    }
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* compressStoreUpperHalf(
        _DesiredType_*                                                      where,
        const type_traits::__deduce_simd_mask_type<_Feature, _DesiredType_> mask,
        const _VectorType_                                                  vector) noexcept
    {
        static_assert(sizeof(_DesiredType_) != 8);
        __m128i shuffle;

       if constexpr (sizeof(_DesiredType_) == 4)
            shuffle = loadUpperHalf<__m128i>(detail::SSE::tables32Bit.shuffle[mask]);
        else if constexpr (sizeof(_DesiredType_) == 2)
            shuffle = loadUpperHalf<__m128i>(detail::SSE::tables16Bit.shuffle[mask]);
        else if constexpr (sizeof(_DesiredType_) == 1)
            shuffle = loadUpperHalf<__m128i>(detail::SSE::tables8Bit.shuffle[mask]);

        const auto destination  = _mm_shuffle_epi8(_Cast_::template cast<_VectorType_, __m128i>(vector), shuffle);
        _mm_storeh_pd(reinterpret_cast<double*>(where), _Cast_::template cast<__m128i, __m128d>(destination));

        if constexpr (sizeof(_DesiredType_) == 4)
            algorithm::AdvanceBytes(where, detail::SSE::tables32Bit.size[mask]);
        else if constexpr (sizeof(_DesiredType_) == 2)
            algorithm::AdvanceBytes(where, detail::SSE::tables16Bit.size[mask]);
        else if constexpr (sizeof(_DesiredType_) == 1)
            algorithm::AdvanceBytes(where, detail::SSE::tables8Bit.size[mask]);

        return where;
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* compressStoreUnaligned(
        _DesiredType_*                                                      where,
        const type_traits::__deduce_simd_mask_type<_Feature, _DesiredType_> mask,
        const _VectorType_                                                  vector) noexcept
    {
        if      constexpr (sizeof(_DesiredType_) == 8) {
            auto shuffle = _mm_loadu_si128(reinterpret_cast<const __m128i*>(detail::SSE::tables64Bit.shuffle[mask]));

            const auto destination = _mm_shuffle_epi8(_Cast_::template cast<_VectorType_, __m128i>(vector), shuffle);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(where), destination);

            algorithm::AdvanceBytes(where, detail::SSE::tables64Bit.size[mask]);
        }
        else if constexpr (sizeof(_DesiredType_) == 4) {
            auto shuffle = _mm_loadu_si128(reinterpret_cast<const __m128i*>(detail::SSE::tables32Bit.shuffle[mask]));

            const auto destination = _mm_shuffle_epi8(_Cast_::template cast<_VectorType_, __m128i>(vector), shuffle);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(where), destination);

            algorithm::AdvanceBytes(where, detail::SSE::tables32Bit.size[mask]);
        }
        else if constexpr (sizeof(_DesiredType_) == 2) {
            auto shuffle = _mm_loadu_si128(reinterpret_cast<const __m128i*>(detail::SSE::tables16Bit.shuffle[mask]));

            const auto destination = _mm_shuffle_epi8(_Cast_::template cast<_VectorType_, __m128i>(vector), shuffle);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(where), destination);

            algorithm::AdvanceBytes(where, detail::SSE::tables16Bit.size[mask]);
        }
        else if constexpr (sizeof(_DesiredType_) == 1) {
            const auto start = where;


            const auto shifted = _mm_movehl_ps(
                _Cast_::template cast<__m128i, __m128>(_mm_slli_si128(_Cast_::template cast<_VectorType_, __m128i>(vector), 8)),
                _Cast_::template cast<_VectorType_, __m128>(vector));

            where = compressStoreLowerHalf(where, mask & 0xFF, vector);
            where = compressStoreLowerHalf(where, (mask >> 8) & 0xFF, shifted);

            const auto remainingElements = sizeof(__m128i) - (where - start);

            auto mask = (1u << remainingElements) - 1u;
            mask = ~mask;

            maskStoreUnaligned(start, mask, vector);
        }

        return where;
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* compressStoreAligned(
        _DesiredType_*                                                      where,
        const type_traits::__deduce_simd_mask_type<_Feature, _DesiredType_> mask,
        const _VectorType_                                                  vector) noexcept
    {
        if      constexpr (sizeof(_DesiredType_) == 8) {
            auto shuffle = _mm_loadu_si128(reinterpret_cast<const __m128i*>(detail::SSE::tables64Bit.shuffle[mask]));

            const auto destination = _mm_shuffle_epi8(_Cast_::template cast<_VectorType_, __m128i>(vector), shuffle);
            _mm_store_si128(reinterpret_cast<__m128i*>(where), destination);

            algorithm::AdvanceBytes(where, detail::SSE::tables64Bit.size[mask]);
        }
        else if constexpr (sizeof(_DesiredType_) == 4) {
            auto shuffle = _mm_loadu_si128(reinterpret_cast<const __m128i*>(detail::SSE::tables32Bit.shuffle[mask]));

            const auto destination = _mm_shuffle_epi8(_Cast_::template cast<_VectorType_, __m128i>(vector), shuffle);
            _mm_store_si128(reinterpret_cast<__m128i*>(where), destination);

            algorithm::AdvanceBytes(where, detail::SSE::tables32Bit.size[mask]);
        }
        else if constexpr (sizeof(_DesiredType_) == 2) {
            auto shuffle = _mm_loadu_si128(reinterpret_cast<const __m128i*>(detail::SSE::tables16Bit.shuffle[mask]));

            const auto destination = _mm_shuffle_epi8(_Cast_::template cast<_VectorType_, __m128i>(vector), shuffle);
            _mm_store_si128(reinterpret_cast<__m128i*>(where), destination);

            algorithm::AdvanceBytes(where, detail::SSE::tables16Bit.size[mask]);
        }
        else if constexpr (sizeof(_DesiredType_) == 1) {
            const auto start = where;

            const auto shifted = _mm_movehl_ps(
                _Cast_::template cast<__m128i, __m128>(_mm_slli_si128(_Cast_::template cast<_VectorType_, __m128i>(vector), 8)),
                _Cast_::template cast<_VectorType_, __m128>(vector));

            where = compressStoreLowerHalf(where, mask & 0xFF, vector);
            where = compressStoreLowerHalf(where, (mask >> 8) & 0xFF, shifted);

            const auto remainingElements = sizeof(__m128i) - (where - start);

            auto mask = (1u << remainingElements) - 1u;
            mask = ~mask;

            maskStoreAligned(start, mask, vector);
        }

        return where;
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* compressStoreMergeUnaligned(
        _DesiredType_*                                                      where,
        const type_traits::__deduce_simd_mask_type<_Feature, _DesiredType_> mask,
        const _VectorType_                                                  vector,
        const _VectorType_                                                  sourceVector) noexcept
    {
        if      constexpr (sizeof(_DesiredType_) == 8) {
            auto shuffle = _mm_loadu_si128(reinterpret_cast<const __m128i*>(detail::SSE::tables64Bit.shuffle[mask]));

            const auto destination = _mm_shuffle_epi8(_Cast_::template cast<_VectorType_, __m128i>(vector), shuffle);
            _mm_store_si128(reinterpret_cast<__m128i*>(where), destination);

            algorithm::AdvanceBytes(where, detail::SSE::tables64Bit.size[mask]);
        }
        else if constexpr (sizeof(_DesiredType_) == 4) {
            auto shuffle = _mm_loadu_si128(reinterpret_cast<const __m128i*>(detail::SSE::tables32Bit.shuffle[mask]));

            const auto destination = _mm_shuffle_epi8(_Cast_::template cast<_VectorType_, __m128i>(vector), shuffle);
            _mm_store_si128(reinterpret_cast<__m128i*>(where), destination);

            algorithm::AdvanceBytes(where, detail::SSE::tables32Bit.size[mask]);
        }
        else if constexpr (sizeof(_DesiredType_) == 2) {
            auto shuffle = _mm_loadu_si128(reinterpret_cast<const __m128i*>(detail::SSE::tables16Bit.shuffle[mask]));

            const auto destination = _mm_shuffle_epi8(_Cast_::template cast<_VectorType_, __m128i>(vector), shuffle);
            _mm_store_si128(reinterpret_cast<__m128i*>(where), destination);

            algorithm::AdvanceBytes(where, detail::SSE::tables16Bit.size[mask]);
        }
        else if constexpr (sizeof(_DesiredType_) == 1) {
            const auto start = where;

            const auto shifted = _mm_movehl_ps(
                _Cast_::template cast<__m128i, __m128>(_mm_slli_si128(_Cast_::template cast<_VectorType_, __m128i>(vector), 8)),
                _Cast_::template cast<_VectorType_, __m128>(vector));

            where = compressStoreLowerHalf(where, mask & 0xFF, vector);
            where = compressStoreLowerHalf(where, (mask >> 8) & 0xFF, shifted);

            const auto remainingElements = sizeof(__m128i) - (where - start);

            auto mask = (1u << remainingElements) - 1u;
            mask = ~mask;

            maskStoreAligned(start, mask, vector);
        }

        return where;
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* compressStoreMergeAligned(
        _DesiredType_*                                                      where,
        const type_traits::__deduce_simd_mask_type<_Feature, _DesiredType_> mask,
        const _VectorType_                                                  vector,
        const _VectorType_                                                  sourceVector) noexcept
    {
        return where;
    }
};

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
