#pragma once 

#include <src/simd_stl/numeric/SimdElementWise.h>
#include <simd_stl/numeric/BasicSimdMask.h>

#include <src/simd_stl/algorithm/AdvanceBytes.h>
#include <src/simd_stl/numeric/IntrinBitcast.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_>
class _SimdMemoryAccess;


template <class _RegisterPolicy_>
class _SimdMemoryAccess<arch::CpuFeature::SSE2, _RegisterPolicy_> {
    template <class _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<arch::CpuFeature::SSE2, _DesiredType_, _RegisterPolicy_>;
public:
    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _LoadUpperHalf(const void* _Where) noexcept {
        return _IntrinBitcast<_VectorType_>(_mm_loadh_pd(_mm_setzero_pd(), static_cast<const double*>(_Where)));
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _LoadLowerHalf(const void* _Where) noexcept {
        return _IntrinBitcast<_VectorType_>(_mm_loadl_pd(_mm_setzero_pd(), static_cast<const double*>(_Where)));
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _NonTemporalLoad(const void* _Where) noexcept {
        return loadAligned<_VectorType_, void>(_Where);
    }

    template <typename _VectorType_>
    static simd_stl_always_inline void _NonTemporalStore(
        void*           _Where,
        _VectorType_    _Vector) noexcept 
    {
        _mm_stream_si128(static_cast<__m128i*>(_Where), _IntrinBitcast<__m128i>(_Vector));
    }

    static simd_stl_always_inline void _StreamingFence() noexcept {
        return _mm_sfence();
    }

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _LoadUnaligned(const _DesiredType_* _Where) noexcept {
        if      constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_loadu_si128(reinterpret_cast<const __m128i*>(_Where));
        else if constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_loadu_pd(reinterpret_cast<const double*>(_Where));
        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_loadu_ps(reinterpret_cast<const float*>(_Where));
    }

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _LoadAligned(const _DesiredType_* _Where) noexcept {
        if      constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_load_si128(reinterpret_cast<const __m128i*>(_Where));
        else if constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_load_pd(reinterpret_cast<const double*>(_Where));
        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_load_ps(reinterpret_cast<const float*>(_Where));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _StoreUnaligned(
        _DesiredType_*      _Where,
        const _VectorType_  _Vector) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_storeu_si128(reinterpret_cast<__m128i*>(_Where), _Vector);
        else if constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_storeu_pd(reinterpret_cast<double*>(_Where), _Vector);
        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_storeu_ps(reinterpret_cast<float*>(_Where), _Vector);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _StoreAligned(
        _DesiredType_*          _Where,
        const _VectorType_      _Vector) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_store_si128(reinterpret_cast<__m128i*>(_Where), _Vector);
        else if constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_store_pd(reinterpret_cast<double*>(_Where), _Vector);
        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_store_ps(reinterpret_cast<float*>(_Where), _Vector);
    }


    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _MaskStoreUnaligned(
        _DesiredType_*                          _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector) noexcept
    {
        const auto _Loaded   = _LoadUnaligned<_VectorType_>(_Where);
        const auto _Blended  = _ElementWise_::template blend<_DesiredType_>(_Loaded, _Vector, _Mask);

        _StoreUnaligned(_Where, _Blended);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void _MaskStoreAligned(
        _DesiredType_*                          _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask,
        const _VectorType_                      _Vector) noexcept
    {
        const auto _Loaded   = loadAligned<_VectorType_>(_Where);
        const auto _Blended  = _ElementWise_::template blend<_DesiredType_>(_Loaded, _Vector, _Mask);

        _StoreAligned(_Where, _Blended);
    }

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ _MaskLoadUnaligned(
        const _DesiredType_*                    _Where,
        const _Simd_mask_type<_DesiredType_>    _Mask) noexcept
    {
        return _ElementWise_::template blend<_DesiredType_>(
            _IntrinBitcast<_VectorType_>(_mm_setzero_si128()), 
            _LoadUnaligned<_VectorType_>(_Where), _Mask);
    }

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ maskLoadAligned(
        const _DesiredType_*                                                                    where,
        const type_traits::__deduce_simd_mask_type<_Feature, _DesiredType_, _RegisterPolicy_>   mask) noexcept
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
        type_traits::__deduce_simd_mask_type<_Feature, _DesiredType_, _RegisterPolicy_>       mask,
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
        const type_traits::__deduce_simd_mask_type<_Feature, _DesiredType_, _RegisterPolicy_> mask,
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
        const type_traits::__deduce_simd_mask_type<_Feature, _DesiredType_, _RegisterPolicy_> mask,
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
        const type_traits::__deduce_simd_mask_type<_Feature, _DesiredType_, _RegisterPolicy_> mask,
        const _VectorType_                                                  vector,
        const _VectorType_                                                  sourceVector) noexcept
    {
        basic_simd_mask<_Feature, _DesiredType_> simdMask = mask;

        if (simdMask.allOf() == false)
            storeAligned(where, sourceVector);

        return compressStoreAligned(where, mask, vector);
    }
};

template <class _RegisterPolicy_>
class _SimdMemoryAccess<arch::CpuFeature::SSE3, _RegisterPolicy_> :
    public _SimdMemoryAccess<arch::CpuFeature::SSE2, _RegisterPolicy_>
{};

template <class _RegisterPolicy_>
class _SimdMemoryAccess<arch::CpuFeature::SSSE3, _RegisterPolicy_> :
    public _SimdMemoryAccess<arch::CpuFeature::SSE3, _RegisterPolicy_>
{
    static constexpr auto _Feature = arch::CpuFeature::SSSE3;
    
    using _Cast_        = _SimdCast<_Feature, _RegisterPolicy_>;
    using _ElementWise_ = _SimdElementWise<_Feature, _RegisterPolicy_>;
public:
    using _SimdMemoryAccess<arch::CpuFeature::SSE3, _RegisterPolicy_>::maskStoreUnaligned;

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* compressStoreLowerHalf(
        _DesiredType_*                                                      where,
        const type_traits::__deduce_simd_mask_type<_Feature, _DesiredType_, _RegisterPolicy_> mask,
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
        const type_traits::__deduce_simd_mask_type<_Feature, _DesiredType_, _RegisterPolicy_> mask,
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
        const type_traits::__deduce_simd_mask_type<_Feature, _DesiredType_, _RegisterPolicy_> mask,
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
            auto start = where;


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
        const type_traits::__deduce_simd_mask_type<_Feature, _DesiredType_, _RegisterPolicy_> mask,
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
            auto start = where;

            const auto shifted = _mm_movehl_ps(
                _Cast_::template cast<__m128i, __m128>(_mm_slli_si128(_Cast_::template cast<_VectorType_, __m128i>(vector), 8)),
                _Cast_::template cast<_VectorType_, __m128>(vector));

            where = compressStoreLowerHalf(where, mask & 0xFF, vector);
            where = compressStoreLowerHalf(where, (mask >> 8) & 0xFF, shifted);

            const auto remainingElements = sizeof(__m128i) - (where - start);

            auto mask = (1u << remainingElements) - 1u;
            mask = ~mask;

           // maskStoreAligned<_DesiredType_>(start, mask, vector);
        }

        return where;
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* compressStoreMergeUnaligned(
        _DesiredType_*                                                      where,
        const type_traits::__deduce_simd_mask_type<_Feature, _DesiredType_, _RegisterPolicy_> mask,
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

            maskStoreUnaligned(start, mask, vector);
        }

        return where;
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_* compressStoreMergeAligned(
        _DesiredType_*                                                      where,
        const type_traits::__deduce_simd_mask_type<_Feature, _DesiredType_, _RegisterPolicy_> mask,
        const _VectorType_                                                  vector,
        const _VectorType_                                                  sourceVector) noexcept
    {
        return where;
    }
};

template <class _RegisterPolicy_>
class _SimdMemoryAccess<arch::CpuFeature::SSE41, _RegisterPolicy_> :
    public _SimdMemoryAccess<arch::CpuFeature::SSSE3, _RegisterPolicy_>
{
    using _Cast_ = _SimdCast<arch::CpuFeature::SSE41, _RegisterPolicy_>;
public:

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ nonTemporalLoad(const void* where) noexcept {
        return _Cast_::template cast<__m128i, _VectorType_>(_mm_stream_load_si128(static_cast<const __m128i*>(where)));
    }
};

template <class _RegisterPolicy_>
class _SimdMemoryAccess<arch::CpuFeature::SSE42, _RegisterPolicy_> :
    public _SimdMemoryAccess<arch::CpuFeature::SSE41, _RegisterPolicy_>
{};

__SIMD_STL_NUMERIC_NAMESPACE_END
