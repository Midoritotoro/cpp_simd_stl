#pragma once 

#include <src/simd_stl/numeric/BasicSimdImplementationUnspecialized.h>

#include <src/simd_stl/type_traits/TypeTraits.h>
#include <simd_stl/compatibility/Inline.h>

#include <simd_stl/numeric/BasicSimdMask.h>
#include <simd_stl/numeric/BasicSimdShuffleMask.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <>
class BasicSimdImplementation<arch::CpuFeature::AVX512F> {
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
        _VectorType_                                            vectorSecond,
        type_traits::__deduce_simd_shuffle_mask_type<
        sizeof(_VectorType_) / sizeof(_DesiredType_)>       shuffleMask) noexcept
    {
        if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_> || is_pd_v<_DesiredType_>)
            return cast<__m512d, _VectorType_>(
                _mm512_shuffle_pd(
                    cast<_VectorType_, __m512d>(vector),
                    cast<_VectorType_, __m512d>(vectorSecond),
                    shuffleMask));
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_> || is_ps_v<_DesiredType_>)
            return cast<__m512, _VectorType_>(_mm512_shuffle_ps(
                cast<_VectorType_, __m512>(vector),
                cast<_VectorType_, __m512>(vectorSecond),
                shuffleMask
            ));
        else if constexpr (
            is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_> ||
            is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>)
        {
            const auto shuffled1 = BasicSimdImplementation<arch::CpuFeature::AVX2>::template shuffle
                <_DesiredType_>(
                    cast<_VectorType_, __m256i>(vector),
                    cast<_VectorType_, __m256i>(vectorSecond),
                    shuffleMask
                );

            const auto shuffled2 = BasicSimdImplementation<arch::CpuFeature::AVX2>::template shuffle
                <_DesiredType_>(
                    _mm512_extracti32x8_epi32(vector, 1),
                    _mm512_extracti32x8_epi32(vectorSecond, 1),
                    shuffleMask
                );

            return cast<__m512i, _VectorType_>(
                _mm512_inserti32x8(cast<__m256i, __m512i>(shuffled1), shuffled2, 1));
        }
    }
    //
    //    
    //    template <
    //        typename _DesiredVectorElementType_,
    //        typename _VectorType_>
    //    static simd_stl_always_inline int32 maskFromVector(_VectorType_ vector) noexcept {
    //        if constexpr (is_pd_v<_DesiredVectorElementType_> || is_epi64_v<_DesiredVectorElementType_> || is_epu64_v<_DesiredVectorElementType_>)
    //        {
    //        }
    //        else if constexpr (is_ps_v<_DesiredVectorElementType_> || is_epi32_v<_DesiredVectorElementType_> || is_epu32_v<_DesiredVectorElementType_>)
    //        {
    //        }
    //        else if constexpr (is_epi16_v<_DesiredVectorElementType_> || is_epu16_v<_DesiredVectorElementType_>) {
    //
    //        }
    //        else if constexpr (is_epi8_v<_DesiredVectorElementType_> || is_epu8_v<_DesiredVectorElementType_>) {
    //
    //        }
    //    }
    //
    //    template <
    //        typename _MaskType_,
    //        typename _DesiredVectorElementType_,
    //        typename _VectorType_> 
    //    static simd_stl_always_inline _VectorType_ maskToVector(_MaskType_ mask) noexcept {
    //        if constexpr (is_epi64_v<_DesiredVectorElementType_> || is_epu64_v<_DesiredVectorElementType_> || is_pd_v<_DesiredVectorElementType_>) {
    //            _DesiredVectorElementType_ arrayTemp[4];
    //
    //            arrayTemp[0] = (mask & 1) ? 0xFFFFFFFFFFFFFFFF : 0;
    //            arrayTemp[1] = ((mask >> 1) & 1) ? 0xFFFFFFFFFFFFFFFF : 0;
    //            arrayTemp[2] = ((mask >> 2) & 1) ? 0xFFFFFFFFFFFFFFFF : 0;
    //            arrayTemp[3] = ((mask >> 3) & 1) ? 0xFFFFFFFFFFFFFFFF : 0;
    //            arrayTemp[4] = ((mask >> 4) & 1) ? 0xFFFFFFFFFFFFFFFF : 0;
    //            arrayTemp[5] = ((mask >> 5) & 1) ? 0xFFFFFFFFFFFFFFFF : 0;
    //            arrayTemp[6] = ((mask >> 6) & 1) ? 0xFFFFFFFFFFFFFFFF : 0;
    //            arrayTemp[7] = ((mask >> 7) & 1) ? 0xFFFFFFFFFFFFFFFF : 0;
    //
    //            return loadUnaligned<_VectorType_>(arrayTemp);
    //        }
    //        else if constexpr (is_epi32_v<_DesiredVectorElementType_> || is_epu32_v<_DesiredVectorElementType_> || is_ps_v<_DesiredVectorElementType_>) {
    //            const auto vshiftСount = _mm512_set_epi32(24, 25, 26, 27, 28, 29, 30, 31);
    //            auto bcast = _mm512_set1_epi32(mask);
    //            // Старший бит каждого элемента - соответствующий бит в маске
    //            auto shifted = _mm512_sllv_epi32(bcast, vshiftСount); // AVX2
    //            return shifted;
    //        }
    //        else if constexpr (is_epi16_v<_DesiredVectorElementType_> || is_epu16_v<_DesiredVectorElementType_>) {
    //            const auto shuffled = _mm512_setr_epi32(0, 0, 0x01010101, 0x01010101);
    //            auto v = shuffle<int8>(broadcast(mask), shuffle);
    //
    //            const auto bitselect = _mm512_setr_epi8(
    //                1, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6, 1U << 7,
    //                1, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6, 1U << 7);
    //
    //            v = _mm512_and_si512(v, bitselect);
    //            v = _mm512_min_epu8(v, _mm512_set1_epi8(1));
    //
    //            return v;
    //        }
    //        else if constexpr (is_epi8_v<_DesiredVectorElementType_> || is_epu8_v<_DesiredVectorElementType_>) {
    //            /* auto vmask = _mm512_set1_epi32(mask);
    //             const auto shuffle = _mm512_setr_epi64x(
    //                 0x0000000000000000, 0x0101010101010101,
    //                 0x0202020202020202, 0x0303030303030303);
    //
    //             vmask = _mm256_shuffle_epi8(vmask, shuffle);
    //             const auto bitMask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
    //
    //             vmask = _mm256_or_si256(vmask, bitMask);
    //             return _mm256_cmpeq_epi8(vmask, _mm256_set1_epi64x(-1));*/
    //        }
    //
    //        template <
    //            typename _DesiredType_,
    //            typename _VectorType_>
    //        static simd_stl_always_inline _VectorType_ loadUnaligned(const _DesiredType_ * where) noexcept {
    //            return _mm512_loadu_si512(reinterpret_cast<const __m512i*>(where));
    //        }
    //
    //        template <
    //            typename _DesiredType_,
    //            typename _VectorType_>
    //        static simd_stl_always_inline _VectorType_ loadAligned(const _DesiredType_ * where) noexcept {
    //            return _mm512_load_si512(reinterpret_cast<const __m512i*>(where));
    //        }
    //
    //        template <
    //            typename _DesiredType_,
    //            typename _VectorType_>
    //        simd_stl_always_inline void storeUnaligned(
    //            _DesiredType_ * where,
    //            const _VectorType_  vector) noexcept
    //        {
    //            return _mm512_storeu_si512(reinterpret_cast<__m512i*>(where), vector);
    //        }
    //
    //        template <
    //            typename _DesiredType_,
    //            typename _VectorType_>
    //        simd_stl_always_inline void storeAligned(
    //            _DesiredType_ * where,
    //            const _VectorType_  vector) noexcept
    //        {
    //            return _mm512_store_si512(reinterpret_cast<__m512i*>(where), vector);
    //        }
    //
    //        template <
    //            typename _DesiredType_,
    //            typename _VectorType_>
    //        simd_stl_always_inline void maskStoreUnaligned(
    //            _DesiredType_ * where,
    //            const uint64 /* ??? */  mask,
    //            const _VectorType_      vector) noexcept
    //        {
    //            if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>)
    //                return _mm512_mask_storeu_epi64(where, mask, cast<_VectorType_, __m512i>(vector));
    //            else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>)
    //                return _mm512_mask_storeu_epi32(where, mask, cast<_VectorType_, __m512i>(vector));
    //            else
    //                return _mm512_storeu_si512(where, cast<_VectorType_, __m512i>(shuffle<_DesiredType_>(vector, mask)));
    //        }
    //
    //        template <
    //            typename _DesiredType_,
    //            typename _VectorType_>
    //        simd_stl_always_inline void maskStoreAligned(
    //            _DesiredType_ * where,
    //            const uint64 /* ??? */  mask,
    //            const _VectorType_      vector) noexcept
    //        {
    //            if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>)
    //                return _mm512_mask_store_epi64(where, mask, cast<_VectorType_, __m512i>(vector));
    //            else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>)
    //                return _mm512_mask_store_epi32(where, mask, cast<_VectorType_, __m512i>(vector));
    //            else
    //                return _mm512_store_si512(where, cast<_VectorType_, __m512i>(shuffle<_DesiredType_>(vector, mask)));
    //        }
    //
    //        template <
    //            typename _DesiredType_,
    //            typename _VectorType_>
    //        simd_stl_always_inline _VectorType_ maskLoadUnaligned(
    //            const _DesiredType_ * where,
    //            const uint64            mask,
    //            const _VectorType_      vector) noexcept
    //        {
    //            if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>)
    //                return _mm512_mask_loadu_epi64(cast<_VectorType_, __m512i>(vector), mask, where);
    //            else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>)
    //                return _mm512_mask_loadu_epi32(cast<_VectorType_, __m512i>(vector), mask, where);
    //            else
    //                return cast<_VectorType_, __m512i>(shuffle<_DesiredType_>(_mm512_loadu_si512(where), mask));
    //        }
    //
    //        template <
    //            typename _DesiredType_,
    //            typename _VectorType_>
    //        simd_stl_always_inline void maskLoadAligned(
    //            const _DesiredType_ * where,
    //            const uint64            mask,
    //            const _VectorType_      vector) noexcept
    //        {
    //            if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>)
    //                return _mm512_mask_load_epi64(cast<_VectorType_, __m512i>(vector), mask, where);
    //            else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>)
    //                return _mm512_mask_load_epi32(cast<_VectorType_, __m512i>(vector), mask, where);
    //            else
    //                return cast<_VectorType_, __m512i>(shuffle<_DesiredType_>(_mm512_load_si512(where), mask));
    //        }
    //
    //
    template <
        typename    _FromVector_,
        typename    _ToVector_,
        bool        _SafeCast_ = false>
    static simd_stl_always_inline _ToVector_ cast(const _FromVector_ from) noexcept {
        if constexpr (std::is_same_v<_ToVector_, _FromVector_>)
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


        else if constexpr (std::is_same_v<_FromVector_, __m256> && std::is_same_v<_ToVector_, __m256i>)
            return _mm256_castps_si256(from);
        else if constexpr (std::is_same_v<_FromVector_, __m256> && std::is_same_v<_ToVector_, __m256d>)
            return _mm256_castps_pd(from);

        else if constexpr (std::is_same_v<_FromVector_, __m256d> && std::is_same_v<_ToVector_, __m256>)
            return _mm256_castpd_ps(from);
        else if constexpr (std::is_same_v<_FromVector_, __m256d> && std::is_same_v<_ToVector_, __m256i>)
            return _mm256_castpd_si256(from);

        else if constexpr (std::is_same_v<_FromVector_, __m256i> && std::is_same_v<_ToVector_, __m256>)
            return _mm256_castsi256_ps(from);
        else if constexpr (std::is_same_v<_FromVector_, __m256i> && std::is_same_v<_ToVector_, __m256d>)
            return _mm256_castsi256_pd(from);


        else if constexpr (std::is_same_v<_FromVector_, __m512> && std::is_same_v<_ToVector_, __m512i>)
            return _mm512_castps_si512(from);
        else if constexpr (std::is_same_v<_FromVector_, __m512> && std::is_same_v<_ToVector_, __m512d>)
            return _mm512_castps_pd(from);

        else if constexpr (std::is_same_v<_FromVector_, __m512d> && std::is_same_v<_ToVector_, __m512>)
            return _mm512_castpd_ps(from);
        else if constexpr (std::is_same_v<_FromVector_, __m512d> && std::is_same_v<_ToVector_, __m512i>)
            return _mm512_castpd_si512(from);

        else if constexpr (std::is_same_v<_FromVector_, __m512i> && std::is_same_v<_ToVector_, __m512>)
            return _mm512_castsi512_ps(from);
        else if constexpr (std::is_same_v<_FromVector_, __m512i> && std::is_same_v<_ToVector_, __m512d>)
            return _mm512_castsi512_pd(from);


        // Zero extend
        else if constexpr (std::is_same_v<_FromVector_, __m128> && std::is_same_v<_ToVector_, __m256> && _SafeCast_ == true)
            return _mm256_insertf128_ps(_mm256_castps128_ps256(from), _mm_setzero_ps(), 1);
        else if constexpr (std::is_same_v<_FromVector_, __m128d> && std::is_same_v<_ToVector_, __m256d> && _SafeCast_ == true)
            return _mm256_insertf128_pd(_mm256_castpd128_pd256(from), _mm_setzero_pd(), 1);
        else if constexpr (std::is_same_v<_FromVector_, __m128i> && std::is_same_v<_ToVector_, __m256i> && _SafeCast_ == true)
            return _mm256_insertf128_si256(_mm256_castsi128_si256(from), _mm_setzero_si128(), 1);

        // Zero extend
        else if constexpr (std::is_same_v<_FromVector_, __m128> && std::is_same_v<_ToVector_, __m256> && _SafeCast_ == false)
            return _mm256_castps128_ps256(from);
        else if constexpr (std::is_same_v<_FromVector_, __m128d> && std::is_same_v<_ToVector_, __m256d> && _SafeCast_ == false)
            return _mm256_castpd128_pd256(from);
        else if constexpr (std::is_same_v<_FromVector_, __m128i> && std::is_same_v<_ToVector_, __m256i> && _SafeCast_ == false)
            return _mm256_castsi128_si256(from);


        // Truncate
        else if constexpr (std::is_same_v<_FromVector_, __m256> && std::is_same_v<_ToVector_, __m128>)
            return _mm256_castps256_ps128(from);
        else if constexpr (std::is_same_v<_FromVector_, __m256d> && std::is_same_v<_ToVector_, __m128d>)
            return _mm256_castpd256_pd128(from);
        else if constexpr (std::is_same_v<_FromVector_, __m256i> && std::is_same_v<_ToVector_, __m128i>)
            return _mm256_castsi256_si128(from);

        // Zero extend
        else if constexpr (std::is_same_v<_FromVector_, __m128> && std::is_same_v<_ToVector_, __m512> && _SafeCast_ == true)
            return _mm512_insertf128_ps(_mm512_castps128_ps512(from), _mm_setzero_ps(), 1);
        else if constexpr (std::is_same_v<_FromVector_, __m128d> && std::is_same_v<_ToVector_, __m512d> && _SafeCast_ == true)
            return _mm512_insertf128_pd(_mm512_castpd128_pd512(from), _mm_setzero_pd(), 1);
        else if constexpr (std::is_same_v<_FromVector_, __m128i> && std::is_same_v<_ToVector_, __m512i> && _SafeCast_ == true)
            return _mm512_insertf128_si512(_mm512_castsi128_si512(from), _mm_setzero_si128(), 1);


        // Undefined
        else if constexpr (std::is_same_v<_FromVector_, __m128> && std::is_same_v<_ToVector_, __m512> && _SafeCast_ == false)
            return _mm512_castps128_ps512(from);
        else if constexpr (std::is_same_v<_FromVector_, __m128d> && std::is_same_v<_ToVector_, __m512d> && _SafeCast_ == false)
            return _mm512_castpd128_pd512(from);
        else if constexpr (std::is_same_v<_FromVector_, __m128i> && std::is_same_v<_ToVector_, __m512i> && _SafeCast_ == false)
            return _mm512_castsi128_si512(from);


        // Truncate
        else if constexpr (std::is_same_v<_FromVector_, __m512> && std::is_same_v<_ToVector_, __m128>)
            return _mm512_castps512_ps128(from);
        else if constexpr (std::is_same_v<_FromVector_, __m512d> && std::is_same_v<_ToVector_, __m128d>)
            return _mm512_castpd512_pd128(from);
        else if constexpr (std::is_same_v<_FromVector_, __m512i> && std::is_same_v<_ToVector_, __m128i>)
            return _mm512_castsi512_si128(from);

        // Zero extend
        else if constexpr (std::is_same_v<_FromVector_, __m256> && std::is_same_v<_ToVector_, __m512> && _SafeCast_ == true)
            return _mm512_insertf256_ps(_mm512_castps256_ps512(from), _mm256_setzero_ps(), 1);
        else if constexpr (std::is_same_v<_FromVector_, __m256d> && std::is_same_v<_ToVector_, __m512d> && _SafeCast_ == true)
            return _mm512_insertf256_pd(_mm512_castpd256_pd512(from), _mm256_setzero_pd(), 1);
        else if constexpr (std::is_same_v<_FromVector_, __m256i> && std::is_same_v<_ToVector_, __m512i> && _SafeCast_ == true)
            return _mm512_insertf256_si512(_mm512_castsi256_si512(from), _mm256_setzero_si256(), 1);

        // Undefined
        else if constexpr (std::is_same_v<_FromVector_, __m256> && std::is_same_v<_ToVector_, __m512> && _SafeCast_ == false)
            return _mm512_castps256_ps512(from);
        else if constexpr (std::is_same_v<_FromVector_, __m256d> && std::is_same_v<_ToVector_, __m512d> && _SafeCast_ == false)
            return _mm512_castpd256_pd512(from);
        else if constexpr (std::is_same_v<_FromVector_, __m256i> && std::is_same_v<_ToVector_, __m512i> && _SafeCast_ == false)
            return _mm512_castsi256_si512(from);

        // Truncate
        else if constexpr (std::is_same_v<_FromVector_, __m512> && std::is_same_v<_ToVector_, __m256>)
            return _mm512_castps512_ps256(from);
        else if constexpr (std::is_same_v<_FromVector_, __m512d> && std::is_same_v<_ToVector_, __m256d>)
            return _mm512_castpd512_pd256(from);
        else if constexpr (std::is_same_v<_FromVector_, __m512i> && std::is_same_v<_ToVector_, __m256i>)
            return _mm512_castsi512_si256(from);
    }

    //        template <
    //            typename _DesiredType_,
    //            typename _VectorType_>
    //        static simd_stl_always_inline _VectorType_ decrement(_VectorType_ vector) noexcept {
    //            return sub(vector, broadcast(1));
    //        }
    //
    //        template <
    //            typename _DesiredType_,
    //            typename _VectorType_>
    //        static simd_stl_always_inline _VectorType_ increment(_VectorType_ vector) noexcept {
    //            return add(vector, broadcast(1));
    //        }
    //
    //        template <
    //            typename _DesiredType_,
    //            typename _VectorType_>
    //        static simd_stl_always_inline _DesiredType_ extract(
    //            _VectorType_    vector,
    //            uint8           where) noexcept
    //        {
    //            if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_> || is_pd_v<_DesiredType_>) {
    //                _DesiredType_ x[4];
    //                storeUnaligned(x);
    //                return x[where & 3];
    //            }
    //            else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_> || is_ps_v<_DesiredType_>) {
    //                _DesiredType_ x[8];
    //                storeUnaligned(x);
    //                return x[where & 7];
    //            }
    //            else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>) {
    //                _DesiredType_ x[16];
    //                storeUnaligned(x);
    //                return x[where & 0x0F];
    //            }
    //            else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>) {
    //                _DesiredType_ x[32];
    //                storeUnaligned(x);
    //                return x[where & 0x1F];
    //            }
    //        }
    //
    //        template <typename _VectorType_>
    //        static simd_stl_always_inline _VectorType_ constructZero() noexcept {
    //            if      constexpr (std::is_same_v<_VectorType_, __m512d>)
    //                return _mm512_setzero_pd();
    //            else if constexpr (std::is_same_v<_VectorType_, __m512i>)
    //                return _mm512_setzero_si512();
    //            else if constexpr (std::is_same_v<_VectorType_, __m512>)
    //                return _mm512_setzero_ps();
    //        }
    //
    //        template <
    //            typename _DesiredType_,
    //            typename _VectorType_>
    //        static simd_stl_always_inline _VectorType_ broadcast(_DesiredType_ value) noexcept {
    //
    //        }
    //
    //        template <
    //            typename _DesiredType_,
    //            typename _VectorType_>
    //        static simd_stl_always_inline _VectorType_ add(
    //            _VectorType_ left,
    //            _VectorType_ right) noexcept
    //        {
    //
    //        }
    //
    //
    //        template <
    //            typename _DesiredType_,
    //            typename _VectorType_>
    //        static simd_stl_always_inline _VectorType_ sub(
    //            _VectorType_ left,
    //            _VectorType_ right) noexcept
    //        {
    //
    //        }
    //
    //        template <
    //            typename _DesiredType_,
    //            typename _VectorType_>
    //        static simd_stl_always_inline _VectorType_ mul(
    //            _VectorType_ left,
    //            _VectorType_ right) noexcept
    //        {
    //
    //        }
    //
    //        template <
    //            typename _DesiredType_,
    //            typename _VectorType_>
    //        static simd_stl_always_inline _VectorType_ div(
    //            _VectorType_ left,
    //            _VectorType_ right) noexcept
    //        {
    //            if      constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>)
    //                return cast<__m512d, _VectorType_>(_mm512_div_pd(
    //                    cast<_VectorType_, __m512d>(left),
    //                    cast<_VectorType_, __m512d>(right)));
    //            else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_> || is_ps_v<_DesiredType_>)
    //                return cast<__m512, _VectorType_>(_mm512_div_ps(
    //                    cast<_VectorType_, __m512>(left),
    //                    cast<_VectorType_, __m512>(right)));
    //            else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>)
    //                return cast<__m512, _VectorType_>(div_u16(
    //                    cast<_VectorType_, __m512i>(left),
    //                    cast<_VectorType_, __m512i>(right)));
    //            else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>)
    //                return cast<__m512i, _VectorType_>(div_u8(
    //                    cast<_VectorType_, __m512i>(left),
    //                    cast<_VectorType_, __m512i>(right)));
    //        }
    //
    //        template <typename _VectorType_>
    //        static simd_stl_always_inline _VectorType_ bitwiseNot(_VectorType_ vector) noexcept {
    //            return cast<__m512i, _VectorType_>(
    //                bitwiseXor(vector, _mm512_cmpeq_epi32(
    //                    cast<_VectorType_, __m512i>(vector))));
    //        }
    //
    //        template <typename _VectorType_>
    //        static simd_stl_always_inline _VectorType_ bitwiseXor(
    //            _VectorType_ left,
    //            _VectorType_ right) noexcept
    //        {
    //            return cast<__m512i, _VectorType_>(_mm512_xor_si512(
    //                cast<_VectorType_, __m512i>(left),
    //                cast<_VectorType_, __m512i>(right)));
    //        }
    //
    //        template <typename _VectorType_>
    //        static simd_stl_always_inline _VectorType_ bitwiseAnd(
    //            _VectorType_ left,
    //            _VectorType_ right) noexcept
    //        {
    //            return cast<__m512i, _VectorType_>(_mm512_and_si512(
    //                cast<_VectorType_, __m512i>(left),
    //                cast<_VectorType_, __m512i>(right)));
    //        }
    //
    //        template <typename _VectorType_>
    //        static simd_stl_always_inline _VectorType_ bitwiseOr(
    //            _VectorType_ left,
    //            _VectorType_ right) noexcept
    //        {
    //            return cast<__m512i, _VectorType_>(_mm512_or_si512(
    //                cast<_VectorType_, __m512i>(left),
    //                cast<_VectorType_, __m512i>(right)));
    //        }
    //private:
    //    static simd_stl_always_inline __m512i divLow_u8_i32x16(
    //        __m512i left,
    //        __m512i right,
    //        float   mul) noexcept
    //    {
    //        const auto af = _mm512_cvtepi32_ps(left);
    //        const auto bf = _mm512_cvtepi32_ps(right);
    //
    //        const auto m1 = _mm512_mul_ps(af, _mm512_set1_ps(1.001f * mul));
    //        const auto m2 = _mm512_rcp14_ps(bf);
    //
    //        return _mm512_cvttps_epi32(_mm512_mul_ps(m1, m2));
    //    }
    //
    //    static simd_stl_always_inline __m512i div_u8(
    //        __m512i left,
    //        __m512i right) noexcept
    //    {
    //        const auto m0 = _mm512_set1_epi32(0x000000ff);
    //        const auto m1 = _mm512_set1_epi32(0x0000ff00);
    //        const auto m2 = _mm512_set1_epi32(0x00ff0000);
    //
    //        const auto r0 = divLow_u8_i32x16(_mm512_and_si512(left, m0), _mm512_and_si512(right, m0), 1);
    //        auto r1 = divLow_u8_i32x16(_mm512_and_si512(left, m1), _mm512_and_si512(right, m1), 1);
    //        r1 = _mm512_slli_epi32(r1, 8);
    //
    //        const auto r2 = divLow_u8_i32x16(_mm512_and_si512(left, m2), _mm512_and_si512(right, m2), 1 << 16);
    //        auto r3 = divLow_u8_i32x16(_mm512_srli_epi32(left, 24), _mm512_srli_epi32(right, 24), 1);
    //
    //        r3 = _mm512_slli_epi32(r3, 24);
    //
    //        auto r01 = _mm512_or_si512(r0, r1);
    //        auto r23 = _mm512_or_si512(r2, r3);
    //
    //        return shuffle<int16>(r01, r23, maskToVector(0xAA));
    //    }
    //
    //    static simd_stl_always_inline __m512i div_u16(
    //        const __m512i left,
    //        const __m512i right) noexcept
    //    {
    //        const auto mask_lo = _mm512_set1_epi32(0x0000ffff);
    //
    //        const auto a_lo_u32 = _mm512_and_si512(left, mask_lo);
    //        const auto b_lo_u32 = _mm512_and_si512(right, mask_lo);
    //
    //        const auto a_hi_u32 = _mm512_srli_epi32(left, 16);
    //        const auto b_hi_u32 = _mm512_srli_epi32(right, 16);
    //
    //        const auto  a_lo_f32 = _mm512_cvtepi32_ps(a_lo_u32);
    //        const auto  a_hi_f32 = _mm512_cvtepi32_ps(a_hi_u32);
    //        const auto  b_lo_f32 = _mm512_cvtepi32_ps(b_lo_u32);
    //        const auto  b_hi_f32 = _mm512_cvtepi32_ps(b_hi_u32);
    //
    //        const auto  c_lo_f32 = _mm512_div_ps(a_lo_f32, b_lo_f32);
    //        const auto  c_hi_f32 = _mm512_div_ps(a_hi_f32, b_hi_f32);
    //
    //        const auto c_lo_i32 = _mm512_cvttps_epi32(c_lo_f32); // values in the u16 range
    //        const auto c_hi_i32_0 = _mm512_cvttps_epi32(c_hi_f32); // values in the u16 range
    //        const auto c_hi_i32 = _mm512_slli_epi32(c_hi_i32_0, 16);
    //
    //        return _mm512_or_si512(c_lo_i32, c_hi_i32);
    //    }
};

template <>
class BasicSimdImplementation<arch::CpuFeature::AVX512BW> :
    public BasicSimdImplementation<arch::CpuFeature::AVX512F>
{

};


__SIMD_STL_NUMERIC_NAMESPACE_END
