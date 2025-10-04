#pragma once 

#include <src/simd_stl/numeric/BasicSimdImplementationUnspecialized.h>

#include <src/simd_stl/type_traits/TypeTraits.h>
#include <simd_stl/compatibility/Inline.h>

#include <simd_stl/numeric/BasicSimdMask.h>
#include <simd_stl/numeric/BasicSimdShuffleMask.h>

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <>
class BasicSimdImplementation<arch::CpuFeature::AVX2> {
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
            return _mm256_shuffle_pd(
                cast<_VectorType_, __m256d>(vector),
                cast<_VectorType_, __m256d>(vectorSecond),
                shuffleMask
            );
        else if constexpr (is_ps_v<_DesiredType_>)
            return _mm256_shuffle_ps(
                cast<_VectorType_, __m256>(vector),
                cast<_VectorType_, __m256>(vectorSecond),
                shuffleMask
            );
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>)
            return _mm256_shuffle_epi32(
                cast<_VectorType_, __m256i>(vector),
                cast<_VectorType_, __m256i>(vectorSecond),
                shuffleMask
            );
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>) {
            const auto shuffledHigh = _mm256_shufflehi_epi16(vector, shuffleMask);
            const auto shuffledLow  = _mm256_shufflelo_epi16(vectorSecond, shuffleMask);

            return _mm256_set_epi64x(
                _mm_cvtsd_f64(cast<_VectorType_, __m128d>(shuffledLow)),
                _mm_cvtsd_f64(cast<_VectorType_, __m128d>(shuffledHigh)),
                extract<int64>(shuffledHigh, 2),
                extract<int64>(shuffledLow, 2)
            );
        }
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>) {
            return _mm256_shuffle_epi8(
                cast<_VectorType_, __m256i>(vector),
                cast<_VectorType_, __m256i>(vectorSecond),
                shuffleMask
            );
        }
    }


    template <
        typename _DesiredVectorElementType_,
        typename _VectorType_>
    static simd_stl_always_inline int32 maskFromVector(_VectorType_ vector) noexcept {
        if constexpr (is_pd_v<_DesiredVectorElementType_> || is_epi64_v<_DesiredVectorElementType_> || is_epu64_v<_DesiredVectorElementType_>)
            return _mm256_movemask_pd(cast<_VectorType_, __m256d>(vector));
        else if constexpr (is_ps_v<_DesiredVectorElementType_> || is_epi32_v<_DesiredVectorElementType_> || is_epu32_v<_DesiredVectorElementType_>)
            return _mm256_movemask_ps(cast<_VectorType_, __m256>(vector));
        else if constexpr (is_epi16_v<_DesiredVectorElementType_> || is_epu16_v<_DesiredVectorElementType_>) {

        }
        else if constexpr (is_epi8_v<_DesiredVectorElementType_> || is_epu8_v<_DesiredVectorElementType_>) {
            return _mm256_movemask_epi8(cast<_VectorType_, __m256i>(vector));
        }
    }

    template <
        typename _MaskType_,
        typename _DesiredVectorElementType_,
        typename _VectorType_> 
    static simd_stl_always_inline _VectorType_ maskToVector(_MaskType_ mask) noexcept {
        if constexpr (is_epi64_v<_DesiredVectorElementType_> || is_epu64_v<_DesiredVectorElementType_> || is_pd_v<_DesiredVectorElementType_>) {
            _DesiredVectorElementType_ arrayTemp[4];

            arrayTemp[0] = (mask        & 1) ? 0xFFFFFFFFFFFFFFFF : 0;
            arrayTemp[1] = ((mask >> 1) & 1) ? 0xFFFFFFFFFFFFFFFF : 0;
            arrayTemp[2] = ((mask >> 2) & 1) ? 0xFFFFFFFFFFFFFFFF : 0;
            arrayTemp[3] = ((mask >> 3) & 1) ? 0xFFFFFFFFFFFFFFFF : 0;

            return loadUnaligned<_VectorType_>(arrayTemp);
        }
        else if constexpr (is_epi32_v<_DesiredVectorElementType_> || is_epu32_v<_DesiredVectorElementType_> || is_ps_v<_DesiredVectorElementType_>) {
            const auto vshiftСount = _mm256_set_epi32(24, 25, 26, 27, 28, 29, 30, 31);
            auto bcast = _mm256_set1_epi32(mask);
            // Старший бит каждого элемента - соответствующий бит в маске
            auto shifted = _mm256_sllv_epi32(bcast, vshiftСount); // AVX2
            return shifted;
        }
        else if constexpr (is_epi16_v<_DesiredVectorElementType_> || is_epu16_v<_DesiredVectorElementType_>) {
            /*const auto shuffle = _mm256_setr_epi32(0, 0, 0x01010101, 0x01010101, 0, 0, 0x01010101, 0x01010101);
            auto v = _mm256_shuffle_epi8(broadcast(mask), shuffle);

            const auto bitselect = _mm256_setr_epi8(
                1, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6, 1U << 7, 1U << 8, 1U << 9, 1U << 10, 1U << 11, 1U << 12, 1U << 13, 1U << 14, 1U << 15
                1, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6, 1U << 7, 1U << 8, 1U << 9, 1U << 10, 1U << 11, 1U << 12, 1U << 13, 1U << 14, 1U << 15);

            v = _mm256_and_si256(v, bitselect);
            v = _mm256_min_epu8(v, _mm256_set1_epi8(1));

            return v;*/
        }
        else if constexpr (is_epi8_v<_DesiredVectorElementType_> || is_epu8_v<_DesiredVectorElementType_>) {
            auto vmask = _mm256_set1_epi32(mask);
            const auto shuffle = _mm256_setr_epi64x(
                0x0000000000000000, 0x0101010101010101,
                0x0202020202020202, 0x0303030303030303);

            vmask = _mm256_shuffle_epi8(vmask, shuffle);
            const auto bitMask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);

            vmask = _mm256_or_si256(vmask, bitMask);
            return _mm256_cmpeq_epi8(vmask, _mm256_set1_epi64x(-1));
        }
    }
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ loadUnaligned(const _DesiredType_* where) noexcept {
        return _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(where));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ loadAligned(const _DesiredType_* where) noexcept {
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(where));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void storeUnaligned(
        _DesiredType_*      where,
        const _VectorType_  vector) noexcept
    {
        return _mm256_storeu_si256(reinterpret_cast<__m256i*>(where), vector);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void storeAligned(
        _DesiredType_*      where,
        const _VectorType_  vector) noexcept
    {
        return _mm256_store_si256(reinterpret_cast<__m256i*>(where), vector);
    }
    
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void maskStoreUnaligned(
        _DesiredType_*                                      where,
        const type_traits::__deduce_simd_shuffle_mask_type<
            sizeof(_VectorType_) / sizeof(_DesiredType_)>   mask,
        const _VectorType_                                  vector) noexcept
    {
        _mm256_maskstore_ps(
            reinterpret_cast<float*>(where), 
            maskToVector(mask), cast<_VectorType_, __m256>(vector));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void maskStoreAligned(
        _DesiredType_*                                          where,
        const type_traits::__deduce_simd_shuffle_mask_type<
            sizeof(_VectorType_) / sizeof(_DesiredType_)>       mask,
        const _VectorType_                                      vector) noexcept
    {
        _mm256_maskstore_ps(
            reinterpret_cast<float*>(where),
            maskToVector(mask), cast<_VectorType_, __m256>(vector));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ maskLoadUnaligned(
        const _DesiredType_*                                where,
        const type_traits::__deduce_simd_shuffle_mask_type<
            sizeof(_VectorType_) / sizeof(_DesiredType_)>   mask) noexcept
    {
        return cast<__m256, _VectorType_>(
            _mm256_maskload_ps(
                reinterpret_cast<const float*>(where),
                maskToVector(mask)));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ maskLoadAligned(
        const _DesiredType_*                                where,
        const type_traits::__deduce_simd_shuffle_mask_type<
            sizeof(_VectorType_) / sizeof(_DesiredType_)>   mask) noexcept
    {
        return cast<__m256, _VectorType_>(
            _mm256_maskload_ps(
                reinterpret_cast<const float*>(where),
                maskToVector(mask)));
    }

    template <
        typename    _FromVector_,
        typename    _ToVector_,
        bool        _SafeCast_ = false>
    static simd_stl_always_inline _ToVector_ cast(_FromVector_ from) noexcept {
        if constexpr (std::is_same_v<_ToVector_, _FromVector_>)
            return from;


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

        // Zero extend
        else if constexpr (std::is_same_v<_FromVector_, __m128> && std::is_same_v<_ToVector_, __m256>   && _SafeCast_ == true)
            return _mm256_insertf128_ps(_mm256_castps128_ps256(from), _mm_setzero_ps(), 1);
        else if constexpr (std::is_same_v<_FromVector_, __m128d> && std::is_same_v<_ToVector_, __m256d> && _SafeCast_ == true)
            return _mm256_insertf128_pd(_mm256_castpd128_pd256(from), _mm_setzero_pd(), 1);
        else if constexpr (std::is_same_v<_FromVector_, __m128i> && std::is_same_v<_ToVector_, __m256i> && _SafeCast_ == true)
            return _mm256_insertf128_si256(_mm256_castsi128_si256(from), _mm_setzero_si128(), 1);

        // Undefined
        else if constexpr (std::is_same_v<_FromVector_, __m128> && std::is_same_v<_ToVector_, __m256>   && _SafeCast_ == false)
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
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ decrement(_VectorType_ vector) noexcept {
        return sub(vector, broadcast(1));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ increment(_VectorType_ vector) noexcept {
        return add(vector, broadcast(1));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_ extract(
        _VectorType_    vector,
        uint8           where) noexcept
    {
        if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_> || is_pd_v<_DesiredType_>) {
            _DesiredType_ x[4];
            storeUnaligned(x);
            return x[where & 3];
        }
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_> || is_ps_v<_DesiredType_>) {
            _DesiredType_ x[8];
            storeUnaligned(x);
            return x[where & 7];
        }
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>) {
            _DesiredType_ x[16];
            storeUnaligned(x);
            return x[where & 0x0F];
        }
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>) {
            _DesiredType_ x[32];
            storeUnaligned(x);
            return x[where & 0x1F];
        }
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ constructZero() noexcept {
        if      constexpr (std::is_same_v<_VectorType_, __m256d>)
            return _mm256_setzero_pd();
        else if constexpr (std::is_same_v<_VectorType_, __m256i>)
            return _mm256_setzero_si256();
        else if constexpr (std::is_same_v<_VectorType_, __m256>)
            return _mm256_setzero_ps();
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ broadcast(_DesiredType_ value) noexcept {
        if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>)
            return cast<__m256i, _VectorType_>(_mm256_set1_epi64x(value));
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>)
            return cast<__m256i, _VectorType_>(_mm256_set1_epi32(value));
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>)
            return cast<__m256i, _VectorType_>(_mm256_set1_epi16(value));
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>)
            return cast<__m256i, _VectorType_>(_mm256_set1_epi8(value));
        else if constexpr (is_ps_v<_DesiredType_>)
            return cast<__m256, _VectorType_>(_mm256_set1_ps(value));
        else if constexpr (is_pd_v<_DesiredType_>)
            return cast<__m256d, _VectorType_>(_mm256_set1_pd(value));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ add(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>)
            return _mm256_add_epi64(left, right);
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>)
            return _mm256_add_epi32(left, right);
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>)
            return _mm256_add_epi16(left, right);
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>)
            return _mm256_add_epi8(left, right);
        else if constexpr (is_ps_v<_DesiredType_>)
            return _mm256_add_ps(left, right);
        else if constexpr (is_pd_v<_DesiredType_>)
            return _mm256_add_pd(left, right);
    }
    

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ sub(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>)
            return _mm256_sub_epi64(left, right);
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>)
            return _mm256_sub_epi32(left, right);
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>)
            return _mm256_sub_epi16(left, right);
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>)
            return _mm256_sub_epi8(left, right);
        else if constexpr (is_ps_v<_DesiredType_>)
            return _mm256_sub_ps(left, right);
        else if constexpr (is_pd_v<_DesiredType_>)
            return _mm256_sub_pd(left, right);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ mul(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>) {
            auto ymm2 = _mm256_mul_epu32(_mm256_srli_epi64(right, 32), left);
            auto ymm3 = _mm256_mul_epu32(_mm256_srli_epi64(left, 32), right);

            ymm2 = _mm256_slli_epi64(_mm256_add_epi64(ymm3, ymm2), 32);
            return _mm256_add_epi64(_mm256_mul_epu32(right, left), ymm2);
        }
        else if constexpr (is_epi32_v<_DesiredType_>)
            return _mm256_mul_epi32(left, right);
        else if constexpr (is_epu32_v<_DesiredType_>)
            return _mm256_mul_epu32(left, right);
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>)
            return _mm256_mullo_epi16(left, right);
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>) {
            auto ymm3 = _mm256_unpacklo_epi8(right, right);
            auto ymm2 = _mm256_unpacklo_epi8(left, left);

            left = _mm256_unpackhi_epi8(left, left);
            right = _mm256_unpackhi_epi8(right, right);

            ymm2 = _mm256_mullo_epi16(ymm2, ymm3);
            left = _mm256_mullo_epi16(left, right);

            ymm2 = _mm256_shuffle_epi8(ymm2, ymm2);
            left = _mm256_shuffle_epi8(left, left);

            return _mm256_blend_epi32(left, ymm2, 0x33);
        }
        else if constexpr (is_ps_v<_DesiredType_>)
            return _mm256_mul_ps(left, right);
        else if constexpr (is_pd_v<_DesiredType_>)
            return _mm256_mul_pd(left, right);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ div(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        if      constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>)
            return cast<__m256d, _VectorType_>(_mm256_div_pd(left, right));
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_> || is_ps_v<_DesiredType_>)
            return cast<__m256, _VectorType_>(_mm256_div_ps(left, right));
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>)
            return cast<__m256i, _VectorType_>(div_u16(left, right));
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>)
            return cast<__m256i, _VectorType_>(div_u8(left, right));
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ bitwiseNot(_VectorType_ vector) noexcept {
        if      constexpr (std::is_same_v<_VectorType_, __m256d>)
            return _mm256_xor_pd(vector, _mm256_cmp_pd(vector, vector, _CMP_EQ_OQ));
        else if constexpr (std::is_same_v<_VectorType_, __m256>)
            return _mm256_xor_ps(vector, _mm256_cmp_ps(vector, vector, _CMP_EQ_OQ));
        else if constexpr (std::is_same_v<_VectorType_, __m256i>)
            return _mm256_xor_si256(vector, _mm256_cmpeq_epi64(vector, vector));
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ bitwiseXor(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m256d>)
            return _mm256_xor_pd(left, right);
        else if constexpr (std::is_same_v<_VectorType_, __m256>)
            return _mm256_xor_ps(left, right);
        else if constexpr (std::is_same_v<_VectorType_, __m256i>)
            return _mm256_xor_si256(left, right);
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ bitwiseAnd(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m256d>)
            return _mm256_and_pd(left, right);
        else if constexpr (std::is_same_v<_VectorType_, __m256>)
            return _mm256_and_ps(left, right);
        else if constexpr (std::is_same_v<_VectorType_, __m256i>)
            return _mm256_and_si256(left, right);
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ bitwiseOr(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m256d>)
            return _mm256_or_pd(left, right);
        else if constexpr (std::is_same_v<_VectorType_, __m256>)
            return _mm256_or_ps(left, right);
        else if constexpr (std::is_same_v<_VectorType_, __m256>)
            return _mm256_or_si256(left, right);
    }

private:
    static simd_stl_always_inline __m256i divLow_u8_i32x8(
        __m256i left,
        __m256i right, 
        float mul) noexcept 
    {
        const auto af = _mm256_cvtepi32_ps(left);
        const auto bf = _mm256_cvtepi32_ps(right);

        const auto m1 = _mm256_mul_ps(af, _mm256_set1_ps(1.001f * mul));
        const auto m2 = _mm256_rcp_ps(bf);

        return _mm256_cvttps_epi32(_mm256_mul_ps(m1, m2));
    }

    static simd_stl_always_inline __m256i div_u8(
        __m256i left,
        __m256i right) noexcept 
    {
        const auto m0 = _mm256_set1_epi32(0x000000ff);
        const auto m1 = _mm256_set1_epi32(0x0000ff00);
        const auto m2 = _mm256_set1_epi32(0x00ff0000);

        const auto r0 = divLow_u8_i32x8(_mm256_and_si256(left, m0), _mm256_and_si256(right, m0), 1);
        auto r1 = divLow_u8_i32x8(_mm256_and_si256(left, m1), _mm256_and_si256(right, m1), 1);
        r1 = _mm256_slli_epi32(r1, 8);

        const auto r2 = divLow_u8_i32x8(_mm256_and_si256(left, m2), _mm256_and_si256(right, m2), 1 << 16);
        auto r3 = divLow_u8_i32x8(_mm256_srli_epi32(left, 24), _mm256_srli_epi32(right, 24), 1);

        r3 = _mm256_slli_epi32(r3, 24);

        const auto r01 = _mm256_or_si256(r0, r1);
        const auto r23 = _mm256_or_si256(r2, r3);

        return _mm256_blend_epi16(r01, r23, 0xAA);
    }

    static simd_stl_always_inline __m256i div_u16(
        const __m256i left, 
        const __m256i right) noexcept
    {
        const auto mask_lo = _mm256_set1_epi32(0x0000ffff);

        const auto a_lo_u32 = _mm256_and_si256(left, mask_lo);
        const auto b_lo_u32 = _mm256_and_si256(right, mask_lo);

        const auto a_hi_u32 = _mm256_srli_epi32(left, 16);
        const auto b_hi_u32 = _mm256_srli_epi32(right, 16);

        const auto  a_lo_f32 = _mm256_cvtepi32_ps(a_lo_u32);
        const auto  a_hi_f32 = _mm256_cvtepi32_ps(a_hi_u32);
        const auto  b_lo_f32 = _mm256_cvtepi32_ps(b_lo_u32);
        const auto  b_hi_f32 = _mm256_cvtepi32_ps(b_hi_u32);

        const auto  c_lo_f32 = _mm256_div_ps(a_lo_f32, b_lo_f32);
        const auto  c_hi_f32 = _mm256_div_ps(a_hi_f32, b_hi_f32);

        const auto c_lo_i32 = _mm256_cvttps_epi32(c_lo_f32); // values in the u16 range
        const auto c_hi_i32_0 = _mm256_cvttps_epi32(c_hi_f32); // values in the u16 range
        const auto c_hi_i32 = _mm256_slli_epi32(c_hi_i32_0, 16);

        return _mm256_or_si256(c_lo_i32, c_hi_i32);
    }
};


__SIMD_STL_NUMERIC_NAMESPACE_END
