#pragma once 

#include <src/simd_stl/numeric/BasicSimdImplementationUnspecialized.h>
#include <src/simd_stl/numeric/xmm/SimdDivisors.h>

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <>
class BasicSimdImplementation<arch::CpuFeature::SSE2> {
public:
    template <
        typename _DesiredType_,
        class    _CompareType_,
        typename _VectorType_>
    static simd_stl_always_inline auto compare(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
       /* if constexpr (std::is_same_v<_CompareType, type_traits::equal_to>) {

        }*/
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void insert(
        _VectorType_& vector,
        const uint8         position,
        const _DesiredType_ value) noexcept
    {
        if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>) {
#if defined(simd_stl_processor_x86_64)
            auto vectorValue = _mm_cvtsi64_si128(value);
#else
            union {
                __m128i vec;
                int64   num;
            } convert;

            convert.num = value;
            auto vectorValue = _mm_loadl_epi64(&convert.vec);
#endif
            if (position == 0) {
                vectorValue = _mm_unpacklo_epi64(vectorValue, vectorValue);
                vector = _mm_unpackhi_epi64(vectorValue, vector);
            }
            else
                vector = _mm_unpacklo_epi64(vector, vectorValue);
        }
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>) {
            const auto broad = _mm_set1_epi32(value);
            const int32 maskl[8] = { 0,0,0,0,-1,0,0,0 };

            const auto mask = _mm_loadu_si128((__m128i const*)(maskl + 4 - (position & 3))); // FFFFFFFF at index position
            vector = _mm_or_si128(_mm_and_si128(mask, broad), _mm_andnot_si128(mask, vector));
        }
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>) {
            // Обход ошибки C2057 MSVC
            switch (position) {
            case 0:
                vector = _mm_insert_epi16(vector, value, 0);
            case 1:
                vector = _mm_insert_epi16(vector, value, 1);
            case 2:
                vector = _mm_insert_epi16(vector, value, 2);
            case 3:
                vector = _mm_insert_epi16(vector, value, 3);
            case 4:
                vector = _mm_insert_epi16(vector, value, 4);
            case 5:
                vector = _mm_insert_epi16(vector, value, 5);
            case 6:
                vector = _mm_insert_epi16(vector, value, 6);
            case 7:
                vector = _mm_insert_epi16(vector, value, 7);
            }

        }
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>) {
            const int8 maskl[32] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
            const auto broad = _mm_set1_epi8(value);

            const auto mask = _mm_loadu_si128((__m128i const*)(maskl + 16 - (position & 0x0F))); // FF at index position
            vector = _mm_or_si128(_mm_and_si128(mask, broad), _mm_andnot_si128(mask, vector));
        }
        else if constexpr (is_pd_v<_DesiredType_>) {
            const auto broadcasted = _mm_set_sd(value);

            vector = (position == 0)
                ? _mm_shuffle_pd(broadcasted, vector, 2)
                : _mm_shuffle_pd(vector, broadcasted, 0);
        }
        else if constexpr (is_ps_v<_DesiredType_>) {
            const int32 maskl[8] = { 0,0,0,0,-1,0,0,0 };

            const auto broadcasted = _mm_set1_ps(value);
            const auto mask = _mm_loadu_ps((float const*)(maskl + 4 - (position & 3))); // FFFFFFFF at index position

            vector = _mm_or_ps(
                _mm_and_ps(mask, broadcasted),
                _mm_andnot_ps(mask, vector));
        }
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _DesiredType_ extract(
        _VectorType_    vector,
        const uint8     where) noexcept
    {
        if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>) {
            // Обход ошибки C2057 MSVC
            switch (where) {
            case 0:
                return _mm_extract_epi16(vector, 0);
            case 1:
                return _mm_extract_epi16(vector, 1);
            case 2:
                return _mm_extract_epi16(vector, 2);
            case 3:
                return _mm_extract_epi16(vector, 3);
            case 4:
                return _mm_extract_epi16(vector, 4);
            case 5:
                return _mm_extract_epi16(vector, 5);
            case 6:
                return _mm_extract_epi16(vector, 6);
            case 7:
                return _mm_extract_epi16(vector, 7);
            }
        }
        else {
            constexpr auto vectorLength = (sizeof(_VectorType_) / sizeof(_DesiredType_));
            _DesiredType_ array[vectorLength];

            storeUnaligned(array, vector);
            return array[where & (vectorLength - 1)];
        }
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ shiftRight(
        _VectorType_    vector,
        uint32          shift) noexcept
    {
        if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_srli_epi64(cast<_VectorType_, __m128i>(vector), shift));
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_srli_epi32(cast<_VectorType_, __m128i>(vector), shift));
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_srli_epi16(cast<_VectorType_, __m128i>(vector), shift));
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>) {
            auto evenVector = _mm_slli_epi16(vector, 8);
            evenVector = _mm_sra_epi16(evenVector, _mm_cvtsi32_si128(shift + 8));

            const auto oddVector = _mm_sra_epi16(vector, _mm_cvtsi32_si128(shift));
            const auto mask = _mm_set1_epi32(0x00FF00FF);

            return _mm_or_si128(_mm_and_si128(mask, evenVector), _mm_andnot_si128(mask, oddVector));
        }
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ shiftLeft(
        _VectorType_    vector,
        uint32          shift) noexcept
    {
        if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_slli_epi64(cast<_VectorType_, __m128i>(vector), shift));
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_slli_epi32(cast<_VectorType_, __m128i>(vector), shift));
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_slli_epi16(cast<_VectorType_, __m128i>(vector), shift));
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>) {
            uint32 mask = (uint32)0xFF >> (uint32)shift;
            const auto andMask = _mm_and_si128(cast<_VectorType_, __m128i>(vector), _mm_set1_epi8((char)mask));

            return cast<__m128i, _VectorType_>(_mm_sll_epi16(andMask, _mm_cvtsi32_si128(shift)));
        }
    }

    template <
        typename        _DesiredType_,
        _DesiredType_   _Divisor_,
        typename        _VectorType_>
    static simd_stl_always_inline _VectorType_ divideByConst(_VectorType_ dividendVector) noexcept
    {

        return divideByConstHelper<_DesiredType_, _Divisor_, _VectorType_>(dividendVector);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ unaryMinus(_VectorType_ vector) noexcept {
        // 0x80000000 == 0b10000000000000000000000000000000
        if constexpr (is_ps_v<_DesiredType_>)
            return _mm_xor_ps(vector, cast<__m128i, __m128>(_mm_set1_epi32(0x80000000)));
        else if constexpr (is_pd_v<_DesiredType_>)
            return _mm_xor_pd(vector, cast<__m128i, __m128d>(_mm_setr_epi32(0, 0x80000000, 0, 0x80000000)));
        else
            return sub<_DesiredType_>(constructZero<_VectorType_>(), vector);
    }

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
        _DesiredType_* where,
        const type_traits::__deduce_simd_shuffle_mask_type<
        sizeof(_VectorType_) / sizeof(_DesiredType_)>   mask,
        const _VectorType_                                  vector) noexcept
    {
        storeUnaligned(where, shuffle<_DesiredType_>(loadUnaligned<_VectorType_>(where), vector, mask));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void maskStoreAligned(
        _DesiredType_* where,
        const type_traits::__deduce_simd_shuffle_mask_type<
        sizeof(_VectorType_) / sizeof(_DesiredType_)>   mask,
        const _VectorType_                                  vector) noexcept
    {
        storeAligned(where, shuffle<_DesiredType_>(loadAligned<_VectorType_>(where), vector, mask));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ maskLoadUnaligned(
        const _DesiredType_* where,
        const type_traits::__deduce_simd_shuffle_mask_type<
        sizeof(_VectorType_) / sizeof(_DesiredType_)>   mask,
        const _VectorType_                                  vector) noexcept
    {
        return shuffle<_DesiredType_>(loadUnaligned<_VectorType_>(where), vector, mask);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline void maskLoadAligned(
        const _DesiredType_* where,
        const type_traits::__deduce_simd_shuffle_mask_type<
        sizeof(_VectorType_) / sizeof(_DesiredType_)>   mask,
        _VectorType_                                        vector) noexcept
    {
        return shuffle<_DesiredType_>(loadAligned<_VectorType_>(where), vector, mask);
    }

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

    template <typename _VectorType_>
    static simd_stl_always_inline int32 convertToMask(_VectorType_ vector) noexcept {
        return 0;
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ decrement(_VectorType_ vector) noexcept {
        return sub(vector, broadcast(1));
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ increment(_VectorType_ vector) noexcept {
        return add(vector, broadcast(1));
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ constructZero() noexcept {
        if      constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_setzero_pd();
        else if constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_setzero_si128();
        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_setzero_ps();
    }

    template <
        typename _VectorType_,
        typename _DesiredType_>
    static simd_stl_always_inline _VectorType_ broadcast(_DesiredType_ value) noexcept {
        if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_set1_epi64x(value));
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_set1_epi32(value));
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_set1_epi16(value));
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_set1_epi8(value));
        else if constexpr (is_ps_v<_DesiredType_>)
            return cast<__m128, _VectorType_>(_mm_set1_ps(value));
        else if constexpr (is_pd_v<_DesiredType_>)
            return cast<__m128d, _VectorType_>(_mm_set1_pd(value));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ add(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_add_epi64(
                cast<_VectorType_, __m128i>(left),
                cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_add_epi32(
                cast<_VectorType_, __m128i>(left),
                cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_add_epi16(
                cast<_VectorType_, __m128i>(left),
                cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_add_epi8(
                cast<_VectorType_, __m128i>(left),
                cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_ps_v<_DesiredType_>)
            return cast<__m128, _VectorType_>(_mm_add_ps(
                cast<_VectorType_, __m128>(left),
                cast<_VectorType_, __m128>(right)));
        else if constexpr (is_pd_v<_DesiredType_>)
            return cast<__m128d, _VectorType_>(_mm_add_pd(
                cast<_VectorType_, __m128d>(left),
                cast<_VectorType_, __m128d>(right)));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ sub(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_sub_epi64(
                cast<_VectorType_, __m128i>(left),
                cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_sub_epi32(
                cast<_VectorType_, __m128i>(left),
                cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_sub_epi16(
                cast<_VectorType_, __m128i>(left),
                cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_sub_epi8(
                cast<_VectorType_, __m128i>(left),
                cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_ps_v<_DesiredType_>)
            return cast<__m128, _VectorType_>(_mm_sub_ps(
                cast<_VectorType_, __m128>(left),
                cast<_VectorType_, __m128>(right)));
        else if constexpr (is_pd_v<_DesiredType_>)
            return cast<__m128d, _VectorType_>(_mm_sub_pd(
                cast<_VectorType_, __m128d>(left),
                cast<_VectorType_, __m128d>(right)));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ mul(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        // ? 
       /* if      constexpr (is_epi64_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_mul_epi64(
                cast<_VectorType_, __m128i>(left),
                cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epu64_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_mul_epu64(
                cast<_VectorType_, __m128i>(left),
                cast<_VectorType_, __m128i>(right)));
        else if constexpr (is_epi32_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_mul_epi32(
                cast<_VectorType_, __m128i>(left),
                cast<_VectorType_, __m128i>(right)));
        else */if constexpr (is_epu32_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_mul_epu32(
                cast<_VectorType_, __m128i>(left),
                cast<_VectorType_, __m128i>(right)));
        //else if constexpr (is_epi16_v<_DesiredType_>)
        //    return cast<__m128i, _VectorType_>(_mm_mul_epi16(
        //        cast<_VectorType_, __m128i>(left),
        //        cast<_VectorType_, __m128i>(right)));
        //else if constexpr (is_epu16_v<_DesiredType_>)
        //    return cast<__m128i, _VectorType_>(_mm_mul_epu16(
        //        cast<_VectorType_, __m128i>(left),
        //        cast<_VectorType_, __m128i>(right)));
        //else if constexpr (is_epi8_v<_DesiredType_>)
        //    return cast<__m128i, _VectorType_>(_mm_mul_epi8(
        //        cast<_VectorType_, __m128i>(left),
        //        cast<_VectorType_, __m128i>(right)));
        //else if constexpr (is_epu8_v<_DesiredType_>)
        //    return cast<__m128i, _VectorType_>(_mm_mul_epi8(
        //        cast<_VectorType_, __m128i>(left),
        //        cast<_VectorType_, __m128i>(right)));
        //else if constexpr (is_ps_v<_DesiredType_>)
        //    return _mm_mul_ps(left, right);
        else if constexpr (is_pd_v<_DesiredType_>)
            return _mm_mul_pd(left, right);

        return broadcast<_VectorType_>(0);
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ div(
        _VectorType_ left,
        _VectorType_ right) noexcept
    {
        if constexpr (is_pd_v<_DesiredType_>)
            return cast<__m128d, _VectorType_>(_mm_div_pd(
                cast<_VectorType_, __m128d>(left),
                cast<_VectorType_, __m128d>(right)));
        else if constexpr (is_ps_v<_DesiredType_>)
            return cast<__m128, _VectorType_>(_mm_div_ps(
                cast<_VectorType_, __m128>(left),
                cast<_VectorType_, __m128>(right)));
        else if constexpr (is_epi32_v<_DesiredType_>) {

        }
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ bitwiseNot(_VectorType_ vector) noexcept {
        if      constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_xor_pd(vector, _mm_cmpeq_pd(vector, vector));
        else if constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_xor_si128(vector, _mm_cmpeq_epi32(vector, vector));
        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_xor_ps(vector, _mm_cmpeq_ps(vector, vector));
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ bitwiseXor(
        const _VectorType_& left,
        const _VectorType_& right) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_xor_pd(left, right);
        else if constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_xor_si128(left, right);
        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_xor_ps(left, right);
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ bitwiseAnd(
        const _VectorType_& left,
        const _VectorType_& right) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_and_pd(left, right);
        else if constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_and_si128(left, right);
        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_and_ps(left, right);
    }

    template <typename _VectorType_>
    static simd_stl_always_inline _VectorType_ bitwiseOr(
        const _VectorType_& left,
        const _VectorType_& right) noexcept
    {
        if      constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_or_pd(left, right);
        else if constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_or_si128(left, right);
        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_or_ps(left, right);
    }
private:
    template <
        typename        _DesiredType_,
        _DesiredType_   _Divisor_,
        typename        _VectorType_>
    static simd_stl_always_inline _VectorType_ divideByConstHelper(_VectorType_ dividendVector) noexcept
    {
        static_assert(_Divisor_ != 0, "Integer division by zero");

        if constexpr (_Divisor_ == 1)
            return dividendVector;

        if constexpr (is_epu32_v<_DesiredType_>) {
            const auto dividendAsInt32 = cast<_VectorType_, __m128i>(dividendVector);
            constexpr auto trailingZerosInDivisor = math::CountTrailingZeroBits(_Divisor_);

            if constexpr ((uint32(_Divisor_) & (uint32(_Divisor_) - 1)) == 0)
                return _mm_srli_epi32(dividendAsInt32, trailingZerosInDivisor);

            constexpr uint32 magicMultiplier = uint32((uint64(1) << (trailingZerosInDivisor + 32)) / _Divisor_);
            constexpr uint64 magicRemainder = (uint64(1) << (trailingZerosInDivisor + 32)) - uint64(_Divisor_) * magicMultiplier;

            constexpr bool useRoundDown = (2 * magicRemainder < _Divisor_);
            constexpr uint32 adjustedMultiplier = useRoundDown ? magicMultiplier : magicMultiplier + 1;

            const auto multiplierBroadcasted = broadcast<_VectorType_>(uint64(adjustedMultiplier));

            auto lowProduct = _mm_mul_epu32(dividendAsInt32, multiplierBroadcasted);    // Умножаем элементы [0] и [2] на multiplier

            if constexpr (useRoundDown)
                lowProduct = _mm_add_epi64(lowProduct, multiplierBroadcasted);

            auto lowProductShifted = _mm_srli_epi64(lowProduct, 32);                   // Получаем старшие 32 бита результата умножения
            auto highParts = _mm_srli_epi64(dividendAsInt32, 32);              // Получаем элементы [1] и [3] из исходного вектора
            auto highProduct = _mm_mul_epu32(highParts, multiplierBroadcasted);  // Умножаем элементы [1] и [3] на multiplier

            if constexpr (useRoundDown)
                highProduct = _mm_add_epi64(highProduct, multiplierBroadcasted);

            auto low32BitMask = _mm_set_epi32(-1, 0, -1, 0);
            auto highProductMasked = _mm_and_si128(highProduct, low32BitMask);

            auto combinedProduct = _mm_or_si128(lowProductShifted, highProductMasked);

            return cast<__m128i, _VectorType_>(_mm_srli_epi32(combinedProduct, trailingZerosInDivisor));
        }
        else if constexpr (is_epi32_v<_DesiredType_>) {
            if constexpr (_Divisor_ == -1)
                return unaryMinus<_DesiredType_>(dividendVector);

            constexpr uint32 absoluteDivisor = _Divisor_ > 0 ? uint32(_Divisor_) : uint32(-_Divisor_);

            if constexpr ((absoluteDivisor & (absoluteDivisor - 1)) == 0) {
                constexpr auto shiftAmount = math::CountLeadingZeroBits(absoluteDivisor);
                __m128i signBits;

                if constexpr (shiftAmount > 1)
                    signBits = _mm_srai_epi32(dividendVector, shiftAmount - 1);
                else
                    signBits = dividendVector;

                auto roundingBias = _mm_srli_epi32(signBits, 32 - shiftAmount);
                auto biasedDividend = _mm_add_epi32(dividendVector, roundingBias);

                auto quotient = _mm_srai_epi32(biasedDividend, shiftAmount);

                if constexpr (_Divisor_ > 0)
                    return quotient;

                return _mm_sub_epi32(_mm_setzero_si128(), quotient);
            }

            constexpr int32 shiftForMagic = math::CountLeadingZeroBits(uint32_t(absoluteDivisor) - 1);
            constexpr int32 magicMultiplier = int32(1 + ((uint64(1) << (32 + shiftForMagic))
                / uint32(absoluteDivisor)) - (int64(1) << 32));

            SimdDivisor<arch::CpuFeature::SSE2, int32> divisorParams(
                magicMultiplier, shiftForMagic, _Divisor_ < 0 ? -1 : 0);

            const auto productLowEven = _mm_mul_epu32(dividendVector, divisorParams.getMultiplier()); // dividendVector[0], dividendVector[2]
            const auto productHighEven = _mm_srli_epi64(productLowEven, 32);

            const auto shiftedDividendOdd = _mm_srli_epi64(dividendVector, 32); // dividendVector[1], dividendVector[3]
            const auto productLowOdd = _mm_mul_epu32(shiftedDividendOdd, divisorParams.getMultiplier());

            const auto oddMask = _mm_set_epi32(-1, 0, -1, 0);
            const auto productHighOdd = _mm_and_si128(productLowOdd, oddMask);

            const auto combinedHighProduct = _mm_or_si128(productHighEven, productHighOdd);

            const auto dividendSignMask = _mm_srai_epi32(dividendVector, 31);
            const auto multiplierSignMask = _mm_srai_epi32(divisorParams.getMultiplier(), 31);

            const auto correctionFromMultiplier = _mm_and_si128(divisorParams.getMultiplier(), dividendSignMask);
            const auto correctionFromDividend = _mm_and_si128(dividendVector, multiplierSignMask);

            const auto totalCorrection = _mm_add_epi32(correctionFromMultiplier, correctionFromDividend);
            const auto signedProduct = _mm_sub_epi32(combinedHighProduct, totalCorrection);

            const auto adjustedSum = _mm_add_epi32(signedProduct, dividendVector);
            const auto shiftedQuotient = _mm_sra_epi32(adjustedSum, divisorParams.getFirstShiftCount());

            const auto signDifference = _mm_sub_epi32(dividendSignMask, divisorParams.getDivisorSign());
            const auto correctedQuotient = _mm_sub_epi32(shiftedQuotient, signDifference);

            return cast<__m128i, _VectorType_>(_mm_xor_si128(correctedQuotient, divisorParams.getDivisorSign()));
        }
        else if constexpr (is_epi16_v<_DesiredType_>) {
            if constexpr (_Divisor_ == -1)
                return unaryMinus<_DesiredType_>(dividendVector);

            constexpr uint32 absoluteDivisor = _Divisor_ > 0 ? uint32_t(_Divisor_) : uint32_t(-_Divisor_);

            if constexpr ((absoluteDivisor & (absoluteDivisor - 1)) == 0) {
                // Делитель — степень двойки
                constexpr auto shiftAmount = math::CountTrailingZeroBits(absoluteDivisor);
                __m128i signBits;

                if constexpr (shiftAmount > 1)
                    signBits = _mm_srai_epi32(dividendVector, shiftAmount - 1);
                else
                    signBits = dividendVector;

                const auto roundingBias = _mm_srli_epi32(signBits, 32 - shiftAmount);
                const auto biasedDividend = _mm_add_epi32(dividendVector, roundingBias);
                const auto quotient = _mm_srai_epi32(biasedDividend, shiftAmount);

                if constexpr (_Divisor_ > 0)
                    return quotient;

                return _mm_sub_epi32(_mm_setzero_si128(), quotient);
            }

            constexpr auto shiftForMagic = math::CountTrailingZeroBits(uint32(absoluteDivisor) - 1);
            constexpr auto magicMultiplier = int32(1 + ((uint64(1) << (32 + shiftForMagic)) / uint32(absoluteDivisor)) - (int64(1) << 32));

            const SimdDivisor<arch::CpuFeature::SSE2, int32_t> divisorParams(
                magicMultiplier, shiftForMagic, _Divisor_ < 0 ? -1 : 0);

            const auto productLowEven = _mm_mul_epu32(dividendVector, divisorParams.getMultiplier()); // dividendVector[0], dividendVector[2]
            const auto productHighEven = _mm_srli_epi64(productLowEven, 32);

            const auto shiftedDividendOdd = _mm_srli_epi64(dividendVector, 32); // dividendVector[1], dividendVector[3]
            const auto productLowOdd = _mm_mul_epu32(shiftedDividendOdd, divisorParams.getMultiplier());

            const auto oddMask = _mm_set_epi32(-1, 0, -1, 0);
            const auto productHighOdd = _mm_and_si128(productLowOdd, oddMask);

            const auto combinedHighProduct = _mm_or_si128(productHighEven, productHighOdd);

            const auto dividendSignMask = _mm_srai_epi32(dividendVector, 31);
            const auto multiplierSignMask = _mm_srai_epi32(divisorParams.getMultiplier(), 31);

            const auto correctionFromMultiplier = _mm_and_si128(divisorParams.getMultiplier(), dividendSignMask);
            const auto correctionFromDividend = _mm_and_si128(dividendVector, multiplierSignMask);

            const auto totalCorrection = _mm_add_epi32(correctionFromMultiplier, correctionFromDividend);
            const auto signedProduct = _mm_sub_epi32(combinedHighProduct, totalCorrection);

            const auto adjustedSum = _mm_add_epi32(signedProduct, dividendVector);
            const auto shiftedQuotient = _mm_sra_epi32(adjustedSum, divisorParams.getFirstShiftCount());

            const auto signDifference = _mm_sub_epi32(dividendSignMask, divisorParams.getDivisorSign());
            const auto correctedQuotient = _mm_sub_epi32(shiftedQuotient, signDifference);

            return _mm_xor_si128(correctedQuotient, divisorParams.getDivisorSign());
        }
        else if constexpr (is_epu16_v<_DesiredType_>) {
            constexpr int trailingZeroBitCount = math::CountTrailingZeroBits(_Divisor_);

            if constexpr ((_Divisor_ & (_Divisor_ - 1u)) == 0)
                return _mm_srli_epi16(dividendVector, trailingZeroBitCount);

            constexpr auto magicDivisionMultiplier = uint16((1u << uint32(trailingZeroBitCount + 16)) / _Divisor_);

            constexpr uint32_t magicDivisionRemainder = ((uint32_t(1) << uint32_t(trailingZeroBitCount + 16))
                - uint32_t(_Divisor_) * magicDivisionMultiplier);

            constexpr bool shouldRoundDown = (2u * magicDivisionRemainder < _Divisor_);

            if (shouldRoundDown)
                dividendVector = dividendVector + _mm_set1_epi16(1);

            constexpr uint16 adjustedMagicMultiplier = shouldRoundDown
                ? magicDivisionMultiplier
                : magicDivisionMultiplier + 1;

            const auto multiplierVector = _mm_set1_epi16(static_cast<int16_t>(adjustedMagicMultiplier));

            auto highProductVector = _mm_mulhi_epu16(dividendVector, multiplierVector);
            auto quotientVector = _mm_srli_epi16(highProductVector, trailingZeroBitCount);

            if constexpr (shouldRoundDown) {
                auto isDividendZeroMask = _mm_cmpeq_epi16(dividendVector, _mm_setzero_si128());

                return _mm_or_si128(
                    _mm_and_si128(
                        isDividendZeroMask,
                        broadcast<__m128i>(uint16_t(adjustedMagicMultiplier >> trailingZeroBitCount))
                    ),
                    _mm_andnot_si128(quotientVector, _mm_set1_epi16(trailingZeroBitCount))
                );
            }
            else
                return quotientVector;

        }
    }
};

template <>
class BasicSimdImplementation<arch::CpuFeature::SSE3> :
    public BasicSimdImplementation<arch::CpuFeature::SSE2>
{

};

template <>
class BasicSimdImplementation<arch::CpuFeature::SSSE3> :
    public BasicSimdImplementation<arch::CpuFeature::SSE3>
{
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
                shuffleMask));
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_> || is_ps_v<_DesiredType_>)
            return cast<__m128i, _VectorType_>(_mm_shuffle_epi32(
                cast<_VectorType_, __m128i>(vector),
                cast<_VectorType_, __m128i>(secondVector),
                shuffleMask)
            );
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>) {
            const auto rawMask = shuffleMask & 0xFFFFFF;

            const int8 index0 = (rawMask >> 0) & 0x7;
            const int8 index1 = (rawMask >> 3) & 0x7;

            const int8 index2 = (rawMask >> 6) & 0x7;
            const int8 index3 = (rawMask >> 9) & 0x7;

            const int8 index4 = (rawMask >> 12) & 0x7;
            const int8 index5 = (rawMask >> 15) & 0x7;

            const int8 index6 = (rawMask >> 18) & 0x7;
            const int8 index7 = (rawMask >> 21) & 0x7;

            const int8 low0 = index0 << 1;
            const int8 low1 = index1 << 1;

            const int8 low2 = index2 << 1;
            const int8 low3 = index3 << 1;

            const int8 low4 = index4 << 1;
            const int8 low5 = index5 << 1;

            const int8 low6 = index6 << 1;
            const int8 low7 = index7 << 1;

            const auto byteMask = _mm_set_epi8(
                low7 + 1, low7, low6 + 1, low6, low5 + 1, low5, low4 + 1, low4,
                low3 + 1, low3, low2 + 1, low2, low1 + 1, low1, low0 + 1, low0
            );

            return _mm_shuffle_epi8(vector, byteMask);
        }
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>) {
            // return _mm_shuffle_epi8(vector, (shuffleMask));
        }
    }
};

template <>
class BasicSimdImplementation<arch::CpuFeature::SSE41> :
    public BasicSimdImplementation<arch::CpuFeature::SSSE3>
{
};

template <>
class BasicSimdImplementation<arch::CpuFeature::SSE42> :
    public BasicSimdImplementation<arch::CpuFeature::SSE41>
{
};