#pragma once 

#include <src/simd_stl/numeric/SimdIntegralTypesCheck.h>
#include <src/simd_stl/numeric/SimdConvert.h>

#include <simd_stl/numeric/BasicSimdShuffleMask.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_>
class _SimdElementWise;

template <class _RegisterPolicy_>
class _SimdElementWise<arch::CpuFeature::SSE2, _RegisterPolicy_> {
    static constexpr auto _Feature = arch::CpuFeature::SSE2;

    using _Cast_    = _SimdCast<_Feature, _RegisterPolicy_>;
    using _Convert_ = _SimdConvert<_Feature, _RegisterPolicy_>;
public:
    template <
        typename    _DesiredType_,
        uint8 ...   _Indices_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ permute(_VectorType_ vector) noexcept
    {
        if      constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_> || is_pd_v<_DesiredType_>)
            return _Cast_::template cast<__m128d, _VectorType_>(_mm_shuffle_pd(
                _Cast_::template cast<_VectorType_, __m128d>(vector),
                _Cast_::template cast<_VectorType_, __m128d>(vector),
                basic_simd_permute_mask<_Indices_...>::template unwrap<_Feature, _DesiredType_>())
            );
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_> || is_ps_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_shuffle_epi32(
                _Cast_::template cast<_VectorType_, __m128i>(vector),
                basic_simd_permute_mask<_Indices_...>::template unwrap<_Feature, _DesiredType_>())
            );
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>) {
            _DesiredType_ sourceVector[8], result[8];

            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(sourceVector),
                _Cast_::template cast<_VectorType_, __m128i>(vector));

            auto index = basic_simd_permute_mask<_Indices_...>::template unwrap<_Feature, _DesiredType_>();

            for (int j = 0; j < 8; j++) 
                result[j] = sourceVector[index[j] & 0x07];

            return _Cast_::template cast<__m128i, _VectorType_>(_mm_loadu_si128(reinterpret_cast<const __m128i*>(result)));
        }
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>) {
            _DesiredType_ sourceVector[16], result[16];

            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(sourceVector), 
                _Cast_::template cast<_VectorType_, __m128i>(vector));

            auto index = basic_simd_permute_mask<_Indices_...>::template unwrap<_Feature, _DesiredType_>();

            for (int32 j = 0; j < 16; j++) 
                result[j] = sourceVector[index[j] & 0x0F];

            return _Cast_::template cast<__m128i, _VectorType_>(_mm_loadu_si128(reinterpret_cast<const __m128i*>(result)));
        }
    }

    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ shuffle(
        _VectorType_                                                                                vector,
        type_traits::__deduce_simd_shuffle_mask_type<_Feature, _DesiredType_, _RegisterPolicy_>     mask) noexcept
    {
        if      constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_> || is_pd_v<_DesiredType_>)
            return _Cast_::template cast<__m128d, _VectorType_>(_mm_shuffle_pd(
                _Cast_::template cast<_VectorType_, __m128d>(vector),
                _Cast_::template cast<_VectorType_, __m128d>(vector), mask)
            );
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_> || is_ps_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_shuffle_epi32(
                _Cast_::template cast<_VectorType_, __m128i>(vector), mask)
            );
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>) {
            _DesiredType_ sourceVector[8], result[8];

            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(sourceVector),
                _Cast_::template cast<_VectorType_, __m128i>(vector));

            for (int j = 0; j < 8; j++)
                result[j] = sourceVector[(mask >> (j * 3)) & 0x07];

            return _Cast_::template cast<__m128i, _VectorType_>(_mm_loadu_si128(reinterpret_cast<const __m128i*>(result)));
        }
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>) {
            _DesiredType_ sourceVector[16], result[16];

            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(sourceVector), 
                _Cast_::template cast<_VectorType_, __m128i>(vector));

            for (int32 j = 0; j < 16; j++) 
                result[j] = sourceVector[(mask >> (j * 4)) & 0x0F];

            return _Cast_::template cast<__m128i, _VectorType_>(_mm_loadu_si128(reinterpret_cast<const __m128i*>(result)));
        }
    }



    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ blend(
        _VectorType_                                                                        firstVector,
        _VectorType_                                                                        secondVector,
        type_traits::__deduce_simd_mask_type<_Feature, _DesiredType_, _RegisterPolicy_>     mask) noexcept
    {
/*        if constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_> || is_pd_v<_DesiredType_>) {
            return _Cast_::cast<__m128d, _VectorType_>(_mm_shuffle_pd(
                _Cast_::cast<_VectorType_, __m128d>(secondVector),
                _Cast_::cast<_VectorType_, __m128d>(firstVector), (~mask))
            );
        }
        else */{
            constexpr auto length = sizeof(__m128i) / sizeof(_DesiredType_);

            _DesiredType_ first[length], second[length], result[length];

            _mm_storeu_si128(reinterpret_cast<__m128i*>(first), _Cast_::template cast<_VectorType_, __m128i>(firstVector));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(second), _Cast_::template cast<_VectorType_, __m128i>(secondVector));
            
            for (auto current = 0; current < length; ++current)
                result[current] = ((mask >> current) & 1) ? second[current] : first[current];
        
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_loadu_si128(reinterpret_cast<const __m128i*>(result)));
        }
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ reverse(_VectorType_ vector) noexcept {
        if constexpr (sizeof(_DesiredType_) == 8) {
            const auto casted = _Cast_::template cast<_VectorType_, __m128d>(vector);
            return _Cast_::template cast<__m128d, _VectorType_>(_mm_shuffle_pd(casted, casted, 0b01));
        }
        else if constexpr (sizeof(_DesiredType_) == 4) {
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_shuffle_epi32(
                _Cast_::template cast<_VectorType_, __m128i>(vector), 0b00011011));
        }
        else if constexpr (sizeof(_DesiredType_) == 2) {
            vector = _Cast_::template cast<__m128d, _VectorType_>(_mm_shuffle_pd(
                _Cast_::template cast<_VectorType_, __m128d>(vector), _Cast_::template cast<_VectorType_, __m128d>(vector), 0b01));

            vector = _Cast_::template cast<__m128i, _VectorType_>(_mm_shufflehi_epi16(vector, 0b00011011));
            vector = _Cast_::template cast<__m128i, _VectorType_>(_mm_shufflelo_epi16(vector, 0b00011011));

            return vector;
        }
        else if constexpr (sizeof(_DesiredType_) == 1) {
            vector = _Cast_::template cast<__m128i, _VectorType_>(
                _mm_or_si128(
                    _mm_srli_epi16(_Cast_::template cast<_VectorType_, __m128i>(vector), 8),
                    _mm_slli_epi16(_Cast_::template cast<_VectorType_, __m128i>(vector), 8)
                )
            );

            vector = _Cast_::template cast<__m128i, _VectorType_>(
                _mm_shufflelo_epi16(_Cast_::template cast<_VectorType_, __m128i>(vector), 0b00011011));

            vector = _Cast_::template cast<__m128i, _VectorType_>(
                _mm_shufflehi_epi16(_Cast_::template cast<_VectorType_, __m128i>(vector), 0b00011011));
  
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_shuffle_epi32(
                _Cast_::template cast<_VectorType_, __m128i>(vector), 0x4E));
        }
    }
};

template <class _RegisterPolicy_>
class _SimdElementWise<arch::CpuFeature::SSE3, _RegisterPolicy_> :
    public _SimdElementWise<arch::CpuFeature::SSE2, _RegisterPolicy_>
{};

template <class _RegisterPolicy_>
class _SimdElementWise<arch::CpuFeature::SSSE3, _RegisterPolicy_> :
    public _SimdElementWise<arch::CpuFeature::SSE3, _RegisterPolicy_>
{
    using _Cast_    = _SimdCast<arch::CpuFeature::SSSE3, _RegisterPolicy_>;
    using _Convert_ = _SimdConvert<arch::CpuFeature::SSSE3, _RegisterPolicy_>;
public:
   /* template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ shuffle(
        _VectorType_                                                            vector,
        type_traits::__deduce_simd_shuffle_mask_type<_Feature, _DesiredType_>   mask) noexcept
    {
        if      constexpr (is_epi64_v<_DesiredType_> || is_epu64_v<_DesiredType_> || is_pd_v<_DesiredType_>)
            return _Cast_::template cast<__m128d, _VectorType_>(_mm_shuffle_pd(
                _Cast_::template cast<_VectorType_, __m128d>(vector),
                _Cast_::template cast<_VectorType_, __m128d>(vector), mask)
            );
        else if constexpr (is_epi32_v<_DesiredType_> || is_epu32_v<_DesiredType_> || is_ps_v<_DesiredType_>)
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_shuffle_epi32(
                _Cast_::template cast<_VectorType_, __m128i>(vector), mask)
            );
        else if constexpr (is_epi16_v<_DesiredType_> || is_epu16_v<_DesiredType_>) {
            _DesiredType_ sourceVector[8], result[8];

            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(sourceVector),
                _Cast_::template cast<_VectorType_, __m128i>(vector));

            for (int j = 0; j < 8; j++)
                result[j] = sourceVector[(mask >> (j * 3)) & 0x07];

            return _Cast_::template cast<__m128i, _VectorType_>(_mm_loadu_si128(reinterpret_cast<const __m128i*>(result)));
        }
        else if constexpr (is_epi8_v<_DesiredType_> || is_epu8_v<_DesiredType_>) {
            return _Cast_::template cast<__m128i, _VectorType_>(_mm_shuffle_epi8(
                _Cast_::template cast<_VectorType_, __m128i>(vector),
                _Convert_::template convertFromMask<__m128i>(mask)
            ));
        }
    }*/
};

template <class _RegisterPolicy_>
class _SimdElementWise<arch::CpuFeature::SSE41, _RegisterPolicy_> :
    public _SimdElementWise<arch::CpuFeature::SSSE3, _RegisterPolicy_>
{};

template <class _RegisterPolicy_>
class _SimdElementWise<arch::CpuFeature::SSE42, _RegisterPolicy_> :
    public _SimdElementWise<arch::CpuFeature::SSE41, _RegisterPolicy_>
{};

__SIMD_STL_NUMERIC_NAMESPACE_END
