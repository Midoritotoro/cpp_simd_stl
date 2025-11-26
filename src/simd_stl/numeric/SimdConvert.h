#pragma once 

#include <src/simd_stl/numeric/IntrinBitcast.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_>
class _SimdConvertImplementation;

template <>
class _SimdConvertImplementation<arch::CpuFeature::SSE2, xmm128> {
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline uint32 _ToMask(_VectorType_ _Vector) noexcept {
        if      constexpr (sizeof(_DesiredType_) == 8)
            return _mm_movemask_pd(_IntrinBitcast<__m128d>(_Vector));

        else if constexpr (sizeof(_DesiredType_) == 4)
            return _mm_movemask_ps(_IntrinBitcast<__m128>(_Vector));

        else if constexpr (sizeof(_DesiredType_) == 2)
            return _mm_movemask_epi8(_mm_packs_epi16(_IntrinBitcast<__m128i>(_Vector), _mm_setzero_si128()));

        else if constexpr (sizeof(_DesiredType_) == 1)
            return _mm_movemask_epi8(_IntrinBitcast<__m128i>(_Vector));
    }

    template <
        typename _VectorType_,
        typename _MaskType_>
    static simd_stl_always_inline _VectorType_ _ToVector(_MaskType_ _Mask) noexcept {
        if (sizeof(_MaskType_) == 4) {
            _MaskType_ _SourceVector[8], _Result[8];

            _mm_storeu_si128(reinterpret_cast<__m128i*>(_SourceVector), _mm_setr_epi32(0, 0, 0x01010101, 0x01010101));

            for (int j = 0; j < 8; ++j)
                _Result[j] = _SourceVector[(_Mask >> (j * 4)) & 0x07];

            const auto _BitSelect = _mm_setr_epi8(
                1, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6, 1U << 7,
                1, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6, 1U << 7);

            auto _Value = _mm_loadu_si128(static_cast<const __m128i*>(_Result));

            _Value = _mm_and_si128(_Value, _BitSelect);
            _Value = _mm_min_epu8(_Value, _mm_set1_epi8(1));

            return _IntrinBitcast<_VectorType_>(_Value);
        }
    }
};

template <>
class _SimdConvertImplementation<arch::CpuFeature::SSE3, xmm128> :
    public _SimdConvertImplementation<arch::CpuFeature::SSE2, xmm128>
{};

template <>
class _SimdConvertImplementation<arch::CpuFeature::SSSE3, xmm128> :
    public _SimdConvertImplementation<arch::CpuFeature::SSE3, xmm128>
{
public:
    template <
        typename _VectorType_,
        typename _MaskType_>
    static simd_stl_always_inline _VectorType_ _ToVector(_MaskType_ _Mask) noexcept {
        if (sizeof(_MaskType_) == 4) {
            const auto _Shuffle = _mm_setr_epi32(0, 0, 0x01010101, 0x01010101);
            auto _Value = _mm_shuffle_epi8(_mm_cvtsi32_si128(_Mask), shuffle);

            const auto _BitSelect = _mm_setr_epi8(
                1, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6, 1U << 7,
                1, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6, 1U << 7);

            _Value = _mm_and_si128(_Value, _BitSelect);
            _Value = _mm_min_epu8(_Value, _mm_set1_epi8(1));

            return _IntrinBitcast<_VectorType_>(_Value);
        }
    }
};

template <>
class _SimdConvertImplementation<arch::CpuFeature::SSE41, xmm128> :
    public _SimdConvertImplementation<arch::CpuFeature::SSSE3, xmm128>
{};

template <>
class _SimdConvertImplementation<arch::CpuFeature::SSE42, xmm128> :
    public _SimdConvertImplementation<arch::CpuFeature::SSE41, xmm128>
{};


template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline uint32 _SimdToMask(_VectorType_ _Vector) noexcept {
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_)
    return _SimdConvertImplementation<_SimdGeneration_, _RegisterPolicy_>::template _ToMask<_DesiredType_>(_Vector);
}

__SIMD_STL_NUMERIC_NAMESPACE_END
