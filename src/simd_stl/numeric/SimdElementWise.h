#pragma once 

#include <src/simd_stl/numeric/SimdIntegralTypesCheck.h>
#include <src/simd_stl/numeric/SimdConvert.h>

#include <simd_stl/numeric/BasicSimdShuffleMask.h>
#include <src/simd_stl/numeric/ShuffleTables.h>

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_>
class _SimdElementWise;

template <>
class _SimdElementWise<arch::CpuFeature::SSE2, xmm128> {
    static constexpr auto _Generation   = arch::CpuFeature::SSE2;
    using _RegisterPolicy               = xmm128;

    template <typename _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ _Blend(
        _VectorType_                        _First,
        _VectorType_                        _Second,
        _Simd_mask_type<_DesiredType_>      _Mask) noexcept
    {
        {
            constexpr auto length = sizeof(__m128i) / sizeof(_DesiredType_);
            _DesiredType_ first[length], second[length], result[length];

            _mm_storeu_si128(reinterpret_cast<__m128i*>(first), _IntrinBitcast<__m128i>(_First));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(second), _IntrinBitcast<__m128i>(_Second));

            for (auto current = 0; current < length; ++current)
                result[current] = ((_Mask >> current) & 1) ? second[current] : first[current];

            return _IntrinBitcast<_VectorType_>(_mm_loadu_si128(reinterpret_cast<const __m128i*>(result)));
        }
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Reverse(_VectorType_ _Vector) noexcept {
        if constexpr (sizeof(_DesiredType_) == 8) {
            return _IntrinBitcast<_VectorType_>(_mm_shuffle_pd(
                _IntrinBitcast<__m128d>(_Vector), _IntrinBitcast<__m128d>(_Vector), 1));
        }
        else if constexpr (sizeof(_DesiredType_) == 4) {
            return _IntrinBitcast<_VectorType_>(_mm_shuffle_epi32(_IntrinBitcast<__m128i>(_Vector), 0x1B));
        }
        else if constexpr (sizeof(_DesiredType_) == 2) {
            _Vector = _IntrinBitcast<_VectorType_>(_mm_shuffle_pd(
                _IntrinBitcast<__m128d>(_Vector), _IntrinBitcast<__m128d>(_Vector), 1));

            _Vector = _IntrinBitcast<_VectorType_>(_mm_shufflehi_epi16(_IntrinBitcast<__m128i>(_Vector), 0x1B));
            return _IntrinBitcast<_VectorType_>(_mm_shufflelo_epi16(_IntrinBitcast<__m128i>(_Vector), 0x1B));
        }
        else if constexpr (sizeof(_DesiredType_) == 1) {
            _Vector = _IntrinBitcast<_VectorType_>(_mm_or_si128(
                _mm_srli_epi16(_IntrinBitcast<__m128i>(_Vector), 8),
                _mm_slli_epi16(_IntrinBitcast<__m128i>(_Vector), 8)));

            _Vector = _IntrinBitcast<_VectorType_>(_mm_shufflelo_epi16(_IntrinBitcast<__m128i>(_Vector), 0x1B));
            _Vector = _IntrinBitcast<_VectorType_>(_mm_shufflehi_epi16(_IntrinBitcast<__m128i>(_Vector), 0x1B));
  
            return _IntrinBitcast<_VectorType_>(_mm_shuffle_epi32(_IntrinBitcast<__m128i>(_Vector), 0x4E));
        }
    }
};

template <>
class _SimdElementWise<arch::CpuFeature::SSE3, xmm128> :
    public _SimdElementWise<arch::CpuFeature::SSE2, xmm128>
{
};

template <>
class _SimdElementWise<arch::CpuFeature::SSSE3, xmm128> :
    public _SimdElementWise<arch::CpuFeature::SSE3, xmm128>
{
public:
    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Reverse(_VectorType_ _Vector) noexcept {
        if constexpr (sizeof(_DesiredType_) == 8) {
            return _IntrinBitcast<_VectorType_>(_mm_shuffle_pd(
                _IntrinBitcast<__m128d>(_Vector), _IntrinBitcast<__m128d>(_Vector), 1));
        }
        else if constexpr (sizeof(_DesiredType_) == 4) {
            return _IntrinBitcast<_VectorType_>(_mm_shuffle_epi32(_IntrinBitcast<__m128i>(_Vector), 0x1B));
        }
        else if constexpr (sizeof(_DesiredType_) == 2) {
            _Vector = _IntrinBitcast<_VectorType_>(_mm_shuffle_pd(
                _IntrinBitcast<__m128d>(_Vector), _IntrinBitcast<__m128d>(_Vector), 1));

            _Vector = _IntrinBitcast<_VectorType_>(_mm_shufflehi_epi16(_IntrinBitcast<__m128i>(_Vector), 0x1B));
            return _IntrinBitcast<_VectorType_>(_mm_shufflelo_epi16(_IntrinBitcast<__m128i>(_Vector), 0x1B));
        }
        else if constexpr (sizeof(_DesiredType_) == 1) {
            return _IntrinBitcast<_VectorType_>(_mm_shuffle_epi8(_IntrinBitcast<__m128i>(_Vector),
                _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)));
        }
    }
};

template <>
class _SimdElementWise<arch::CpuFeature::SSE41, xmm128> :
    public _SimdElementWise<arch::CpuFeature::SSSE3, xmm128>
{};

template <>
class _SimdElementWise<arch::CpuFeature::SSE42, xmm128> :
    public _SimdElementWise<arch::CpuFeature::SSE41, xmm128>
{};


template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ _SimdReverse(_VectorType_ _Vector) noexcept {
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdElementWise<_SimdGeneration_, _RegisterPolicy_>::template _Reverse<_DesiredType_>(_Vector);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ _SimdBlend(
    _VectorType_                            _First,
    _VectorType_                            _Second,
    type_traits::__deduce_simd_mask_type<_SimdGeneration_,
        _DesiredType_, _RegisterPolicy_>    _Mask) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdElementWise<_SimdGeneration_, _RegisterPolicy_>::template _Blend<_DesiredType_>(_First, _Second, _Mask);
}

__SIMD_STL_NUMERIC_NAMESPACE_END
