#pragma once 

#include <src/simd_stl/numeric/SimdIntegralTypesCheck.h>
#include <src/simd_stl/numeric/SimdConvert.h>

#include <simd_stl/numeric/BasicSimdShuffleMask.h>


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_>
class _SimdElementWise;

template <>
class _SimdElementWise<arch::CpuFeature::SSE2, xmm128> {
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
            const auto _Shuffle = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
            return _IntrinBitcast<_VectorType_>(_mm_shuffle_epi8(_IntrinBitcast<__m128i>(_Vector), _Shuffle));
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

__SIMD_STL_NUMERIC_NAMESPACE_END
