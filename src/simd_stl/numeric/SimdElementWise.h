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

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ _SimdReverse(_VectorType_ _Vector) noexcept;

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ _SimdBlend(
    _VectorType_                            _First,
    _VectorType_                            _Second,
    type_traits::__deduce_simd_mask_type<_SimdGeneration_,
        _DesiredType_, _RegisterPolicy_>    _Mask) noexcept;

#pragma region Sse2-Sse4.2 Simd element wise 

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
        _VectorType_ _First,
        _VectorType_ _Second,
        _VectorType_ _Mask) noexcept
    {
        return _IntrinBitcast<_VectorType_>(_mm_or_si128(
            _mm_and_si128(_IntrinBitcast<__m128i>(_Mask), _IntrinBitcast<__m128i>(_First)),
            _mm_andnot_si128(_IntrinBitcast<__m128i>(_Mask), _IntrinBitcast<__m128i>(_Second))));
    }

    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ _Blend(
        _VectorType_                        _First,
        _VectorType_                        _Second,
        _Simd_mask_type<_DesiredType_>      _Mask) noexcept
    {
        return _Blend<_DesiredType_>(_First, _Second, _SimdToVector<_Generation, _RegisterPolicy, _VectorType_, _DesiredType_>(_Mask));
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
{
    static constexpr auto _Generation   = arch::CpuFeature::SSE41;
    using _RegisterPolicy               = xmm128;

    template <typename _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ _Blend(
        _VectorType_ _First,
        _VectorType_ _Second,
        _VectorType_ _Mask) noexcept
    {
        return _mm_blendv_epi8(_IntrinBitcast<__m128i>(_Second),
            _IntrinBitcast<__m128i>(_First), _IntrinBitcast<__m128i>(_Mask));
    }

    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ _Blend(
        _VectorType_                        _First,
        _VectorType_                        _Second,
        _Simd_mask_type<_DesiredType_>      _Mask) noexcept
    {
        return _Blend<_DesiredType_>(_First, _Second, _SimdToVector<_Generation, _RegisterPolicy, _VectorType_>(_Mask));
    }
};

template <>
class _SimdElementWise<arch::CpuFeature::SSE42, xmm128> :
    public _SimdElementWise<arch::CpuFeature::SSE41, xmm128>
{};

#pragma endregion

#pragma region Avx-Avx2 Simd element wise

template <>
class _SimdElementWise<arch::CpuFeature::AVX, ymm256>
{
    static constexpr auto _Generation   = arch::CpuFeature::AVX;
    using _RegisterPolicy               = ymm256;

    template <typename _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ _Blend(
        _VectorType_ _First,
        _VectorType_ _Second,
        _VectorType_ _Mask) noexcept
    {
        return _IntrinBitcast<_VectorType_>(_mm256_or_ps(
            _mm256_and_ps(_IntrinBitcast<__m256>(_Mask), _IntrinBitcast<__m256>(_First)),
            _mm256_andnot_ps(_IntrinBitcast<__m256>(_Mask), _IntrinBitcast<__m256>(_Second))));
    }

    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ _Blend(
        _VectorType_                        _First,
        _VectorType_                        _Second,
        _Simd_mask_type<_DesiredType_>      _Mask) noexcept
    {
        return _Blend<_DesiredType_>(_First, _Second, _SimdToVector<_Generation, _RegisterPolicy, _VectorType_>(_Mask));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Reverse(_VectorType_ _Vector) noexcept {
        if constexpr (sizeof(_DesiredType_) == 8) {
            const auto _ReversedXmmLanes = _mm256_shuffle_pd(
                _IntrinBitcast<__m256d>(_Vector), _IntrinBitcast<__m256d>(_Vector), 0x05);

            return _IntrinBitcast<_VectorType_>(_mm256_permute2f128_pd(_ReversedXmmLanes, _ReversedXmmLanes, 1));
        }
        else if constexpr (sizeof(_DesiredType_) == 4) {
            const auto _ReversedXmmLanes = _mm256_shuffle_ps(_IntrinBitcast<__m256>(_Vector), 0x1B);
            return _IntrinBitcast<_VectorType_>(_mm256_permute2f128_ps(_ReversedXmmLanes, _ReversedXmmLanes, 1));
        }
        else if constexpr (sizeof(_DesiredType_) == 2) {
            auto _Low   = _IntrinBitcast<__m128i>(_Vector);
            auto _High  = _mm256_extractf128_si256(_IntrinBitcast<__m256i>(_Vector), 1);

            const auto _Mask = _mm_set_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);

            _Low    = _mm_shuffle_epi8(_Low, _Mask);
            _High   = _mm_shuffle_epi8(_High, _Mask);

            return _mm256_insertf128_si256(_IntrinBitcast<__m256i>(_High), _Low, 1);
        }
        else if constexpr (sizeof(_DesiredType_) == 1) {
            auto _Low   = _IntrinBitcast<__m128i>(_Vector);
            auto _High  = _mm256_extractf128_si256(_IntrinBitcast<__m256i>(_Vector), 1);

            const auto _Mask = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

            _Low    = _mm_shuffle_epi8(_Low, _Mask);
            _High   = _mm_shuffle_epi8(_High, _Mask);

            return _mm256_insertf128_si256(_IntrinBitcast<__m256i>(_High), _Low, 1);
        }
    }
};

template <>
class _SimdElementWise<arch::CpuFeature::AVX2, ymm256>:
    public _SimdElementWise<arch::CpuFeature::AVX, ymm256>
{
    static constexpr auto _Generation   = arch::CpuFeature::AVX;
    using _RegisterPolicy               = ymm256;

    template <typename _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ _Blend(
        _VectorType_ _First,
        _VectorType_ _Second,
        _VectorType_ _Mask) noexcept
    {
        return _mm256_blendv_epi8(_IntrinBitcast<__m256i>(_Second),
            _IntrinBitcast<__m256i>(_First), _IntrinBitcast<__m256i>(_Mask));
    }

    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ _Blend(
        _VectorType_                        _First,
        _VectorType_                        _Second,
        _Simd_mask_type<_DesiredType_>      _Mask) noexcept
    {
        return _Blend<_DesiredType_>(_First, _Second, _SimdToVector<_Generation, _RegisterPolicy, _VectorType_>(_Mask));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Reverse(_VectorType_ _Vector) noexcept {
        if constexpr (sizeof(_DesiredType_) == 2) {
            const auto _ReversedXmmLanes = _IntrinBitcast<__m256>(_mm256_shuffle_epi8(_IntrinBitcast<__m256i>(_Vector),
                _mm256_setr_epi8(30, 31, 28, 29, 26, 27, 24, 25, 22, 23, 20, 21, 18, 19, 16, 17,
                    14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1)));

            return _IntrinBitcast<_VectorType_>(_mm256_permute2f128_si256(_ReversedXmmLanes, _ReversedXmmLanes, 1));
        }
        else if constexpr (sizeof(_DesiredType_) == 1) {
            const auto _ReversedXmmLanes = _IntrinBitcast<__m256>(_mm256_shuffle_epi8(_IntrinBitcast<__m256i>(_Vector),
                _mm256_setr_epi8(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 
                    15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)));

            return _IntrinBitcast<_VectorType_>(_mm256_permute2f128_si256(_ReversedXmmLanes, _ReversedXmmLanes, 1));
        }
        else {
            return _SimdReverse<arch::CpuFeature::AVX, _RegisterPolicy, _DesiredType_>(_Vector);
        }
    }
};

#pragma endregion

#pragma region Avx512 Simd element wise

template <>
class _SimdElementWise<arch::CpuFeature::AVX512F, zmm512>
{
    static constexpr auto _Generation   = arch::CpuFeature::AVX512F;
    using _RegisterPolicy               = zmm512;

    template <typename _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ _Blend(
        _VectorType_ _First,
        _VectorType_ _Second,
        _VectorType_ _Mask) noexcept
    {
        return _IntrinBitcast<_VectorType_>(_mm512_or_si512(
            _mm512_and_si512(_IntrinBitcast<__m512i>(_Mask), _IntrinBitcast<__m512i>(_First)),
            _mm512_andnot_si512(_IntrinBitcast<__m512i>(_Mask), _IntrinBitcast<__m512i>(_Second))));
    }

    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ _Blend(
        _VectorType_                        _First,
        _VectorType_                        _Second,
        _Simd_mask_type<_DesiredType_>      _Mask) noexcept
    {
        return _Blend<_DesiredType_>(_First, _Second, _SimdToVector<_Generation, _RegisterPolicy, _VectorType_>(_Mask));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Reverse(_VectorType_ _Vector) noexcept {
        if constexpr (sizeof(_DesiredType_) == 8) {
            return _IntrinBitcast<_VectorType_>(_mm512_permutexvar_epi64(
                _mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0),
                _IntrinBitcast<__m512i>(_Vector)));
        }
        else if constexpr (sizeof(_DesiredType_) == 4) {
            return _IntrinBitcast<_VectorType_>(_mm512_permutexvar_epi32(
                _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
                _IntrinBitcast<__m512i>(_Vector)));
        }
        else {
            const auto _Low     = _SimdReverse<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(_IntrinBitcast<__m256i>(_Vector));
            const auto _High    = _SimdReverse<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(
                _mm512_extracti64x4_epi64(_IntrinBitcast<__m512i>(_Vector), 1));

            return _IntrinBitcast<_VectorType_>(_mm512_inserti64x4(_IntrinBitcast<__m512i>(_Low), _IntrinBitcast<__m256i>(_High), 1));
        }
    }
};

template <>
class _SimdElementWise<arch::CpuFeature::AVX512BW, zmm512>:
    public _SimdElementWise<arch::CpuFeature::AVX512F, zmm512>
{
    static constexpr auto _Generation   = arch::CpuFeature::AVX512BW;
    using _RegisterPolicy               = zmm512;

    template <typename _DesiredType_>
    using _Simd_mask_type = type_traits::__deduce_simd_mask_type<_Generation, _DesiredType_, _RegisterPolicy>;
public:
    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ _Blend(
        _VectorType_ _First,
        _VectorType_ _Second,
        _VectorType_ _Mask) noexcept
    {
        return _Blend<_DesiredType_>(_First, _Second, _SimdToMask<_Generation, _RegisterPolicy, _DesiredType_>(_Mask));
    }

    template <
        typename    _DesiredType_,
        typename    _VectorType_>
    static simd_stl_always_inline _VectorType_ _Blend(
        _VectorType_                        _First,
        _VectorType_                        _Second,
        _Simd_mask_type<_DesiredType_>      _Mask) noexcept
    {
        return _IntrinBitcast<_VectorType_>(_mm512_mask_blend_epi8(
            _Mask, _IntrinBitcast<__m512i>(_First), _IntrinBitcast<__m512i>(_Second)));
    }

    template <
        typename _DesiredType_,
        typename _VectorType_>
    static simd_stl_always_inline _VectorType_ _Reverse(_VectorType_ _Vector) noexcept {   
        if constexpr (sizeof(_DesiredType_) == 2) {
            const auto _Shuffle = _mm512_setr_epi16(
                31, 30, 29, 28, 27, 26, 25, 24,
                23, 22, 21, 20, 19, 18, 17, 16,
                15, 14, 13, 12, 11, 10, 9, 8,
                7, 6, 5, 4, 3, 2, 1, 0);

            return _IntrinBitcast<_VectorType_>(_mm512_permutexvar_epi16(_Shuffle, _IntrinBitcast<__m512i>(_Vector)));
        }
        else {
            return _SimdReverse<arch::CpuFeature::AVX512F, _RegisterPolicy, _DesiredType_>(_Vector);
        }
    }
};

template <>
class _SimdElementWise<arch::CpuFeature::AVX512DQ, zmm512> :
    public _SimdElementWise<arch::CpuFeature::AVX512BW, zmm512>
{};

template <>
class _SimdElementWise<arch::CpuFeature::AVX512VL, zmm512> :
    public _SimdElementWise<arch::CpuFeature::AVX512DQ, zmm512>
{};

#pragma endregion

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

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    typename            _DesiredType_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ _SimdBlend(
    _VectorType_    _First,
    _VectorType_    _Second,
    _VectorType_    _Mask) noexcept
{
    _VerifyRegisterPolicy(_SimdGeneration_, _RegisterPolicy_);
    return _SimdElementWise<_SimdGeneration_, _RegisterPolicy_>::template _Blend<_DesiredType_>(_First, _Second, _Mask);
}

__SIMD_STL_NUMERIC_NAMESPACE_END
