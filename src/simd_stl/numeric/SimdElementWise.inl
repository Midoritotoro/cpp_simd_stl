#pragma once 

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

#pragma region Sse2-Sse4.2 Simd element wise 

template <
    typename    _DesiredType_,
    typename    _VectorType_>
simd_stl_always_inline _VectorType_ _SimdElementWise<arch::CpuFeature::SSE2, xmm128>::_Blend(
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
simd_stl_always_inline _VectorType_ _SimdElementWise<arch::CpuFeature::SSE2, xmm128>::_Blend(
    _VectorType_                        _First,
    _VectorType_                        _Second,
    _Simd_mask_type<_DesiredType_>      _Mask) noexcept
{
    return _Blend<_DesiredType_>(_First, _Second, _SimdToVector<_Generation, _RegisterPolicy, _VectorType_, _DesiredType_>(_Mask));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdElementWise<arch::CpuFeature::SSE2, xmm128>::_Reverse(_VectorType_ _Vector) noexcept {
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

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdElementWise<arch::CpuFeature::SSSE3, xmm128>::_Reverse(_VectorType_ _Vector) noexcept {
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


template <
    typename    _DesiredType_,
    typename    _VectorType_>
simd_stl_always_inline _VectorType_ _SimdElementWise<arch::CpuFeature::SSE41, xmm128>::_Blend(
    _VectorType_ _First,
    _VectorType_ _Second,
    _VectorType_ _Mask) noexcept
{
    return _IntrinBitcast<_VectorType_>(_mm_blendv_epi8(_IntrinBitcast<__m128i>(_Second),
        _IntrinBitcast<__m128i>(_First), _IntrinBitcast<__m128i>(_Mask)));
}

template <
    typename    _DesiredType_,
    typename    _VectorType_>
simd_stl_always_inline _VectorType_ _SimdElementWise<arch::CpuFeature::SSE41, xmm128>::_Blend(
    _VectorType_                        _First,
    _VectorType_                        _Second,
    _Simd_mask_type<_DesiredType_>      _Mask) noexcept
{
    return _Blend<_DesiredType_>(_First, _Second, _SimdToVector<_Generation, _RegisterPolicy, _VectorType_, _DesiredType_>(_Mask));
}

#pragma endregion

#pragma region Avx-Avx2 Simd element wise

template <
    typename    _DesiredType_,
    typename    _VectorType_>
simd_stl_always_inline _VectorType_ _SimdElementWise<arch::CpuFeature::AVX, ymm256>::_Blend(
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
simd_stl_always_inline _VectorType_ _SimdElementWise<arch::CpuFeature::AVX, ymm256>::_Blend(
    _VectorType_                        _First,
    _VectorType_                        _Second,
    _Simd_mask_type<_DesiredType_>      _Mask) noexcept
{
    return _Blend<_DesiredType_>(_First, _Second, _SimdToVector<_Generation, _RegisterPolicy, _VectorType_, _DesiredType_>(_Mask));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdElementWise<arch::CpuFeature::AVX, ymm256>::_Reverse(_VectorType_ _Vector) noexcept {
    if constexpr (sizeof(_DesiredType_) == 8) {
        return _IntrinBitcast<_VectorType_>(_mm256_permute4x64_epi64(_IntrinBitcast<__m256>(_Vector), 0x1B));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        return _IntrinBitcast<_VectorType_>(_mm256_permutevar8x32_epi32(
            _IntrinBitcast<__m256>(_Vector), _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7)));
    }
    else if constexpr (sizeof(_DesiredType_) == 2) {
        const auto _ReverseXmmMask = _mm256_set_epi8(
            1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
            1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);

        const auto _Shuffled = _mm256_permute4x64_epi64(_IntrinBitcast<__m256>(_Vector), 0x4E);
        return _mm256_shuffle_epi8(_Shuffled, _ReverseXmmMask);
    }
    else if constexpr (sizeof(_DesiredType_) == 1) {
        const auto _ReverseXmmMask = _mm256_set_epi8(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

        const auto _Shuffled = _mm256_permute4x64_epi64(_IntrinBitcast<__m256>(_Vector), 0x4E);
        return _mm256_shuffle_epi8(_Shuffled, _ReverseXmmMask);
    }
}

template <
    typename    _DesiredType_,
    typename    _VectorType_>
simd_stl_always_inline _VectorType_ _SimdElementWise<arch::CpuFeature::AVX2, ymm256>::_Blend(
    _VectorType_ _First,
    _VectorType_ _Second,
    _VectorType_ _Mask) noexcept
{
    return _IntrinBitcast<_VectorType_>(_mm256_blendv_epi8(_IntrinBitcast<__m256i>(_Second),
        _IntrinBitcast<__m256i>(_First), _IntrinBitcast<__m256i>(_Mask)));
}

template <
    typename    _DesiredType_,
    typename    _VectorType_>
simd_stl_always_inline _VectorType_ _SimdElementWise<arch::CpuFeature::AVX2, ymm256>::_Blend(
    _VectorType_                        _First,
    _VectorType_                        _Second,
    _Simd_mask_type<_DesiredType_>      _Mask) noexcept
{
    return _Blend<_DesiredType_>(_First, _Second, _SimdToVector<_Generation, _RegisterPolicy, _VectorType_, _DesiredType_>(_Mask));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdElementWise<arch::CpuFeature::AVX2, ymm256>::_Reverse(_VectorType_ _Vector) noexcept {
    if constexpr (sizeof(_DesiredType_) == 8) {
        return _IntrinBitcast<_VectorType_>(_mm256_permute4x64_epi64(_IntrinBitcast<__m256i>(_Vector), 0x1B));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        return _IntrinBitcast<_VectorType_>(_mm256_permutevar8x32_epi32(
            _IntrinBitcast<__m256i>(_Vector), _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7)));
    }
    else if constexpr (sizeof(_DesiredType_) == 2) {
        const auto _ReverseXmmMask = _mm256_set_epi8(
            1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
            1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);

        const auto _Shuffled = _mm256_permute4x64_epi64(_IntrinBitcast<__m256i>(_Vector), 0x4E);
        return _IntrinBitcast<_VectorType_>(_mm256_shuffle_epi8(_Shuffled, _ReverseXmmMask));
    }
    else if constexpr (sizeof(_DesiredType_) == 1) {
        const auto _ReverseXmmMask = _mm256_set_epi8(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

        const auto _Shuffled = _mm256_permute4x64_epi64(_IntrinBitcast<__m256i>(_Vector), 0x4E);
        return _IntrinBitcast<_VectorType_>(_mm256_shuffle_epi8(_Shuffled, _ReverseXmmMask));
    }
}

#pragma endregion

#pragma region Avx512 Simd element wise


template <
    typename    _DesiredType_,
    typename    _VectorType_>
simd_stl_always_inline _VectorType_ _SimdElementWise<arch::CpuFeature::AVX512F, zmm512>::_Blend(
    _VectorType_ _First,
    _VectorType_ _Second,
    _VectorType_ _Mask) noexcept
{
    return _IntrinBitcast<_VectorType_>(_mm512_ternarylogic_epi32(_IntrinBitcast<__m512i>(_Mask),
        _IntrinBitcast<__m512i>(_First), _IntrinBitcast<__m512i>(_Second), 0xCA));
}

template <
    typename    _DesiredType_,
    typename    _VectorType_>
simd_stl_always_inline _VectorType_ _SimdElementWise<arch::CpuFeature::AVX512F, zmm512>::_Blend(
    _VectorType_                        _First,
    _VectorType_                        _Second,
    _Simd_mask_type<_DesiredType_>      _Mask) noexcept
{
    return _Blend<_DesiredType_>(_First, _Second, _SimdToVector<_Generation, _RegisterPolicy, _VectorType_, _DesiredType_>(_Mask));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdElementWise<arch::CpuFeature::AVX512F, zmm512>::_Reverse(_VectorType_ _Vector) noexcept {
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
        const auto _Low = _SimdReverse<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(_IntrinBitcast<__m256i>(_Vector));
        const auto _High = _SimdReverse<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(
            _mm512_extracti64x4_epi64(_IntrinBitcast<__m512i>(_Vector), 1));

        return _IntrinBitcast<_VectorType_>(_mm512_inserti64x4(_IntrinBitcast<__m512i>(_Low), _IntrinBitcast<__m256i>(_High), 1));
    }
}

static constexpr auto _Make8BitMaskExpandTable() noexcept {
    std::array<uint64, 256> _Result;

    for (int i = 0; i < 256; ++i) {
        uint64 _Temp = 0;

        for (int j = 0; j < 8; ++j)
            if (i & (1 << j))
                _Temp |= (uint64(0xFF) << (j * 8));

        _Result[i] = _Temp;
    }

    return _Result;
}

static constexpr auto _Make16BitMaskExpandTable() noexcept {
    std::array<uint32, 256> _Result;

    for (int v = 0; v < 256; ++v) {
        uint32 _Temp = 0;

        for (int i = 0; i < 8; ++i)
            if (v & (1 << i))
                _Temp |= (0xFu << (i * 4));

        _Result[v] = _Temp;
    }

    return _Result;
}

static inline constexpr auto _Mask8BitExpandTableAvx512BW = _Make8BitMaskExpandTable();
static inline constexpr auto _Mask16BitExpandTableAvx512BW = _Make16BitMaskExpandTable();

template <typename _Type_>
simd_stl_always_inline auto _SimdElementWise<arch::CpuFeature::AVX512BW, zmm512>::_ExpandMaskBits(_Type_ _Mask) noexcept {
    if constexpr (sizeof(_Type_) == 1) {
        return _Mask8BitExpandTableAvx512BW[_Mask];
    }
    else if constexpr (sizeof(_Type_) == 2) {
        uint8 _Low = _Mask & 0xFF;
        uint8 _High = _Mask >> 8;

        return ((static_cast<uint64>(_Mask16BitExpandTableAvx512BW[_High]) << 32)) | _Mask16BitExpandTableAvx512BW[_Low];
    }
    else {
        return _Mask;
    }

}

template <
    typename    _DesiredType_,
    typename    _VectorType_>
static simd_stl_always_inline _VectorType_ _SimdElementWise<arch::CpuFeature::AVX512BW, zmm512>::_Blend(
    _VectorType_ _First,
    _VectorType_ _Second,
    _VectorType_ _Mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 2)
        return _IntrinBitcast<_VectorType_>(
            _mm512_ternarylogic_epi32(_IntrinBitcast<__m512i>(_Mask),
                _IntrinBitcast<__m512i>(_First), _IntrinBitcast<__m512i>(_Second), 0xCA));

    else
        return _Blend<_DesiredType_>(_First, _Second, _SimdToMask<_Generation, _RegisterPolicy, _DesiredType_>(_Mask));

}

template <
    typename    _DesiredType_,
    typename    _VectorType_>
static simd_stl_always_inline _VectorType_ _SimdElementWise<arch::CpuFeature::AVX512BW, zmm512>::_Blend(
    _VectorType_                        _First,
    _VectorType_                        _Second,
    _Simd_mask_type<_DesiredType_>      _Mask) noexcept
{
    if constexpr (sizeof(_Simd_mask_type<_DesiredType_>) == 4)
        return _Blend<_DesiredType_>(_First, _Second, _SimdToVector<_Generation, _RegisterPolicy, __m512i, _DesiredType_>(_Mask));

    else
        return _IntrinBitcast<_VectorType_>(_mm512_mask_blend_epi8(
            _ExpandMaskBits(_Mask), _IntrinBitcast<__m512i>(_Second), _IntrinBitcast<__m512i>(_First)));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
static simd_stl_always_inline _VectorType_ _SimdElementWise<arch::CpuFeature::AVX512BW, zmm512>::_Reverse(_VectorType_ _Vector) noexcept {
    if constexpr (sizeof(_DesiredType_) == 2) {
        const auto _Shuffle = _mm512_setr_epi16(
            31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
            15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

        return _IntrinBitcast<_VectorType_>(_mm512_permutexvar_epi16(_Shuffle, _IntrinBitcast<__m512i>(_Vector)));
    }
    else if constexpr (sizeof(_DesiredType_) == 1) {
        const auto _Shuffle = _mm512_setr_epi8(
            63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48,
            47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32,
            31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
            15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

        return _IntrinBitcast<_VectorType_>(_mm512_shuffle_epi8(_IntrinBitcast<__m512i>(_Vector),  _Shuffle));
    }
    else {
        return _SimdReverse<arch::CpuFeature::AVX512F, _RegisterPolicy, _DesiredType_>(_Vector);
    }
}

#pragma endregion

__SIMD_STL_NUMERIC_NAMESPACE_END
