#pragma once 

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

#pragma region Sse2-Sse4.2 Simd element wise 

template <
    typename    _DesiredType_,
    typename    _VectorType_>
simd_stl_always_inline _VectorType_ __simd_element_wise<arch::CpuFeature::SSE2, xmm128>::__blend(
    _VectorType_ __first,
    _VectorType_ __second,
    _VectorType_ __mask) noexcept
{
    return __intrin_bitcast<_VectorType_>(_mm_or_si128(
        _mm_and_si128(__intrin_bitcast<__m128i>(__mask), __intrin_bitcast<__m128i>(__first)),
        _mm_andnot_si128(__intrin_bitcast<__m128i>(__mask), __intrin_bitcast<__m128i>(__second))));
}

template <
    typename    _DesiredType_,
    typename    _VectorType_>
simd_stl_always_inline _VectorType_ __simd_element_wise<arch::CpuFeature::SSE2, xmm128>::__blend(
    _VectorType_                        __first,
    _VectorType_                        __second,
    __simd_mask_type<_DesiredType_>     __mask) noexcept
{
    return __blend<_DesiredType_>(__first, __second, 
        __simd_to_vector<__generation, __register_policy, _VectorType_, _DesiredType_>(__mask));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_element_wise<arch::CpuFeature::SSE2, xmm128>::__reverse(_VectorType_ __vector) noexcept {
    if constexpr (sizeof(_DesiredType_) == 8) {
        return __intrin_bitcast<_VectorType_>(_mm_shuffle_pd(
            __intrin_bitcast<__m128d>(__vector), __intrin_bitcast<__m128d>(__vector), 1));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        return __intrin_bitcast<_VectorType_>(_mm_shuffle_epi32(__intrin_bitcast<__m128i>(__vector), 0x1B));
    }
    else if constexpr (sizeof(_DesiredType_) == 2) {
        __vector = __intrin_bitcast<_VectorType_>(_mm_shuffle_pd(
            __intrin_bitcast<__m128d>(__vector), __intrin_bitcast<__m128d>(__vector), 1));

        __vector = __intrin_bitcast<_VectorType_>(_mm_shufflehi_epi16(__intrin_bitcast<__m128i>(__vector), 0x1B));
        return __intrin_bitcast<_VectorType_>(_mm_shufflelo_epi16(__intrin_bitcast<__m128i>(__vector), 0x1B));
    }
    else if constexpr (sizeof(_DesiredType_) == 1) {
        __vector = __intrin_bitcast<_VectorType_>(_mm_or_si128(
            _mm_srli_epi16(__intrin_bitcast<__m128i>(__vector), 8),
            _mm_slli_epi16(__intrin_bitcast<__m128i>(__vector), 8)));

        __vector = __intrin_bitcast<_VectorType_>(_mm_shufflelo_epi16(__intrin_bitcast<__m128i>(__vector), 0x1B));
        __vector = __intrin_bitcast<_VectorType_>(_mm_shufflehi_epi16(__intrin_bitcast<__m128i>(__vector), 0x1B));

        return __intrin_bitcast<_VectorType_>(_mm_shuffle_epi32(__intrin_bitcast<__m128i>(__vector), 0x4E));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_element_wise<arch::CpuFeature::SSSE3, xmm128>::__reverse(_VectorType_ __vector) noexcept {
    if constexpr (sizeof(_DesiredType_) == 8) {
        return __intrin_bitcast<_VectorType_>(_mm_shuffle_pd(
            __intrin_bitcast<__m128d>(__vector), __intrin_bitcast<__m128d>(__vector), 1));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        return __intrin_bitcast<_VectorType_>(_mm_shuffle_epi32(__intrin_bitcast<__m128i>(__vector), 0x1B));
    }
    else if constexpr (sizeof(_DesiredType_) == 2) {
        __vector = __intrin_bitcast<_VectorType_>(_mm_shuffle_pd(
            __intrin_bitcast<__m128d>(__vector), __intrin_bitcast<__m128d>(__vector), 1));

        __vector = __intrin_bitcast<_VectorType_>(_mm_shufflehi_epi16(__intrin_bitcast<__m128i>(__vector), 0x1B));
        return __intrin_bitcast<_VectorType_>(_mm_shufflelo_epi16(__intrin_bitcast<__m128i>(__vector), 0x1B));
    }
    else if constexpr (sizeof(_DesiredType_) == 1) {
        return __intrin_bitcast<_VectorType_>(_mm_shuffle_epi8(__intrin_bitcast<__m128i>(__vector),
            _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)));
    }
}


template <
    typename    _DesiredType_,
    typename    _VectorType_>
simd_stl_always_inline _VectorType_ __simd_element_wise<arch::CpuFeature::SSE41, xmm128>::__blend(
    _VectorType_ __first,
    _VectorType_ __second,
    _VectorType_ __mask) noexcept
{
    return __intrin_bitcast<_VectorType_>(_mm_blendv_epi8(__intrin_bitcast<__m128i>(__second),
        __intrin_bitcast<__m128i>(__first), __intrin_bitcast<__m128i>(__mask)));
}

template <
    typename    _DesiredType_,
    typename    _VectorType_>
simd_stl_always_inline _VectorType_ __simd_element_wise<arch::CpuFeature::SSE41, xmm128>::__blend(
    _VectorType_                        __first,
    _VectorType_                        __second,
    __simd_mask_type<_DesiredType_>     __mask) noexcept
{
    return __blend<_DesiredType_>(__first, __second, 
        __simd_to_vector<__generation, __register_policy, _VectorType_, _DesiredType_>(__mask));
}

#pragma endregion

#pragma region Avx-Avx2 Simd element wise

template <
    typename    _DesiredType_,
    typename    _VectorType_>
simd_stl_always_inline _VectorType_ __simd_element_wise<arch::CpuFeature::AVX, ymm256>::__blend(
    _VectorType_ __first,
    _VectorType_ __second,
    _VectorType_ __mask) noexcept
{
    return __intrin_bitcast<_VectorType_>(_mm256_or_ps(
        _mm256_and_ps(__intrin_bitcast<__m256>(__mask), __intrin_bitcast<__m256>(__first)),
        _mm256_andnot_ps(__intrin_bitcast<__m256>(__mask), __intrin_bitcast<__m256>(__second))));
}

template <
    typename    _DesiredType_,
    typename    _VectorType_>
simd_stl_always_inline _VectorType_ __simd_element_wise<arch::CpuFeature::AVX, ymm256>::__blend(
    _VectorType_                        __first,
    _VectorType_                        __second,
    __simd_mask_type<_DesiredType_>     __mask) noexcept
{
    return __blend<_DesiredType_>(__first, __second,
        __simd_to_vector<__generation, __register_policy, _VectorType_, _DesiredType_>(__mask));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_element_wise<arch::CpuFeature::AVX, ymm256>::__reverse(_VectorType_ __vector) noexcept {
    if constexpr (sizeof(_DesiredType_) == 8) {
        return __intrin_bitcast<_VectorType_>(_mm256_permute4x64_epi64(__intrin_bitcast<__m256>(__vector), 0x1B));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        return __intrin_bitcast<_VectorType_>(_mm256_permutevar8x32_epi32(
            __intrin_bitcast<__m256>(__vector), _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7)));
    }
    else if constexpr (sizeof(_DesiredType_) == 2) {
        const auto __reverse_mask = _mm256_set_epi8(
            1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
            1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);

        const auto __shuffled = _mm256_permute4x64_epi64(__intrin_bitcast<__m256>(__vector), 0x4E);
        return _mm256_shuffle_epi8(__shuffled, __reverse_mask);
    }
    else if constexpr (sizeof(_DesiredType_) == 1) {
        const auto __reverse_mask = _mm256_set_epi8(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

        const auto __shuffled = _mm256_permute4x64_epi64(__intrin_bitcast<__m256>(__vector), 0x4E);
        return _mm256_shuffle_epi8(__shuffled, __reverse_mask);
    }
}

template <
    typename    _DesiredType_,
    typename    _VectorType_>
simd_stl_always_inline _VectorType_ __simd_element_wise<arch::CpuFeature::AVX2, ymm256>::__blend(
    _VectorType_ __first,
    _VectorType_ __second,
    _VectorType_ __mask) noexcept
{
    return __intrin_bitcast<_VectorType_>(_mm256_blendv_epi8(__intrin_bitcast<__m256i>(__second),
        __intrin_bitcast<__m256i>(__first), __intrin_bitcast<__m256i>(__mask)));
}

template <
    typename    _DesiredType_,
    typename    _VectorType_>
simd_stl_always_inline _VectorType_ __simd_element_wise<arch::CpuFeature::AVX2, ymm256>::__blend(
    _VectorType_                        __first,
    _VectorType_                        __second,
    __simd_mask_type<_DesiredType_>     __mask) noexcept
{
    return __blend<_DesiredType_>(__first, __second, 
        __simd_to_vector<__generation, __register_policy, _VectorType_, _DesiredType_>(__mask));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_element_wise<arch::CpuFeature::AVX2, ymm256>::__reverse(_VectorType_ __vector) noexcept {
    if constexpr (sizeof(_DesiredType_) == 8) {
        return __intrin_bitcast<_VectorType_>(_mm256_permute4x64_epi64(__intrin_bitcast<__m256i>(__vector), 0x1B));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        return __intrin_bitcast<_VectorType_>(_mm256_permutevar8x32_epi32(
            __intrin_bitcast<__m256i>(__vector), _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7)));
    }
    else if constexpr (sizeof(_DesiredType_) == 2) {
        const auto __reverse_mask = _mm256_set_epi8(
            1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
            1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);

        const auto __shuffled = _mm256_permute4x64_epi64(__intrin_bitcast<__m256i>(__vector), 0x4E);
        return __intrin_bitcast<_VectorType_>(_mm256_shuffle_epi8(__shuffled, __reverse_mask));
    }
    else if constexpr (sizeof(_DesiredType_) == 1) {
        const auto __reverse_mask = _mm256_set_epi8(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

        const auto __shuffled = _mm256_permute4x64_epi64(__intrin_bitcast<__m256i>(__vector), 0x4E);
        return __intrin_bitcast<_VectorType_>(_mm256_shuffle_epi8(__shuffled, __reverse_mask));
    }
}

#pragma endregion

#pragma region Avx512 Simd element wise


template <
    typename    _DesiredType_,
    typename    _VectorType_>
simd_stl_always_inline _VectorType_ __simd_element_wise<arch::CpuFeature::AVX512F, zmm512>::__blend(
    _VectorType_ __first,
    _VectorType_ __second,
    _VectorType_ __mask) noexcept
{
    return __intrin_bitcast<_VectorType_>(_mm512_ternarylogic_epi32(__intrin_bitcast<__m512i>(__mask),
        __intrin_bitcast<__m512i>(__first), __intrin_bitcast<__m512i>(__second), 0xCA));
}

template <
    typename    _DesiredType_,
    typename    _VectorType_>
simd_stl_always_inline _VectorType_ __simd_element_wise<arch::CpuFeature::AVX512F, zmm512>::__blend(
    _VectorType_                        __first,
    _VectorType_                        __second,
    __simd_mask_type<_DesiredType_>     __mask) noexcept
{
    return __blend<_DesiredType_>(__first, __second, 
        __simd_to_vector<__generation, __register_policy, _VectorType_, _DesiredType_>(__mask));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_element_wise<arch::CpuFeature::AVX512F, zmm512>::__reverse(_VectorType_ __vector) noexcept {
    if constexpr (sizeof(_DesiredType_) == 8) {
        return __intrin_bitcast<_VectorType_>(_mm512_permutexvar_epi64(
            _mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0), __intrin_bitcast<__m512i>(__vector)));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        return __intrin_bitcast<_VectorType_>(_mm512_permutexvar_epi32(
            _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0), __intrin_bitcast<__m512i>(__vector)));
    }
    else {
        const auto __low = __simd_reverse<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(__intrin_bitcast<__m256i>(__vector));
        const auto __high = __simd_reverse<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(
            _mm512_extracti64x4_epi64(__intrin_bitcast<__m512i>(__vector), 1));

        return __intrin_bitcast<_VectorType_>(_mm512_inserti64x4(__intrin_bitcast<__m512i>(__low),
            __intrin_bitcast<__m256i>(__high), 1));
    }
}

template <
    typename    _DesiredType_,
    typename    _VectorType_>
static simd_stl_always_inline _VectorType_ __simd_element_wise<arch::CpuFeature::AVX512BW, zmm512>::__blend(
    _VectorType_ __first,
    _VectorType_ __second,
    _VectorType_ __mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 2)
        return __intrin_bitcast<_VectorType_>(
            _mm512_ternarylogic_epi32(__intrin_bitcast<__m512i>(__mask),
                __intrin_bitcast<__m512i>(__first), __intrin_bitcast<__m512i>(__second), 0xCA));

    else
        return __blend<_DesiredType_>(__first, __second,
            __simd_to_mask<__generation, __register_policy, _DesiredType_>(__mask));

}

template <
    typename    _DesiredType_,
    typename    _VectorType_>
static simd_stl_always_inline _VectorType_ __simd_element_wise<arch::CpuFeature::AVX512BW, zmm512>::__blend(
    _VectorType_                        __first,
    _VectorType_                        __second,
    __simd_mask_type<_DesiredType_>     __mask) noexcept
{
    if constexpr (sizeof(__simd_mask_type<_DesiredType_>) == 4)
        return __blend<_DesiredType_>(__first, __second, 
            __simd_to_vector<__generation, __register_policy, __m512i, _DesiredType_>(__mask));

    else
        return __intrin_bitcast<_VectorType_>(_mm512_mask_blend_epi8(
            __expand_mask_bits_zmm(__mask), __intrin_bitcast<__m512i>(__second),
            __intrin_bitcast<__m512i>(__first)));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
static simd_stl_always_inline _VectorType_ __simd_element_wise<arch::CpuFeature::AVX512BW, zmm512>::__reverse(_VectorType_ __vector) noexcept {
    if constexpr (sizeof(_DesiredType_) == 2) {
        const auto __shuffle = _mm512_setr_epi16(
            31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
            15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

        return __intrin_bitcast<_VectorType_>(_mm512_permutexvar_epi16(__shuffle, __intrin_bitcast<__m512i>(__vector)));
    }
    else if constexpr (sizeof(_DesiredType_) == 1) {
        const auto __shuffle = _mm512_setr_epi8(
            63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48,
            47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32,
            31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
            15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

        return __intrin_bitcast<_VectorType_>(_mm512_shuffle_epi8(__intrin_bitcast<__m512i>(__vector), __shuffle));
    }
    else {
        return __simd_reverse<arch::CpuFeature::AVX512F, __register_policy, _DesiredType_>(__vector);
    }
}

#pragma endregion

__SIMD_STL_NUMERIC_NAMESPACE_END
