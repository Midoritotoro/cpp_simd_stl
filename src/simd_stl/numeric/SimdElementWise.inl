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
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline std::pair<int32, _VectorType_> __simd_element_wise<arch::CpuFeature::SSE2, xmm128>::__compress(
    _VectorType_    __vector,
    _VectorType_    __mask) noexcept
{
    return __compress<_DesiredType_>(__simd_to_mask<__generation, __register_policy, _DesiredType_>(__vector, __mask));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline std::pair<int32, _VectorType_> __simd_element_wise<arch::CpuFeature::SSE2, xmm128>::__compress(
    _VectorType_                    __vector,
    __simd_mask_type<_DesiredType_> __mask) noexcept
{
    if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_> || __is_pd_v<_DesiredType_>) {
        switch ((__mask & 0x3)) {
            case 0:
                return { 16, __vector };
            case 1:
                __vector = __intrin_bitcast<_VectorType_>(_mm_shuffle_pd(
                    __intrin_bitcast<__m128d>(__vector), __intrin_bitcast<__m128d>(__vector), 0x3));
                return { 8, __vector };

            case 2:
                return { 8, __vector };

            default:
                simd_stl_assert_unreachable();
                break;
        }
    }
    else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_> || __is_ps_v<_DesiredType_>) {
        switch ((__mask & 0xF)) {
            case 0x0:
                return { 16, __vector };
            case 0x1:
                __vector = __intrin_bitcast<_VectorType_>(
                    _mm_shuffle_ps(__intrin_bitcast<__m128>(__vector), __intrin_bitcast<__m128>(__vector), 0xF9));
                return { 12, __vector };
            case 0x2:
                __vector = __intrin_bitcast<_VectorType_>(_mm_shuffle_ps(
                    __intrin_bitcast<__m128>(__vector), __intrin_bitcast<__m128>(__vector), 0xF8));
                return { 12, __vector };
            case 0x3:
                __vector = __intrin_bitcast<_VectorType_>(_mm_shuffle_ps(
                    __intrin_bitcast<__m128>(__vector), __intrin_bitcast<__m128>(__vector), 0xEE));
                return { 8, __vector };
            case 0x4:
                __vector = __intrin_bitcast<_VectorType_>(_mm_shuffle_ps(
                    __intrin_bitcast<__m128>(__vector), __intrin_bitcast<__m128>(__vector), 0xF4));
                return { 12, __vector };
            case 0x5:
                __vector = __intrin_bitcast<_VectorType_>(_mm_shuffle_ps(
                    __intrin_bitcast<__m128>(__vector), __intrin_bitcast<__m128>(__vector), 0xED));
                return { 8, __vector };
            case 0x6:
                __vector = __intrin_bitcast<_VectorType_>(_mm_shuffle_ps(
                    __intrin_bitcast<__m128>(__vector), __intrin_bitcast<__m128>(__vector), 0xEC));
                
                return { 8, __vector };
            case 0x7:
                __vector = __intrin_bitcast<_VectorType_>(_mm_shuffle_ps(
                    __intrin_bitcast<__m128>(__vector), __intrin_bitcast<__m128>(__vector), 0xE7));
                
                return { 4, __vector };
            case 0x8:
                return { 12, __vector };
            case 0x9:
                __vector = __intrin_bitcast<_VectorType_>(_mm_shuffle_ps(
                    __intrin_bitcast<__m128>(__vector), __intrin_bitcast<__m128>(__vector), 0xE9));
                return { 8, __vector };
            case 0xA:
                __vector = __intrin_bitcast<_VectorType_>(_mm_shuffle_ps(
                    __intrin_bitcast<__m128>(__vector), __intrin_bitcast<__m128>(__vector), 0xE8));
                return { 8, __vector };
            case 0xB:
                __vector = __intrin_bitcast<_VectorType_>(_mm_shuffle_ps(
                    __intrin_bitcast<__m128>(__vector), __intrin_bitcast<__m128>(__vector), 0xE6));
                return { 4, __vector };
            case 0xC:
                __vector = __intrin_bitcast<_VectorType_>(_mm_shuffle_ps(
                    __intrin_bitcast<__m128>(__vector), __intrin_bitcast<__m128>(__vector), 0xE5));

                return { 12, __vector };
            case 0xD:
                __vector = __intrin_bitcast<_VectorType_>(_mm_shuffle_ps(
                    __intrin_bitcast<__m128>(__vector), __intrin_bitcast<__m128>(__vector), 0xE5));
                return { 4, __vector };
            case 0xE:
                return { 4, __vector };
            case 0xF:
                return { 0, __vector };
            default:
                simd_stl_assert_unreachable();
        }
    }
    else {
        constexpr auto __length = sizeof(_VectorType_) / sizeof(_DesiredType_);
        _DesiredType_ __source[__length], __result[__length];

        _mm_storeu_si128(reinterpret_cast<__m128i*>(__source), __intrin_bitcast<__m128i>(__vector));

        _DesiredType_* __result_pointer = __result;
        auto __start = __result_pointer;

        for (auto __index = 0; __index < __length; ++__index)
            if (!((__mask >> __index) & 1))
                *__result_pointer++ = __source[__index];

        const auto __processed_size   = (__result_pointer - __start);
        const auto __processed_bytes  = __processed_size * sizeof(_DesiredType_);

        std::memcpy(__result_pointer, __source + __processed_size, sizeof(_VectorType_) - __processed_bytes);
        return { __processed_bytes, __intrin_bitcast<_VectorType_>(_mm_loadu_si128(reinterpret_cast<const __m128i*>(__result))) };
    }
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
simd_stl_always_inline std::pair<int32, _VectorType_> __simd_element_wise<arch::CpuFeature::SSSE3, xmm128>::__compress(
    _VectorType_    __vector,
    _VectorType_    __mask) noexcept
{
    return __compress<_DesiredType_>(__simd_to_mask<__generation, __register_policy, _DesiredType_>(__vector, __mask));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline std::pair<int32, _VectorType_> __simd_element_wise<arch::CpuFeature::SSSE3, xmm128>::__compress(
    _VectorType_                    __vector,
    __simd_mask_type<_DesiredType_> __mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 1) {
        const auto __low_mask   = __mask & 0xFF;
        const auto __high_mask  = (__mask >> 8) & 0xFF;

        const auto __processed_low_bytes    = __tables_sse<sizeof(_DesiredType_)>.__size[__low_mask];
        const auto __processed_high_bytes   = __tables_sse<sizeof(_DesiredType_)>.__size[__high_mask];

        const auto __processed_bytes = __processed_low_bytes + __processed_high_bytes;

        const auto __unprocessed_tail_blend_mask = (1u << (sizeof(_VectorType_) - __processed_bytes)) - 1;
        const auto __unprocessed_tail_vector_blend_mask = __simd_to_vector<__generation, __register_policy, __m128i, _DesiredType_>(__unprocessed_tail_blend_mask);

        // src = 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
        // mask - 10101010_10101010
        // __shuffled_low - 1, 3, 5, 7, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0
        // __shuffled_high - 2, 4, 6, 8, 10, 12, 14, 16, 0, 0, 0, 0, 0, 0, 0, 0

        const auto __shuffle_mask_low   = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(__tables_sse<sizeof(_DesiredType_)>.__shuffle[__low_mask]));
        const auto __shuffle_mask_high  = __intrin_bitcast<__m128i>(_mm_loadh_pd(_mm_setzero_pd(), reinterpret_cast<const double*>(__tables_sse<sizeof(_DesiredType_)>.__shuffle[__high_mask])));

        const auto __unprocessed_elements_low = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(__unprocessed_shuffle_chars_table.__shuffle[__low_mask]));
        const auto __unprocessed_elements_high = __intrin_bitcast<__m128i>(_mm_loadh_pd(_mm_setzero_pd(), reinterpret_cast<const double*>(__unprocessed_shuffle_chars_table.__shuffle[__high_mask])));

        const auto __swapped_halfs = __intrin_bitcast<__m128i>(_mm_movehl_ps(__intrin_bitcast<__m128>(__vector), __intrin_bitcast<__m128>(__vector)));

        const auto __shuffled_low = _mm_shuffle_epi8(__intrin_bitcast<__m128i>(__vector), __shuffle_mask_low);
        const auto __shuffled_high = _mm_shuffle_epi8(__swapped_halfs, __intrin_bitcast<__m128i>(__shuffle_mask_high));

        const auto __shuffled = __intrin_bitcast<__m128i>(_mm_shuffle_pd(
            __intrin_bitcast<__m128d>(__shuffled_high), __intrin_bitcast<__m128d>(__shuffled_low), 1));

        const auto __final = _mm_shuffle_epi8(__shuffled, __unprocessed_tail_vector_blend_mask);

        return { __processed_bytes, __intrin_bitcast<_VectorType_>(__final) };
    }
    else {
        const auto __shuffle_mask = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(__tables_sse<sizeof(_DesiredType_)>.__shuffle[__mask]));
        const auto __processed_bytes = __tables_sse<sizeof(_DesiredType_)>.__size[__mask];

        __vector = __intrin_bitcast<_VectorType_>(_mm_shuffle_epi8(__intrin_bitcast<__m128i>(__vector), __shuffle_mask));
        return { __processed_bytes, __vector };
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
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline std::pair<int32, _VectorType_> __simd_element_wise<arch::CpuFeature::SSE41, xmm128>::__compress(
    _VectorType_    __vector,
    _VectorType_    __mask) noexcept
{
    return __compress<_DesiredType_>(__simd_to_mask<__generation, __register_policy, _DesiredType_>(__vector, __mask));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline std::pair<int32, _VectorType_> __simd_element_wise<arch::CpuFeature::SSE41, xmm128>::__compress(
    _VectorType_                    __vector,
    __simd_mask_type<_DesiredType_> __mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 1) {
        const auto __low_mask   = __mask & 0xFF;
        const auto __high_mask  = (__mask >> 8) & 0xFF;

        const auto __processed_bytes = __tables_sse<sizeof(_DesiredType_)>.__size[__low_mask] + __tables_sse<sizeof(_DesiredType_)>.__size[__high_mask];

        // src = 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
        // mask - 00001010_10101010
        // __shuffled_low - 1, 3, 5, 7, 9, 11, 13, 15, 0, 0, 0, 0, 0, 0, 0, 0
        // __shuffled_high - 2, 4, 6, 8, 10, 12, 14, 16, 0, 0, 0, 0, 0, 0, 0, 0

        const auto __shuffle_mask_low               = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(__tables_sse<sizeof(_DesiredType_)>.__shuffle[__low_mask]));
        const auto __shuffle_mask_high              = _mm_loadh_pd(_mm_setzero_pd(), reinterpret_cast<const double*>(__tables_sse<sizeof(_DesiredType_)>.__shuffle[__high_mask]));

      //  const auto __unprocessed_tail_blend_mask    = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(__tables_sse<sizeof(_DesiredType_)>.__unprocessed_tail[__processed_bytes]));

        const auto __swapped_halfs = __intrin_bitcast<__m128i>(_mm_movehl_ps(
            __intrin_bitcast<__m128>(_mm_slli_si128(__intrin_bitcast<__m128i>(__vector), 8)),
            __intrin_bitcast<__m128>(__vector)));

        const auto __shuffled_low = _mm_shuffle_epi8(__intrin_bitcast<__m128i>(__vector), __shuffle_mask_low);
        const auto __shuffled_high = _mm_shuffle_epi8(__intrin_bitcast<__m128i>(__vector), __swapped_halfs);

        //const auto __blended = __blend<_DesiredType_>(__shuffled_low, __shuffled_high, __unprocessed_tail_blend_mask);

        return { __processed_bytes, __vector };
    }
    else {
        const auto __shuffle_mask = _mm_lddqu_si128(reinterpret_cast<const __m128*>(__tables_sse<sizeof(_DesiredType_)>.__shuffle[__mask]));
        const auto __processed_bytes = __tables_sse<sizeof(_DesiredType_)>.__size[__mask];

        __vector = __intrin_bitcast<_VectorType_>(_mm_shuffle_epi8(__intrin_bitcast<__m128i>(__vector), __shuffle_mask));
        return { __processed_bytes, __vector };
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
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline std::pair<int32, _VectorType_> __simd_element_wise<arch::CpuFeature::AVX2, ymm256>::__compress(
    _VectorType_    __vector,
    _VectorType_    __mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) >= 4) {
        const auto __processed_bytes = __tables_avx<sizeof(_DesiredType_)>.__size[__mask];
        const auto __shuffle = _mm256_cvtepu8_epi32(_mm_loadu_si64(__tables_avx<sizeof(_DesiredType_)>.__shuffle[__mask]));

        __vector = __intrin_bitcast<_VectorType_>(_mm256_permutevar8x32_epi32(__intrin_bitcast<__m256i>(__vector), __shuffle));
        return { __processed_bytes, __vector };
    }
    else {
        using _MaskType = __simd_mask_type<_DesiredType_>;
        using _HalfType = IntegerForSize<__constexpr_max<(sizeof(_DesiredType_) >> 1), 1>()>::Unsigned;

        constexpr auto __maximum    = math::__maximum_integral_limit<_HalfType>();
        constexpr auto __shift      = (sizeof(_MaskType) << 2);

        const auto __low    = __intrin_bitcast<__m128i>(__vector);
        const auto __high   = _mm256_extracti128_si256(__intrin_bitcast<__m256i>(__vector), 1);

        auto __processed = std::pair<int32, _VectorType_>{};

        const auto __compressed_low     = __simd_compress<arch::CpuFeature::SSE42, xmm128, _DesiredType_>((__mask & __maximum), __low);
        const auto __compressed_high    = __simd_compress<arch::CpuFeature::SSE42, xmm128, _DesiredType_>(((__mask >> __shift) & __maximum), __high);

        __processed.first += __compressed_low.first;
        __processed.first += __compressed_high.first;

       // const auto __length = (__address - __start);
       // const auto __store_mask = (1u << (sizeof(_VectorType_) - (__length * sizeof(_DesiredType_)))) - 1;
        return __processed;
       // __mask_store_unaligned<_DesiredType_>(__address, __mask, __vector);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline std::pair<int32, _VectorType_> __simd_element_wise<arch::CpuFeature::AVX2, ymm256>::__compress(
    _VectorType_                    __vector,
    __simd_mask_type<_DesiredType_> __mask) noexcept
{

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

        return __intrin_bitcast<_VectorType_>(_mm512_inserti64x4(__intrin_bitcast<__m512i>(__high),
            __intrin_bitcast<__m256i>(__low), 1));
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
        const auto __shuffle_bytes = _mm512_setr_epi8(
            63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48,
            47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32,
            31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
            15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

        const auto __shuffle_qwords = _mm512_setr_epi64(6, 7, 4, 5, 2, 3, 0, 1);

        const auto __shuffled1 = _mm512_shuffle_epi8(__intrin_bitcast<__m512i>(__vector), __shuffle_bytes);
        return __intrin_bitcast<_VectorType_>(_mm512_permutexvar_epi64(__shuffle_qwords, __shuffled1));
    }
    else {
        return __simd_reverse<arch::CpuFeature::AVX512F, __register_policy, _DesiredType_>(__vector);
    }
}

#pragma endregion

__SIMD_STL_NUMERIC_NAMESPACE_END
