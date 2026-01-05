#pragma once 

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

#pragma region Sse2-Sse4.2 Simd compare


template <
    class               _DesiredType_,
    __simd_comparison   _CompareType_,
    class               _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::SSE2, xmm128>::__simd_mask_type<_DesiredType_>
    __simd_compare_implementation<arch::CpuFeature::SSE2, xmm128>::__mask_compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    return __simd_to_mask<__generation, __register_policy, _DesiredType_>(
        __compare<_DesiredType_, _CompareType_>(__left, __right));
}

template <
    typename            DesiredType_,
    __simd_comparison   _CompareType_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ __simd_compare_implementation<arch::CpuFeature::SSE2, xmm128>::__compare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::equal))
        return __compare_equal<_DesiredType_>(__left, __right);

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::not_equal))
        return __simd_bit_not<__generation, __register_policy>(__compare_equal<_DesiredType_>(__left, __right));

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::less))
        return __compare_less<_DesiredType_>(__left, __right);

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::less_equal))
        return __simd_bit_not<__generation, __register_policy>(__compare_greater<_DesiredType_>(__left, __right));

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::greater))
        return __compare_greater<_DesiredType_>(__left, __right);

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::greater_equal))
        return __simd_bit_not<__generation, __register_policy>(__compare_less<_DesiredType_>(__left, __right));
}

template <
    typename            _DesiredType_,
    __simd_comparison   _CompareType_,
    typename            _VectorType_>
simd_stl_always_inline auto __simd_compare_implementation<arch::CpuFeature::SSE2, xmm128>::__native_compare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    return __compare<_DesiredType_, _CompareType_>(__left, __right);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_compare_implementation<arch::CpuFeature::SSE2, xmm128>::__compare_equal(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>) {
        const auto __equal_mask = _mm_cmpeq_epi32(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));

        const auto __rotated_mask = _mm_shuffle_epi32(__equal_mask, 0xB1);
        const auto __combined_mask = _mm_and_si128(__equal_mask, __rotated_mask);

        return __intrin_bitcast<_VectorType_>(__combined_mask);
    }
    else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_cmpeq_epi32(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else if constexpr (__is_epi16_v<_DesiredType_> || __is_epu16_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_cmpeq_epi16(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else if constexpr (__is_epi8_v<_DesiredType_> || __is_epu8_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_cmpeq_epi8(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else if constexpr (__is_ps_v<_DesiredType_> || __is_ps_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_cmpeq_ps(
            __intrin_bitcast<__m128>(__left), __intrin_bitcast<__m128>(__right)));
    }
    else if constexpr (__is_pd_v<_DesiredType_> || __is_pd_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_cmpeq_pd(
            __intrin_bitcast<__m128d>(__left), __intrin_bitcast<__m128d>(__right)));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_compare_implementation<arch::CpuFeature::SSE2, xmm128>::__compare_less(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (__is_epi64_v<_DesiredType_>) {
        const auto __difference64 = _mm_sub_epi64(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));

        const auto __xor_mask = _mm_xor_si128(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));
        const auto __left_andnot_right = _mm_andnot_si128(__intrin_bitcast<__m128i>(__right), __intrin_bitcast<__m128i>(__left));
        const auto __difference_andnot_xor = _mm_andnot_si128(__xor_mask, __difference64);

        const auto __combined_mask = _mm_or_si128(__left_andnot_right, __difference_andnot_xor);

        const auto __sign_bits32 = _mm_srai_epi32(__combined_mask, 31);
        const auto __sign_bits64 = _mm_shuffle_epi32(__sign_bits32, 0xF5);

        return __intrin_bitcast<_VectorType_>(__sign_bits64);
    }
    else if constexpr (__is_epu64_v<_DesiredType_>) {
        const auto __32bit_sign       = _mm_set1_epi32(0x80000000);

        const auto __signed_32bit_left  = _mm_xor_si128(__intrin_bitcast<__m128i>(__left), __32bit_sign);
        const auto __signed_32bit_right = _mm_xor_si128(__intrin_bitcast<__m128i>(__right), __32bit_sign);
        
        const auto __equal   = _mm_cmpeq_epi32(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));
        const auto __bigger  = _mm_cmplt_epi32(__signed_32bit_left, __signed_32bit_right);

        const auto __shuffled_bigger  = _mm_shuffle_epi32(__bigger, 0xA0);
        const auto __equal_bigger     = _mm_and_si128(__equal, __shuffled_bigger);

        const auto __result = _mm_shuffle_epi32(_mm_or_si128(__bigger, __equal_bigger), 0xF5);
        return __intrin_bitcast<_VectorType_>(__result);
    }
    else if constexpr (__is_epi32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_cmplt_epi32(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else if constexpr (__is_epu32_v<_DesiredType_>) {
        const auto __sign        = _mm_set1_epi32(0x80000000);

        const auto __signed_left  = _mm_xor_si128(__intrin_bitcast<__m128i>(__left), __sign);
        const auto __signed_right = _mm_xor_si128(__intrin_bitcast<__m128i>(__right), __sign);
        
        return __intrin_bitcast<_VectorType_>(_mm_cmplt_epi32(__signed_left, __signed_right));
    }
    else if constexpr (__is_epi16_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_cmplt_epi16(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else if constexpr (__is_epu16_v<_DesiredType_>) {
        const auto __substracted = _mm_subs_epu16(__intrin_bitcast<__m128i>(__right), __intrin_bitcast<__m128i>(__left));
        return __simd_bit_not<__generation, __register_policy>(__intrin_bitcast<_VectorType_>(
            _mm_cmpeq_epi16(__substracted, _mm_setzero_si128())));
    }
    else if constexpr (__is_epi8_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_cmplt_epi8(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else if constexpr (__is_epu8_v<_DesiredType_>) {
        const auto __substracted = _mm_subs_epu8(__intrin_bitcast<__m128i>(__right), __intrin_bitcast<__m128i>(__left));
        return __simd_bit_not<__generation, __register_policy>(__intrin_bitcast<_VectorType_>(
            _mm_cmpeq_epi8(__substracted, _mm_setzero_si128())));
    }
    else if constexpr (__is_ps_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_cmplt_ps(
            __intrin_bitcast<__m128>(__left), __intrin_bitcast<__m128>(__right)));
    }
    else if constexpr (__is_pd_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_cmplt_pd(
            __intrin_bitcast<__m128d>(__left), __intrin_bitcast<__m128d>(__right)));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_compare_implementation<arch::CpuFeature::SSE2, xmm128>::__compare_greater(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    return __compare_less<_DesiredType_>(__right, __left);
}

template <
    class               _DesiredType_,
    __simd_comparison   _CompareType_,
    class               _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::SSSE3, xmm128>::__simd_mask_type<_DesiredType_> 
    __simd_compare_implementation<arch::CpuFeature::SSSE3, xmm128>::__mask_compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    return __simd_to_mask<__generation, __register_policy, _DesiredType_>(
        __compare<_DesiredType_, _CompareType_>(__left, __right));
}


template <
    class               _DesiredType_,
    __simd_comparison   _CompareType_,
    class               _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::SSE41, xmm128>::__simd_mask_type<_DesiredType_>
    __simd_compare_implementation<arch::CpuFeature::SSE41, xmm128>::__mask_compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    return __simd_to_mask<__generation, __register_policy, _DesiredType_>(
        __compare<_DesiredType_, _CompareType_>(__left, __right));
}

template <
    typename            _DesiredType_,
    __simd_comparison   _CompareType_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ __simd_compare_implementation<arch::CpuFeature::SSE41, xmm128>::__compare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::equal))
        return __compare_equal<_DesiredType_>(__left, __right);

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::not_equal))
        return __simd_bit_not<__generation, __register_policy>(__compare_equal<_DesiredType_>(__left, __right));

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::less))
        return __compare_less<_DesiredType_>(__left, __right);

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::less_equal))
        return __simd_bit_not<__generation, __register_policy>(__compare_greater<_DesiredType_>(__left, __right));

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::greater))
        return __compare_greater<_DesiredType_>(__left, __right);

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::greater_equal))
        return __simd_bit_not<__generation, __register_policy>(__compare_less<_DesiredType_>(__left, __right));
}

template <
    typename            _DesiredType_,
    __simd_comparison   _CompareType_,
    typename            _VectorType_>
simd_stl_always_inline auto __simd_compare_implementation<arch::CpuFeature::SSE41, xmm128>::__native_compare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    return __compare<_DesiredType_, _CompareType_>(__left, __right);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_compare_implementation<arch::CpuFeature::SSE41, xmm128>::__compare_equal(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_cmpeq_epi64(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_cmpeq_epi32(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else if constexpr (__is_epi16_v<_DesiredType_> || __is_epu16_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_cmpeq_epi16(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else if constexpr (__is_epi8_v<_DesiredType_> || __is_epu8_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_cmpeq_epi8(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else if constexpr (__is_ps_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_cmpeq_ps(
            __intrin_bitcast<__m128>(__left), __intrin_bitcast<__m128>(__right)));
    }
    else if constexpr (__is_pd_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_cmpeq_pd(
            __intrin_bitcast<__m128d>(__left), __intrin_bitcast<__m128d>(__right)));
    }
}


template <
    class               _DesiredType_,
    __simd_comparison   _CompareType_,
    class               _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::SSE42, xmm128>::__simd_mask_type<_DesiredType_> 
    __simd_compare_implementation<arch::CpuFeature::SSE42, xmm128>::__mask_compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    return __simd_to_mask<__generation, __register_policy, _DesiredType_>(__compare<_DesiredType_, _CompareType_>(__left, __right));
}

template <
    typename            _DesiredType_,
    __simd_comparison   _CompareType_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ __simd_compare_implementation<arch::CpuFeature::SSE42, xmm128>::__compare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::equal))
        return __compare_equal<_DesiredType_>(__left, __right);

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::not_equal))
        return __simd_bit_not<__generation, __register_policy>(__compare_equal<_DesiredType_>(__left, __right));

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::less))
        return __compare_less<_DesiredType_>(__left, __right);

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::less_equal))
        return __simd_bit_not<__generation, __register_policy>(__compare_greater<_DesiredType_>(__left, __right));

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::greater))
        return __compare_greater<_DesiredType_>(__left, __right);

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::greater_equal))
        return __simd_bit_not<__generation, __register_policy>(__compare_less<_DesiredType_>(__left, __right));
}

template <
    typename            _DesiredType_,
    __simd_comparison   _CompareType_,
    typename            _VectorType_>
simd_stl_always_inline auto __simd_compare_implementation<arch::CpuFeature::SSE42, xmm128>::__native_compare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    return __compare<_DesiredType_, _CompareType_>(__left, __right);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_compare_implementation<arch::CpuFeature::SSE42, xmm128>::__compare_greater(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (__is_epi64_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(
            _mm_cmpgt_epi64(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else if constexpr (__is_epu64_v<_DesiredType_>) {
        const auto __sign_64bit   = _mm_set1_epi64x(0x8000000000000000);

        const auto __left_signed  = _mm_xor_si128(__intrin_bitcast<__m128i>(__left), __sign_64bit);
        const auto __right_signed = _mm_xor_si128(__intrin_bitcast<__m128i>(__right), __sign_64bit);

        return __intrin_bitcast<_VectorType_>(_mm_cmpgt_epi64(__left_signed, __right_signed));
    }
    else {
        return __simd_compare<arch::CpuFeature::SSE2, __register_policy, _DesiredType_, __simd_comparison::greater>(__left, __right);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_compare_implementation<arch::CpuFeature::SSE42, xmm128>::__compare_less(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    return __compare_greater<_DesiredType_>(__right, __left);
}

#pragma endregion

#pragma region Avx Simd compare

template <
    class               _DesiredType_,
    __simd_comparison   _CompareType_,
    class               _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX2, ymm256>::__simd_mask_type<_DesiredType_> 
    __simd_compare_implementation<arch::CpuFeature::AVX2, ymm256>::__mask_compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    return __simd_to_mask<__generation, __register_policy, _DesiredType_>(
        __compare<_DesiredType_, _CompareType_>(__left, __right));
}

template <
    typename            _DesiredType_,
    __simd_comparison   _CompareType_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ __simd_compare_implementation<arch::CpuFeature::AVX2, ymm256>::__compare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::equal))
        return __compare_equal<_DesiredType_>(__left, __right);

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::not_equal))
        return __simd_bit_not<__generation, __register_policy>(__compare_equal<_DesiredType_>(__left, __right));

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::less))
        return __compare_less<_DesiredType_>(__left, __right);

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::less_equal))
        return __simd_bit_not<__generation, __register_policy>(__compare_greater<_DesiredType_>(__left, __right));

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::greater))
        return __compare_greater<_DesiredType_>(__left, __right);

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::greater_equal))
        return __simd_bit_not<__generation, __register_policy>(__compare_less<_DesiredType_>(__left, __right));
}

template <
    typename            _DesiredType_,
    __simd_comparison   _CompareType_,
    typename            _VectorType_>
simd_stl_always_inline auto __simd_compare_implementation<arch::CpuFeature::AVX2, ymm256>::__native_compare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    return __compare<_DesiredType_, _CompareType_>(__left, __right);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_compare_implementation<arch::CpuFeature::AVX2, ymm256>::__compare_equal(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (__is_pd_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_cmp_pd(
            __intrin_bitcast<__m256d>(__left), __intrin_bitcast<__m256d>(__right), _CMP_EQ_OQ));

    else if constexpr (__is_ps_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_cmp_ps(
            __intrin_bitcast<__m256>(__left), __intrin_bitcast<__m256>(__right), _CMP_EQ_OQ));

    else if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_cmpeq_epi64(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));

    else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_cmpeq_epi32(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));

    else if constexpr (__is_epi16_v<_DesiredType_> || __is_epu16_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_cmpeq_epi16(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));

    else if constexpr (__is_epi8_v<_DesiredType_> || __is_epu8_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_cmpeq_epi8(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_compare_implementation<arch::CpuFeature::AVX2, ymm256>::__compare_less(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    return __compare_greater<_DesiredType_>(__right, __left);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_compare_implementation<arch::CpuFeature::AVX2, ymm256>::__compare_greater(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (__is_pd_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_cmp_pd(
            __intrin_bitcast<__m256d>(__left), __intrin_bitcast<__m256d>(__right), _MM_CMPINT_GT));
    }
    else if constexpr (__is_ps_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_cmp_ps(
            __intrin_bitcast<__m256>(__left), __intrin_bitcast<__m256>(__right), _MM_CMPINT_GT));
    }
    else if constexpr (__is_epi64_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_cmpgt_epi64(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));
    }
    else if constexpr (__is_epu64_v<_DesiredType_>) {
        const auto __sign_64bit = _mm256_set1_epi64x(0x8000000000000000);

        const auto __left_signed  = _mm256_xor_si256(__intrin_bitcast<__m256i>(__left), __sign_64bit);
        const auto __right_signed = _mm256_xor_si256(__intrin_bitcast<__m256i>(__right), __sign_64bit);

        return __intrin_bitcast<_VectorType_>(_mm256_cmpgt_epi64(__left_signed, __right_signed));
    }
    else if constexpr (__is_epi32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_cmpgt_epi32(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));
    }
    else if constexpr (__is_epu32_v<_DesiredType_>) {
        const auto __sign_32bit = _mm256_set1_epi32(0x80000000);

        const auto __left_signed = _mm256_xor_si256(__intrin_bitcast<__m256i>(__left), __sign_32bit);
        const auto __right_signed = _mm256_xor_si256(__intrin_bitcast<__m256i>(__right), __sign_32bit);

        return __intrin_bitcast<_VectorType_>(_mm256_cmpgt_epi32(__left_signed, __right_signed));
    }
    else if constexpr (__is_epi16_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_cmpgt_epi16(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));
    }
    else if constexpr (__is_epu16_v<_DesiredType_>) {
        const auto __sign_16bit = _mm256_set1_epi16(0x8000);

        const auto __left_signed    = _mm256_xor_si256(__intrin_bitcast<__m256i>(__left), __sign_16bit);
        const auto __right_signed   = _mm256_xor_si256(__intrin_bitcast<__m256i>(__right), __sign_16bit);

        return __intrin_bitcast<_VectorType_>(_mm256_cmpgt_epi16(__left_signed, __right_signed));
    }
    else if constexpr (__is_epi8_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_cmpgt_epi8(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));
    }
    else if constexpr (__is_epu8_v<_DesiredType_>) {
        const auto __sign_8bit   = _mm256_set1_epi8(0x80);

        const auto __left_signed    = _mm256_xor_si256(__intrin_bitcast<__m256i>(__left), __sign_8bit);
        const auto __right_signed   = _mm256_xor_si256(__intrin_bitcast<__m256i>(__right), __sign_8bit);

        return __intrin_bitcast<_VectorType_>(_mm256_cmpgt_epi8(__left_signed, __right_signed));
    }
}

#pragma endregion

#pragma region Avx512 Simd compare


template <
    class               _DesiredType_,
    __simd_comparison   _CompareType_,
    class               _VectorType_>
simd_stl_always_inline _VectorType_ __simd_compare_implementation<arch::CpuFeature::AVX512F, zmm512>::__blockwise_compare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    const auto __compared_low128 = __simd_compare<arch::CpuFeature::SSE42, xmm128, _DesiredType_, _CompareType_>(
        __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));

    const auto __compared2_low128 = __simd_compare<arch::CpuFeature::SSE42, xmm128, _DesiredType_, _CompareType_>(
        _mm256_extractf128_si256(__intrin_bitcast<__m256i>(__left), 1), _mm256_extractf128_si256(__intrin_bitcast<__m256i>(__right), 1));

    const auto __compared_high128 = __simd_compare<arch::CpuFeature::SSE42, xmm128, _DesiredType_, _CompareType_>(
        _mm512_extracti32x4_epi32(__intrin_bitcast<__m512i>(__left), 2), _mm512_extracti32x4_epi32(__intrin_bitcast<__m512i>(__right), 2));

    const auto __compared2_high128 = __simd_compare<arch::CpuFeature::SSE42, xmm128, _DesiredType_, _CompareType_>(
        _mm512_extracti32x4_epi32(__intrin_bitcast<__m512i>(__left), 3), _mm512_extracti32x4_epi32(__intrin_bitcast<__m512i>(__right), 3));

    auto __result = __intrin_bitcast<__m512i>(__compared_low128);

    __result = _mm512_inserti32x4(__result, __compared2_low128, 1);
    __result = _mm512_inserti32x4(__result, __compared_high128, 2);

    return __intrin_bitcast<_VectorType_>(_mm512_inserti32x4(__result, __compared2_high128, 3));
}

template <
    class               _DesiredType_,
    __simd_comparison   _CompareType_,
    class               _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512F, zmm512>::__simd_mask_type<_DesiredType_>
    __simd_compare_implementation<arch::CpuFeature::AVX512F, zmm512>::__mask_compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::equal))
        return __mask_compare_equal<_DesiredType_>(__left, __right);

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::not_equal))
        return ~__mask_compare_equal<_DesiredType_>(__left, __right);

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::less))
        return __mask_compare_less<_DesiredType_>(__left, __right);

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::less_equal))
        return  ~__mask_compare_greater<_DesiredType_>(__left, __right));

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::greater))
        return __mask_compare_greater<_DesiredType_>(__left, __right);

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::greater_equal))
        return ~__mask_compare_less<_DesiredType_>(__left, __right);
}

template <
    typename            _DesiredType_,
    __simd_comparison   _CompareType_,
    typename            _VectorType_>
simd_stl_always_inline auto __simd_compare_implementation<arch::CpuFeature::AVX512F, zmm512>::__native_compare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (sizeof(_DesiredType_) >= 4)
        return __mask_compare<_DesiredType_, _CompareType_>(__left, __right);
    else
        return __compare<_DesiredType_, _CompareType_>(__left, __right);
}

template <
    typename            _DesiredType_,
    __simd_comparison   _CompareType_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ __simd_compare_implementation<arch::CpuFeature::AVX512F, zmm512>::_Compare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (__is_epi16_v<_DesiredType_> || __is_epu16_v<_DesiredType_> ||
        __is_epi8_v<_DesiredType_> || __is_epu8_v<_DesiredType_>)
    {
        return __blockwise_compare<_DesiredType_, _CompareType_>(__left, __right);
    }
    else {
        return __simd_to_vector<__generation, __register_policy, _VectorType_>(
            __mask_compare<_DesiredType_, _CompareType_>(__left, __right));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512F, zmm512>::__simd_mask_type<_DesiredType_>
    __simd_compare_implementation<arch::CpuFeature::AVX512F, zmm512>::__mask_compare_equal(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    if constexpr (__is_pd_v<_DesiredType_>) {
        return _mm512_cmpeq_pd_mask(__intrin_bitcast<__m512d>(__left), __intrin_bitcast<__m512d>(__right));
    }
    else if constexpr (__is_ps_v<_DesiredType_>) {
        return _mm512_cmpeq_ps_mask(__intrin_bitcast<__m512>(__left), __intrin_bitcast<__m512>(__right));
    }
    else if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>) {
        return _mm512_cmpeq_epi64_mask(__intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right));
    }
    else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>) {
        return _mm512_cmpeq_epi32_mask(__intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right));
    }
    else {
        return __simd_to_mask<__generation, __register_policy, _DesiredType_>(
            _BlockwiseCompare<_DesiredType_, __simd_comparison::equal>(__left, __right));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512F, zmm512>::__simd_mask_type<_DesiredType_> 
    __simd_compare_implementation<arch::CpuFeature::AVX512F, zmm512>::__mask_compare_less(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    if constexpr (__is_epi64_v<_DesiredType_>) {
        return _mm512_cmplt_epi64_mask(__intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right));
    }
    else if constexpr (__is_epu64_v<_DesiredType_>) {
        return _mm512_cmplt_epu64_mask(__intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right));
    }
    else if constexpr (__is_epi32_v<_DesiredType_>) {
        return _mm512_cmplt_epi32_mask(__intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right));
    }
    else if constexpr (__is_epu32_v<_DesiredType_>) {
        return _mm512_cmplt_epu32_mask(__intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right));
    }
    else if constexpr (__is_ps_v<_DesiredType_>) {
        return _mm512_cmplt_ps_mask(__intrin_bitcast<__m512>(__left), __intrin_bitcast<__m512>(__right));
    }
    else if constexpr (__is_pd_v<_DesiredType_>) {
        return _mm512_cmplt_pd_mask(__intrin_bitcast<__m512d>(__left), __intrin_bitcast<__m512d>(__right));
    }
    else {
        return __simd_to_mask<__generation, __register_policy, _DesiredType_>(
            __blockwise_compare<_DesiredType_, __simd_comparison::less>(__left, __right));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512F, zmm512>::__simd_mask_type<_DesiredType_> 
    __simd_compare_implementation<arch::CpuFeature::AVX512F, zmm512>::__mask_compare_greater(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    return __mask_compare_less<_DesiredType_>(__right, __left);
}


template <
    class   _DesiredType_,
    __simd_comparison   _CompareType_,
    class   _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512BW, zmm512>::__simd_mask_type<_DesiredType_> 
    __simd_compare_implementation<arch::CpuFeature::AVX512BW, zmm512>::__mask_compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::equal))
        return __mask_compare_equal<_DesiredType_>(__left, __right);

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::not_equal))
        return ~__mask_compare_equal<_DesiredType_>(__left, __right);

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::less))
        return __mask_compare_less<_DesiredType_>(__left, __right);

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::less_equal))
        return  ~__mask_compare_greater<_DesiredType_>(__left, __right));

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::greater))
        return __mask_compare_greater<_DesiredType_>(__left, __right);

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::greater_equal))
        return ~__mask_compare_less<_DesiredType_>(__left, __right);
}

template <
    typename _DesiredType_,
    __simd_comparison    _CompareType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_compare_implementation<arch::CpuFeature::AVX512BW, zmm512>::__compare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    return __simd_to_vector<__generation, __register_policy, _VectorType_>(
        __mask_compare<_DesiredType_, _CompareType_>(__left, __right));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512BW, zmm512>::__simd_mask_type<_DesiredType_> 
    __simd_compare_implementation<arch::CpuFeature::AVX512BW, zmm512>::__mask_compare_equal(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    if constexpr (__is_pd_v<_DesiredType_>)
        return _mm512_cmpeq_pd_mask(__intrin_bitcast<__m512d>(__left), __intrin_bitcast<__m512d>(__right));

    else if constexpr (__is_ps_v<_DesiredType_>)
        return _mm512_cmpeq_ps_mask(__intrin_bitcast<__m512>(__left), __intrin_bitcast<__m512>(__right));

    else if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>)
        return _mm512_cmpeq_epi64_mask(__intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right));

    else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>)
        return _mm512_cmpeq_epi32_mask(__intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right));

    else if constexpr (__is_epi16_v<_DesiredType_> || __is_epu16_v<_DesiredType_>)
        return _mm512_cmpeq_epi16_mask(__intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right));

    else if constexpr (__is_epi8_v<_DesiredType_> || __is_epu8_v<_DesiredType_>)
        return _mm512_cmpeq_epi8_mask(__intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right));
}

template <
    typename _DesiredType_,
    __simd_comparison    _CompareType_,
    typename _VectorType_>
simd_stl_always_inline auto __simd_compare_implementation<arch::CpuFeature::AVX512BW, zmm512>::__native_compare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    return __mask_compare<_DesiredType_, _CompareType_>(__left, __right);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512BW, zmm512>::__simd_mask_type<_DesiredType_> 
    __simd_compare_implementation<arch::CpuFeature::AVX512BW, zmm512>::__mask_compare_less(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    return __mask_compare_greater<_DesiredType_>(__right, __left);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512BW, zmm512>::__simd_mask_type<_DesiredType_> 
    __simd_compare_implementation<arch::CpuFeature::AVX512BW, zmm512>::__mask_compare_greater(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    if constexpr (__is_epi64_v<_DesiredType_>) {
        return _mm512_cmpgt_epi64_mask(__intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right));
    }
    else if constexpr (__is_epu64_v<_DesiredType_>) {
        return _mm512_cmpgt_epu64_mask(__intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right));
    }
    else if constexpr (__is_epi32_v<_DesiredType_>) {
        return _mm512_cmpgt_epi32_mask(__intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right));
    }
    else if constexpr (__is_epu32_v<_DesiredType_>) {
        return _mm512_cmpgt_epu32_mask(__intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right));
    }
    else if constexpr (__is_epi16_v<_DesiredType_>) {
        return _mm512_cmpgt_epi16_mask(__intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right));
    }
    else if constexpr (__is_epu16_v<_DesiredType_>) {
        return _mm512_cmpgt_epu16_mask(__intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right));
    }
    else if constexpr (__is_epi8_v<_DesiredType_>) {
        return _mm512_cmpgt_epi8_mask(__intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right));
    }
    else if constexpr (__is_epu8_v<_DesiredType_>) {
        return _mm512_cmpgt_epu8_mask(__intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right));
    }
    else if constexpr (__is_ps_v<_DesiredType_>) {
        return _mm512_cmplt_ps_mask(__intrin_bitcast<__m512>(__right), __intrin_bitcast<__m512>(__left));
    }
    else if constexpr (__is_pd_v<_DesiredType_>) {
        return _mm512_cmplt_pd_mask(__intrin_bitcast<__m512d>(__right), __intrin_bitcast<__m512d>(__left));
    }
}

template <
    class               _DesiredType_,
    __simd_comparison   _CompareType_,
    class               _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512VLF, ymm256>::__simd_mask_type<_DesiredType_>
    __simd_compare_implementation<arch::CpuFeature::AVX512VLF, ymm256>::__mask_compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::equal))
        return __mask_compare_equal<_DesiredType_>(__left, __right);

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::not_equal))
        return ~__mask_compare_equal<_DesiredType_>(__left, __right);

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::less))
        return __mask_compare_less<_DesiredType_>(__left, __right);

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::less_equal))
        return ~__mask_compare_greater<_DesiredType_>(__left, __right));

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::greater))
        return __mask_compare_greater<_DesiredType_>(__left, __right);

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::greater_equal))
        return ~__mask_compare_less<_DesiredType_>(__left, __right);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512VLF, ymm256>::__simd_mask_type<_DesiredType_> 
    __simd_compare_implementation<arch::CpuFeature::AVX512VLF, ymm256>::__mask_compare_equal(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    if constexpr (__is_pd_v<_DesiredType_>)
        return _mm256_cmp_pd_mask(__intrin_bitcast<__m256d>(__left), __intrin_bitcast<__m256d>(__right), _CMP_EQ_OQ);

    else if constexpr (__is_ps_v<_DesiredType_>)
        return _mm256_cmp_ps_mask(__intrin_bitcast<__m256>(__left), __intrin_bitcast<__m256>(__right), _CMP_EQ_OQ);

    else if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>)
        return _mm256_cmpeq_epi64_mask(__intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right));

    else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>)
        return _mm256_cmpeq_epi32_mask(__intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right));

    else
        return __simd_mask_compare<arch::CpuFeature::AVX2, __register_policy, _DesiredType_, __simd_comparison::equal>(__left, __right);
}

template <
    typename _DesiredType_,
    __simd_comparison    _CompareType_,
    typename _VectorType_>
simd_stl_always_inline auto __simd_compare_implementation<arch::CpuFeature::AVX512VLF, ymm256>::__native_compare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (sizeof(_DesiredType_) >= 4)
        return __mask_compare<_DesiredType_, _CompareType_>(__left, __right);
    else
        return __compare<_DesiredType_, _CompareType_>(__left, __right);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512VLF, ymm256>::__simd_mask_type<_DesiredType_> 
    __simd_compare_implementation<arch::CpuFeature::AVX512VLF, ymm256>::__mask_compare_less(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    return __mask_compare_greater<_DesiredType_>(__right, __left);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512VLF, ymm256>::__simd_mask_type<_DesiredType_> 
    __simd_compare_implementation<arch::CpuFeature::AVX512VLF, ymm256>::__mask_compare_greater(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    if constexpr (__is_epi64_v<_DesiredType_>) {
        return _mm256_cmpgt_epi64_mask(__intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right));
    }
    else if constexpr (__is_epu64_v<_DesiredType_>) {
        return _mm256_cmpgt_epu64_mask(__intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right));
    }
    else if constexpr (__is_epi32_v<_DesiredType_>) {
        return _mm256_cmpgt_epi32_mask(__intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right));
    }
    else if constexpr (__is_epu32_v<_DesiredType_>) {
        return _mm256_cmpgt_epu32_mask(__intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right));
    }
    else if constexpr (__is_ps_v<_DesiredType_>) {
        return _mm256_cmp_ps_mask(__intrin_bitcast<__m256>(__left), __intrin_bitcast<__m256>(__right), _CMP_GT_OQ);
    }
    else if constexpr (__is_pd_v<_DesiredType_>) {
        return _mm256_cmp_pd_mask(__intrin_bitcast<__m256d>(__left), __intrin_bitcast<__m256d>(__right), _CMP_GT_OQ);
    }
    else {
        return __simd_mask_compare<arch::CpuFeature::AVX2, __register_policy, _DesiredType_, __simd_comparison::greater>(__left, __right);
    }
}

template <
    class               _DesiredType_,
    __simd_comparison   _CompareType_,
    class               _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, ymm256>::__simd_mask_type<_DesiredType_> 
    __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, ymm256>::__mask_compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::equal))
        return __mask_compare_equal<_DesiredType_>(__left, __right);

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::not_equal))
        return ~__mask_compare_equal<_DesiredType_>(__left, __right);

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::less))
        return __mask_compare_less<_DesiredType_>(__left, __right);

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::less_equal))
        return  ~__mask_compare_greater<_DesiredType_>(__left, __right));

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::greater))
        return __mask_compare_greater<_DesiredType_>(__left, __right);

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::greater_equal))
        return ~__mask_compare_less<_DesiredType_>(__left, __right);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, ymm256>::__simd_mask_type<_DesiredType_> 
    __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, ymm256>::__mask_compare_equal(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    if constexpr (__is_epi16_v<_DesiredType_>) {
        return _mm256_cmpeq_epi16_mask(__intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right));
    }
    else if constexpr (__is_epu16_v<_DesiredType_>) {
        return _mm256_cmpeq_epu16_mask(__intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right));
    }
    else if constexpr (__is_epi8_v<_DesiredType_>) {
        return _mm256_cmpeq_epi8_mask(__intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right));
    }
    else if constexpr (__is_epu8_v<_DesiredType_>) {
        return _mm256_cmpeq_epu8_mask(__intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right));
    }
    else {
        return __simd_mask_compare<arch::CpuFeature::AVX512VLF, __register_policy, _DesiredType_ __simd_comparison::equal>(__left, __right);
    }
}

template <
    typename _DesiredType_,
    __simd_comparison    _CompareType_,
    typename _VectorType_>
simd_stl_always_inline auto __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, ymm256>::__native_compare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    return __mask_compare<_DesiredType_, _CompareType_>(__left, __right);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, ymm256>::__simd_mask_type<_DesiredType_> 
    __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, ymm256>::__mask_compare_less(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    return __mask_compare_greater<_DesiredType_>(__right, __left);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, ymm256>::__simd_mask_type<_DesiredType_> 
    __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, ymm256>::__mask_compare_greater(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    if constexpr (__is_epi16_v<_DesiredType_>) {
        return _mm256_cmpgt_epi16_mask(__intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right));
    }
    else if constexpr (__is_epu16_v<_DesiredType_>) {
        return _mm256_cmpgt_epu16_mask(__intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right));
    }
    else if constexpr (__is_epi8_v<_DesiredType_>) {
        return _mm256_cmpgt_epi8_mask(__intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right));
    }
    else if constexpr (__is_epu8_v<_DesiredType_>) {
        return _mm256_cmpgt_epu8_mask(__intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right));
    }
    else {
        return __simd_mask_compare<arch::CpuFeature::AVX512VLF, __register_policy, _DesiredType_, __simd_comparison::greater>(__left, __right);
    }
}

template <
    class   _DesiredType_,
    __simd_comparison   _CompareType_,
    class   _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512VLF, xmm128>::__simd_mask_type<_DesiredType_>
    __simd_compare_implementation<arch::CpuFeature::AVX512VLF, xmm128>::__mask_compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::equal))
        return __mask_compare_equal<_DesiredType_>(__left, __right);

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::not_equal))
        return ~__mask_compare_equal<_DesiredType_>(__left, __right);

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::less))
        return __mask_compare_less<_DesiredType_>(__left, __right);

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::less_equal))
        return  ~__mask_compare_greater<_DesiredType_>(__left, __right));

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::greater))
        return __mask_compare_greater<_DesiredType_>(__left, __right);

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::greater_equal))
        return ~__mask_compare_less<_DesiredType_>(__left, __right);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512VLF, xmm128>::__simd_mask_type<_DesiredType_>
    __simd_compare_implementation<arch::CpuFeature::AVX512VLF, xmm128>::__mask_compare_equal(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    if constexpr (__is_pd_v<_DesiredType_>)
        return _mm_cmp_pd_mask(__intrin_bitcast<__m128d>(__left), __intrin_bitcast<__m128d>(__right), _CMP_EQ_OQ);

    else if constexpr (__is_ps_v<_DesiredType_>)
        return _mm_cmp_ps_mask(__intrin_bitcast<__m128>(__left), __intrin_bitcast<__m128>(__right), _CMP_EQ_OQ);

    else if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>)
        return _mm_cmpeq_epi64_mask(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));

    else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>)
        return _mm_cmpeq_epi32_mask(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));

    else
        return __simd_mask_compare<arch::CpuFeature::SSE42, __register_policy, _DesiredType_, __simd_comparison::equal>(__left, __right);
}

template <
    typename            _DesiredType_,
    __simd_comparison   _CompareType_,
    typename            _VectorType_>
simd_stl_always_inline auto __simd_compare_implementation<arch::CpuFeature::AVX512VLF, xmm128>::__native_compare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (sizeof(_DesiredType_) >= 4)
        return __mask_compare<_DesiredType_, _CompareType_>(__left, __right);
    else
        return __compare<_DesiredType_, _CompareType_>(__left, __right);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512VLF, xmm128>::__simd_mask_type<_DesiredType_>
    __simd_compare_implementation<arch::CpuFeature::AVX512VLF, xmm128>::__mask_compare_less(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    return __mask_compare_greater<_DesiredType_>(__right, __left);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512VLF, xmm128>::__simd_mask_type<_DesiredType_>
    __simd_compare_implementation<arch::CpuFeature::AVX512VLF, xmm128>::__mask_compare_greater(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    if constexpr (__is_epi64_v<_DesiredType_>) {
        return _mm_cmpgt_epi64_mask(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));
    }
    else if constexpr (__is_epu64_v<_DesiredType_>) {
        return _mm_cmpgt_epu64_mask(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));
    }
    else if constexpr (__is_epi32_v<_DesiredType_>) {
        return _mm_cmpgt_epi32_mask(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));
    }
    else if constexpr (__is_epu32_v<_DesiredType_>) {
        return _mm_cmpgt_epu32_mask(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));
    }
    else if constexpr (__is_ps_v<_DesiredType_>) {
        return _mm_cmp_ps_mask(__intrin_bitcast<__m128>(__left), __intrin_bitcast<__m128>(__right), _CMP_GT_OQ);
    }
    else if constexpr (__is_pd_v<_DesiredType_>) {
        return _mm_cmp_pd_mask(__intrin_bitcast<__m128d>(__left), __intrin_bitcast<__m128d>(__right), _CMP_GT_OQ);
    }
    else {
        return _SimdMaskCompare<arch::CpuFeature::SSE42, __register_policy, _DesiredType_, type_traits::greater<>>(__left, __right);
    }
}

template <
    class   _DesiredType_,
    __simd_comparison   _CompareType_,
    class   _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, xmm128>::__simd_mask_type<_DesiredType_>
    __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, xmm128>::__mask_compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::equal))
        return __mask_compare_equal<_DesiredType_>(__left, __right);

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::not_equal))
        return ~__mask_compare_equal<_DesiredType_>(__left, __right);

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::less))
        return __mask_compare_less<_DesiredType_>(__left, __right);

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::less_equal))
        return  ~__mask_compare_greater<_DesiredType_>(__left, __right));

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::greater))
        return __mask_compare_greater<_DesiredType_>(__left, __right);

    else if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::greater_equal))
        return ~__mask_compare_less<_DesiredType_>(__left, __right);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, xmm128>::__simd_mask_type<_DesiredType_>
    __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, xmm128>::__mask_compare_equal(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    if constexpr (__is_epi16_v<_DesiredType_>) {
        return _mm_cmpeq_epi16_mask(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));
    }
    else if constexpr (__is_epu16_v<_DesiredType_>) {
        return _mm_cmpeq_epu16_mask(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));
    }
    else if constexpr (__is_epi8_v<_DesiredType_>) {
        return _mm_cmpeq_epi8_mask(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));
    }
    else if constexpr (__is_epu8_v<_DesiredType_>) {
        return _mm_cmpeq_epu8_mask(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));
    }
    else {
        return _SimdMaskCompare<arch::CpuFeature::AVX512VLF, __register_policy, _DesiredType_, type_traits::equal_to<>>(__left, __right);
    }
}

template <
    typename _DesiredType_,
    __simd_comparison    _CompareType_,
    typename _VectorType_>
simd_stl_always_inline auto __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, xmm128>::__native_compare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    return __mask_compare<_DesiredType_, _CompareType_>(__left, __right);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, xmm128>::__simd_mask_type<_DesiredType_>
    __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, xmm128>::__mask_compare_less(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    return __mask_compare_greater<_DesiredType_>(__right, __left);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, xmm128>::__simd_mask_type<_DesiredType_>
    __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, xmm128>::__mask_compare_greater(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    if constexpr (__is_epi16_v<_DesiredType_>) {
        return _mm_cmpgt_epi16_mask(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));
    }
    else if constexpr (__is_epu16_v<_DesiredType_>) {
        return _mm_cmpgt_epu16_mask(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));
    }
    else if constexpr (__is_epi8_v<_DesiredType_>) {
        return _mm_cmpgt_epi8_mask(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));
    }
    else if constexpr (__is_epu8_v<_DesiredType_>) {
        return _mm_cmpgt_epu8_mask(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));
    }
    else {
        return __simd_mask_compare<arch::CpuFeature::AVX512VLF, __register_policy, _DesiredType_, __simd_comparison::greater>(__left, __right);
    }
}


#pragma endregion 

__SIMD_STL_NUMERIC_NAMESPACE_END
