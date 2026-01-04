#pragma once 

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

#pragma region Sse2-Sse4.2 Simd compare


template <
    class               _DesiredType_,
    __simd_comparison  _CompareType_,
    class               _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::SSE2, xmm128>::__simd_mask_type<_DesiredType_>
    __simd_compare_implementation<arch::CpuFeature::SSE2, xmm128>::__mask_compare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    return __simd_to_mask<__generation, __register_policy, _DesiredType_>(__compare<_DesiredType_, _CompareType_>(__left, __right));
}

template <
    typename            DesiredType_,
    __simd_comparison   _CompareType_,
    typename            _VectorType_>
simd_stl_always_inline _VectorType_ __simd_compare_implementation<arch::CpuFeature::SSE2, xmm128>::_Compare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (static_cast<int>(_CompareType_) == static_cast<int>(__simd_comparison::equal))
        return _CompareEqual<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::not_equal_to<>>)
        return __simd_bit_not<_Generation, __register_policy>(_CompareEqual<_DesiredType_>(__left, __right));

    else if constexpr (std::is_same_v<_CompareType_, type_traits::less<>>)
        return _CompareLess<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::less_equal<>>)
        return __simd_bit_not<_Generation, __register_policy>(_CompareGreater<_DesiredType_>(__left, __right));

    else if constexpr (std::is_same_v<_CompareType_, type_traits::greater<>>)
        return _CompareGreater<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::greater_equal<>>)
        return __simd_bit_not<_Generation, __register_policy>(_CompareLess<_DesiredType_>(__left, __right));
}

template <
    typename _DesiredType_,
    class    _CompareType_,
    typename _VectorType_>
simd_stl_always_inline auto __simd_compare_implementation<arch::CpuFeature::SSE2, xmm128>::_NativeCompare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    return _Compare<_DesiredType_, _CompareType_>(__left, __right);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_compare_implementation<arch::CpuFeature::SSE2, xmm128>::_CompareEqual(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>) {
        const auto _EqualMask = _mm_cmpeq_epi32(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));

        const auto _RotatedMask = _mm_shuffle_epi32(_EqualMask, 0xB1);
        const auto _CombinedMask = _mm_and_si128(_EqualMask, _RotatedMask);

        return __intrin_bitcast<_VectorType_>(_CombinedMask);
    }
    else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_cmpeq_epi32(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_cmpeq_epi16(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_cmpeq_epi8(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else if constexpr (_Is_ps_v<_DesiredType_> || _Is_ps_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_cmpeq_ps(
            __intrin_bitcast<__m128>(__left), __intrin_bitcast<__m128>(__right)));
    }
    else if constexpr (_Is_pd_v<_DesiredType_> || _Is_pd_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_cmpeq_pd(
            __intrin_bitcast<__m128d>(__left), __intrin_bitcast<__m128d>(__right)));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_compare_implementation<arch::CpuFeature::SSE2, xmm128>::_CompareLess(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (_Is_epi64_v<_DesiredType_>) {
        const auto __leftToInteger = __intrin_bitcast<__m128i>(__left);
        const auto __rightToInteger = __intrin_bitcast<__m128i>(__right);

        const auto _Difference64 = _mm_sub_epi64(__leftToInteger, __rightToInteger);

        const auto _XorMask = _mm_xor_si128(__leftToInteger, __rightToInteger);
        const auto __leftAndNotRight = _mm_andnot_si128(__rightToInteger, __leftToInteger);
        const auto _DifferenceAndNotXor = _mm_andnot_si128(_XorMask, _Difference64);

        const auto _CombinedMask = _mm_or_si128(__leftAndNotRight, _DifferenceAndNotXor);

        const auto _SignBits32 = _mm_srai_epi32(_CombinedMask, 31);
        const auto _SignBits64 = _mm_shuffle_epi32(_SignBits32, 0xF5);

        return __intrin_bitcast<_VectorType_>(_SignBits64);
    }
    else if constexpr (_Is_epu64_v<_DesiredType_>) {
        const auto _32BitSign       = _mm_set1_epi32(0x80000000);

        const auto _Signed32BitLeft     = _mm_xor_si128(__intrin_bitcast<__m128i>(__left), _32BitSign);
        const auto _Signed32BitRight    = _mm_xor_si128(__intrin_bitcast<__m128i>(__right), _32BitSign);
        
        const auto _Equal   = _mm_cmpeq_epi32(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));
        const auto _Bigger  = _mm_cmplt_epi32(_Signed32BitLeft, _Signed32BitRight);

        const auto _ShuffledBigger  = _mm_shuffle_epi32(_Bigger, 0xA0);
        const auto _EqualBigger     = _mm_and_si128(_Equal, _ShuffledBigger);

        const auto _Result = _mm_shuffle_epi32(_mm_or_si128(_Bigger, _EqualBigger), 0xF5);
        return __intrin_bitcast<_VectorType_>(_Result);
    }
    else if constexpr (_Is_epi32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_cmplt_epi32(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else if constexpr (_Is_epu32_v<_DesiredType_>) {
        const auto _Sign        = _mm_set1_epi32(0x80000000);

        const auto _SignedLeft  = _mm_xor_si128(__intrin_bitcast<__m128i>(__left), _Sign);
        const auto _SignedRight = _mm_xor_si128(__intrin_bitcast<__m128i>(__right), _Sign);
        
        return __intrin_bitcast<_VectorType_>(_mm_cmplt_epi32(_SignedLeft, _SignedRight));
    }
    else if constexpr (_Is_epi16_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_cmplt_epi16(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else if constexpr (_Is_epu16_v<_DesiredType_>) {
        const auto _Substracted = _mm_subs_epu16(__intrin_bitcast<__m128i>(__right), __intrin_bitcast<__m128i>(__left));
        return __simd_bit_not<_Generation, __register_policy>(__intrin_bitcast<_VectorType_>(
            _mm_cmpeq_epi16(_Substracted, _mm_setzero_si128())));
    }
    else if constexpr (_Is_epi8_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_cmplt_epi8(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else if constexpr (_Is_epu8_v<_DesiredType_>) {
        const auto _Substracted = _mm_subs_epu8(__intrin_bitcast<__m128i>(__right), __intrin_bitcast<__m128i>(__left));
        return __simd_bit_not<_Generation, __register_policy>(__intrin_bitcast<_VectorType_>(
            _mm_cmpeq_epi8(_Substracted, _mm_setzero_si128())));
    }
    else if constexpr (_Is_ps_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_cmplt_ps(
            __intrin_bitcast<__m128>(__left), __intrin_bitcast<__m128>(__right)));
    }
    else if constexpr (_Is_pd_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_cmplt_pd(
            __intrin_bitcast<__m128d>(__left), __intrin_bitcast<__m128d>(__right)));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_compare_implementation<arch::CpuFeature::SSE2, xmm128>::_CompareGreater(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    return _CompareLess<_DesiredType_>(__right, __left);
}

template <
    class   _DesiredType_,
    class   _CompareType_,
    class   _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::SSSE3, xmm128>::__simd_mask_type<_DesiredType_> 
    __simd_compare_implementation<arch::CpuFeature::SSSE3, xmm128>::_MaskCompare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    return _SimdToMask<_Generation, __register_policy, _DesiredType_>(_Compare<_DesiredType_, _CompareType_>(__left, __right));
}


template <
    class   _DesiredType_,
    class   _CompareType_,
    class   _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::SSE41, xmm128>::__simd_mask_type<_DesiredType_>
    __simd_compare_implementation<arch::CpuFeature::SSE41, xmm128>::_MaskCompare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    return _SimdToMask<_Generation, __register_policy, _DesiredType_>(_Compare<_DesiredType_, _CompareType_>(__left, __right));
}

template <
    typename _DesiredType_,
    class    _CompareType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_compare_implementation<arch::CpuFeature::SSE41, xmm128>::_Compare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (std::is_same_v<_CompareType_, type_traits::equal_to<>>)
        return _CompareEqual<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::not_equal_to<>>)
        return __simd_bit_not<_Generation, __register_policy>(_CompareEqual<_DesiredType_>(__left, __right));

    else if constexpr (std::is_same_v<_CompareType_, type_traits::less<>>)
        return _CompareLess<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::less_equal<>>)
        return __simd_bit_not<_Generation, __register_policy>(_CompareGreater<_DesiredType_>(__left, __right));

    else if constexpr (std::is_same_v<_CompareType_, type_traits::greater<>>)
        return _CompareGreater<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::greater_equal<>>)
        return __simd_bit_not<_Generation, __register_policy>(_CompareLess<_DesiredType_>(__left, __right));
}

template <
    typename _DesiredType_,
    class    _CompareType_,
    typename _VectorType_>
simd_stl_always_inline auto __simd_compare_implementation<arch::CpuFeature::SSE41, xmm128>::_NativeCompare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    return _Compare<_DesiredType_, _CompareType_>(__left, __right);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_compare_implementation<arch::CpuFeature::SSE41, xmm128>::_CompareEqual(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_cmpeq_epi64(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_cmpeq_epi32(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_cmpeq_epi16(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_cmpeq_epi8(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else if constexpr (_Is_ps_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_cmpeq_ps(
            __intrin_bitcast<__m128>(__left), __intrin_bitcast<__m128>(__right)));
    }
    else if constexpr (_Is_pd_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_cmpeq_pd(
            __intrin_bitcast<__m128d>(__left), __intrin_bitcast<__m128d>(__right)));
    }
}


template <
    class   _DesiredType_,
    class   _CompareType_,
    class   _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::SSE42, xmm128>::__simd_mask_type<_DesiredType_> 
    __simd_compare_implementation<arch::CpuFeature::SSE42, xmm128>::_MaskCompare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    return _SimdToMask<_Generation, __register_policy, _DesiredType_>(_Compare<_DesiredType_, _CompareType_>(__left, __right));
}

template <
    typename _DesiredType_,
    class    _CompareType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_compare_implementation<arch::CpuFeature::SSE42, xmm128>::_Compare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (std::is_same_v<_CompareType_, type_traits::equal_to<>>)
        return _CompareEqual<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::not_equal_to<>>)
        return __simd_bit_not<_Generation, __register_policy>(_CompareEqual<_DesiredType_>(__left, __right));

    else if constexpr (std::is_same_v<_CompareType_, type_traits::less<>>)
        return _CompareLess<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::less_equal<>>)
        return __simd_bit_not<_Generation, __register_policy>(_CompareGreater<_DesiredType_>(__left, __right));

    else if constexpr (std::is_same_v<_CompareType_, type_traits::greater<>>)
        return _CompareGreater<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::greater_equal<>>)
        return __simd_bit_not<_Generation, __register_policy>(_CompareLess<_DesiredType_>(__left, __right));
}

template <
    typename _DesiredType_,
    class    _CompareType_,
    typename _VectorType_>
simd_stl_always_inline auto __simd_compare_implementation<arch::CpuFeature::SSE42, xmm128>::_NativeCompare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    return _Compare<_DesiredType_, _CompareType_>(__left, __right);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_compare_implementation<arch::CpuFeature::SSE42, xmm128>::_CompareGreater(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (_Is_epi64_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(
            _mm_cmpgt_epi64(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else if constexpr (_Is_epu64_v<_DesiredType_>) {
        const auto _Sign64Bit   = _mm_set1_epi64x(0x8000000000000000);

        const auto __leftSigned  = _mm_xor_si128(__intrin_bitcast<__m128i>(__left), _Sign64Bit);
        const auto __rightSigned = _mm_xor_si128(__intrin_bitcast<__m128i>(__right), _Sign64Bit);

        return __intrin_bitcast<_VectorType_>(_mm_cmpgt_epi64(__leftSigned, __rightSigned));
    }
    else {
        return _SimdCompare<arch::CpuFeature::SSE2, __register_policy, _DesiredType_, type_traits::greater<>>(__left, __right);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_compare_implementation<arch::CpuFeature::SSE42, xmm128>::_CompareLess(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    return _CompareGreater<_DesiredType_>(__right, __left);
}

#pragma endregion

#pragma region Avx Simd compare


template <
    class   _DesiredType_,
    class   _CompareType_,
    class   _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX, ymm256>::__simd_mask_type<_DesiredType_> 
    __simd_compare_implementation<arch::CpuFeature::AVX, ymm256>::_MaskCompare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    return _SimdToMask<_Generation, __register_policy, _DesiredType_>(_Compare<_DesiredType_, _CompareType_>(__left, __right));
}

template <
    typename _DesiredType_,
    class    _CompareType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_compare_implementation<arch::CpuFeature::AVX, ymm256>::_Compare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (std::is_same_v<_CompareType_, type_traits::equal_to<>>)
        return _CompareEqual<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::not_equal_to<>>)
        return ~_CompareEqual<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::less<>>)
        return _CompareLess<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::less_equal<>>)
        return ~_CompareGreater<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::greater<>>)
        return _CompareGreater<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::greater_equal<>>)
        return ~_CompareLess<_DesiredType_>(__left, __right);
}

template <
    typename _DesiredType_,
    class    _CompareType_,
    typename _VectorType_>
simd_stl_always_inline auto __simd_compare_implementation<arch::CpuFeature::AVX, ymm256>::_NativeCompare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    return _Compare<_DesiredType_, _CompareType_>(__left, __right);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_compare_implementation<arch::CpuFeature::AVX, ymm256>::_CompareEqual(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (_Is_pd_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_cmp_pd(
            __intrin_bitcast<__m256d>(__left), __intrin_bitcast<__m256d>(__right), _CMP_EQ_OQ));
    }
    else if constexpr (_Is_ps_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_cmp_ps(
            __intrin_bitcast<__m256>(__left), __intrin_bitcast<__m256>(__right), _CMP_EQ_OQ));
    }
    else {
        const auto _Low = _SimdCompare<arch::CpuFeature::SSE42, xmm128, _DesiredType_, type_traits::equal_to<>>(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));

        const auto _High = _SimdCompare<arch::CpuFeature::SSE42, xmm128, _DesiredType_, type_traits::equal_to<>>(
            _mm256_extractf128_si256(__intrin_bitcast<__m256i>(__left), 1),
            _mm256_extractf128_si256(__intrin_bitcast<__m256i>(__right), 1));

        return __intrin_bitcast<_VectorType_>(_mm256_insertf128_si256(__intrin_bitcast<__m256i>(_Low), _High, 1));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_compare_implementation<arch::CpuFeature::AVX, ymm256>::_CompareLess(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    return _CompareGreater<_DesiredType_>(__right, __left);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_compare_implementation<arch::CpuFeature::AVX, ymm256>::_CompareGreater(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (_Is_pd_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_cmp_pd(
            __intrin_bitcast<__m256d>(__left), __intrin_bitcast<__m256d>(__right), _MM_CMPINT_GT));
    }
    else if constexpr (_Is_ps_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_cmp_ps(
            __intrin_bitcast<__m256>(__left), __intrin_bitcast<__m256>(__right), _MM_CMPINT_GT));
    }
    else {
        const auto _Low = _SimdCompare<arch::CpuFeature::SSE42, xmm128, _DesiredType_, type_traits::greater<>>(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));

        const auto _High = _SimdCompare<arch::CpuFeature::SSE42, xmm128, _DesiredType_, type_traits::greater<>>(
            _mm256_extractf128_si256(__intrin_bitcast<__m256i>(__left), 1),
            _mm256_extractf128_si256(__intrin_bitcast<__m256i>(__right), 1));

        return __intrin_bitcast<_VectorType_>(_mm256_insertf128_si256(__intrin_bitcast<__m256i>(_Low), _High, 1));
    }
}


template <
    class   _DesiredType_,
    class   _CompareType_,
    class   _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX2, ymm256>::__simd_mask_type<_DesiredType_> 
    __simd_compare_implementation<arch::CpuFeature::AVX2, ymm256>::_MaskCompare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    return _SimdToMask<_Generation, __register_policy, _DesiredType_>(_Compare<_DesiredType_, _CompareType_>(__left, __right));
}

template <
    typename _DesiredType_,
    class    _CompareType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_compare_implementation<arch::CpuFeature::AVX2, ymm256>::_Compare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (std::is_same_v<_CompareType_, type_traits::equal_to<>>)
        return _CompareEqual<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::not_equal_to<>>)
        return __simd_bit_not<_Generation, __register_policy>(_CompareEqual<_DesiredType_>(__left, __right));

    else if constexpr (std::is_same_v<_CompareType_, type_traits::less<>>)
        return _CompareLess<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::less_equal<>>)
        return __simd_bit_not<_Generation, __register_policy>(_CompareGreater<_DesiredType_>(__left, __right));

    else if constexpr (std::is_same_v<_CompareType_, type_traits::greater<>>)
        return _CompareGreater<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::greater_equal<>>)
        return __simd_bit_not<_Generation, __register_policy>(_CompareLess<_DesiredType_>(__left, __right));
}

template <
    typename _DesiredType_,
    class    _CompareType_,
    typename _VectorType_>
simd_stl_always_inline auto __simd_compare_implementation<arch::CpuFeature::AVX2, ymm256>::_NativeCompare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    return _Compare<_DesiredType_, _CompareType_>(__left, __right);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_compare_implementation<arch::CpuFeature::AVX2, ymm256>::_CompareEqual(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (_Is_pd_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_cmp_pd(
            __intrin_bitcast<__m256d>(__left), __intrin_bitcast<__m256d>(__right), _CMP_EQ_OQ));

    else if constexpr (_Is_ps_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_cmp_ps(
            __intrin_bitcast<__m256>(__left), __intrin_bitcast<__m256>(__right), _CMP_EQ_OQ));

    else if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_cmpeq_epi64(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));

    else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_cmpeq_epi32(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));

    else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_cmpeq_epi16(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));

    else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_cmpeq_epi8(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_compare_implementation<arch::CpuFeature::AVX2, ymm256>::_CompareLess(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    return _CompareGreater<_DesiredType_>(__right, __left);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_compare_implementation<arch::CpuFeature::AVX2, ymm256>::_CompareGreater(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (_Is_pd_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_cmp_pd(
            __intrin_bitcast<__m256d>(__left), __intrin_bitcast<__m256d>(__right), _MM_CMPINT_GT));
    }
    else if constexpr (_Is_ps_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_cmp_ps(
            __intrin_bitcast<__m256>(__left), __intrin_bitcast<__m256>(__right), _MM_CMPINT_GT));
    }
    else if constexpr (_Is_epi64_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_cmpgt_epi64(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));
    }
    else if constexpr (_Is_epu64_v<_DesiredType_>) {
        const auto _Sign64Bit = _mm256_set1_epi64x(0x8000000000000000);

        const auto __leftSigned  = _mm256_xor_si256(__intrin_bitcast<__m256i>(__left), _Sign64Bit);
        const auto __rightSigned = _mm256_xor_si256(__intrin_bitcast<__m256i>(__right), _Sign64Bit);

        return __intrin_bitcast<_VectorType_>(_mm256_cmpgt_epi64(__leftSigned, __rightSigned));
    }
    else if constexpr (_Is_epi32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_cmpgt_epi32(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));
    }
    else if constexpr (_Is_epu32_v<_DesiredType_>) {
        const auto _Sign64Bit = _mm256_set1_epi32(0x80000000);

        const auto __leftSigned  = _mm256_xor_si256(__intrin_bitcast<__m256i>(__left), _Sign64Bit);
        const auto __rightSigned = _mm256_xor_si256(__intrin_bitcast<__m256i>(__right), _Sign64Bit);

        return __intrin_bitcast<_VectorType_>(_mm256_cmpgt_epi32(__leftSigned, __rightSigned));
    }
    else if constexpr (_Is_epi16_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_cmpgt_epi16(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));
    }
    else if constexpr (_Is_epu16_v<_DesiredType_>) {
        const auto _Sign64Bit = _mm256_set1_epi16(0x8000);

        const auto __leftSigned  = _mm256_xor_si256(__intrin_bitcast<__m256i>(__left), _Sign64Bit);
        const auto __rightSigned = _mm256_xor_si256(__intrin_bitcast<__m256i>(__right), _Sign64Bit);

        return __intrin_bitcast<_VectorType_>(_mm256_cmpgt_epi16(__leftSigned, __rightSigned));
    }
    else if constexpr (_Is_epi8_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_cmpgt_epi8(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));
    }
    else if constexpr (_Is_epu8_v<_DesiredType_>) {
        const auto _Sign64Bit   = _mm256_set1_epi8(0x80);

        const auto __leftSigned  = _mm256_xor_si256(__intrin_bitcast<__m256i>(__left), _Sign64Bit);
        const auto __rightSigned = _mm256_xor_si256(__intrin_bitcast<__m256i>(__right), _Sign64Bit);

        return __intrin_bitcast<_VectorType_>(_mm256_cmpgt_epi8(__leftSigned, __rightSigned));
    }
}

#pragma endregion

#pragma region Avx512 Simd compare


template <
    class _DesiredType_,
    class _CompareType_,
    class _VectorType_>
simd_stl_always_inline _VectorType_ __simd_compare_implementation<arch::CpuFeature::AVX512F, zmm512>::_BlockwiseCompare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    const auto _ComparedLow128 = _SimdCompare<arch::CpuFeature::SSE42, xmm128, _DesiredType_, _CompareType_>(
        __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));

    const auto _Compared2Low128 = _SimdCompare<arch::CpuFeature::SSE42, xmm128, _DesiredType_, _CompareType_>(
        _mm256_extractf128_si256(__intrin_bitcast<__m256i>(__left), 1), _mm256_extractf128_si256(__intrin_bitcast<__m256i>(__right), 1));

    const auto _ComparedHigh128 = _SimdCompare<arch::CpuFeature::SSE42, xmm128, _DesiredType_, _CompareType_>(
        _mm512_extracti32x4_epi32(__intrin_bitcast<__m512i>(__left), 2), _mm512_extracti32x4_epi32(__intrin_bitcast<__m512i>(__right), 2));

    const auto _Compared2High128 = _SimdCompare<arch::CpuFeature::SSE42, xmm128, _DesiredType_, _CompareType_>(
        _mm512_extracti32x4_epi32(__intrin_bitcast<__m512i>(__left), 3), _mm512_extracti32x4_epi32(__intrin_bitcast<__m512i>(__right), 3));

    auto _Result = __intrin_bitcast<__m512i>(_ComparedLow128);

    _Result = _mm512_inserti32x4(_Result, _Compared2Low128, 1);
    _Result = _mm512_inserti32x4(_Result, _ComparedHigh128, 2);

    return __intrin_bitcast<_VectorType_>(_mm512_inserti32x4(_Result, _Compared2High128, 3));
}

template <
    class   _DesiredType_,
    class   _CompareType_,
    class   _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512F, zmm512>::__simd_mask_type<_DesiredType_>
    __simd_compare_implementation<arch::CpuFeature::AVX512F, zmm512>::_MaskCompare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    if constexpr (std::is_same_v<_CompareType_, type_traits::equal_to<>>)
        return _MaskCompareEqual<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::not_equal_to<>>)
        return ~_MaskCompareEqual<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::less<>>)
        return _MaskCompareLess<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::less_equal<>>)
        return ~_MaskCompareGreater<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::greater<>>)
        return _MaskCompareGreater<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::greater_equal<>>)
        return ~_MaskCompareLess<_DesiredType_>(__left, __right);
}

template <
    typename _DesiredType_,
    class    _CompareType_,
    typename _VectorType_>
simd_stl_always_inline auto __simd_compare_implementation<arch::CpuFeature::AVX512F, zmm512>::_NativeCompare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (sizeof(_DesiredType_) >= 4)
        return _MaskCompare<_DesiredType_, _CompareType_>(__left, __right);
    else
        return _Compare<_DesiredType_, _CompareType_>(__left, __right);
}

template <
    typename _DesiredType_,
    class    _CompareType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_compare_implementation<arch::CpuFeature::AVX512F, zmm512>::_Compare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_> ||
        _Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>)
    {
        return _BlockwiseCompare<_DesiredType_, _CompareType_>(__left, __right);
    }
    else {
        return _SimdToVector<_Generation, __register_policy, _VectorType_>(_MaskCompare<_DesiredType_, _CompareType_>(__left, __right));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512F, zmm512>::__simd_mask_type<_DesiredType_>
    __simd_compare_implementation<arch::CpuFeature::AVX512F, zmm512>::_MaskCompareEqual(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    if constexpr (_Is_pd_v<_DesiredType_>) {
        return _mm512_cmpeq_pd_mask(__intrin_bitcast<__m512d>(__left), __intrin_bitcast<__m512d>(__right));
    }
    else if constexpr (_Is_ps_v<_DesiredType_>) {
        return _mm512_cmpeq_ps_mask(__intrin_bitcast<__m512>(__left), __intrin_bitcast<__m512>(__right));
    }
    else if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>) {
        return _mm512_cmpeq_epi64_mask(__intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right));
    }
    else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>) {
        return _mm512_cmpeq_epi32_mask(__intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right));
    }
    else {
        return _SimdToMask<_Generation, __register_policy, _DesiredType_>(
            _BlockwiseCompare<_DesiredType_, type_traits::equal_to<>>(__left, __right));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512F, zmm512>::__simd_mask_type<_DesiredType_> 
    __simd_compare_implementation<arch::CpuFeature::AVX512F, zmm512>::_MaskCompareLess(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    if constexpr (_Is_epi64_v<_DesiredType_>) {
        return _mm512_cmplt_epi64_mask(__intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right));
    }
    else if constexpr (_Is_epu64_v<_DesiredType_>) {
        return _mm512_cmplt_epu64_mask(__intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right));
    }
    else if constexpr (_Is_epi32_v<_DesiredType_>) {
        return _mm512_cmplt_epi32_mask(__intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right));
    }
    else if constexpr (_Is_epu32_v<_DesiredType_>) {
        return _mm512_cmplt_epu32_mask(__intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right));
    }
    else if constexpr (_Is_ps_v<_DesiredType_>) {
        return _mm512_cmplt_ps_mask(__intrin_bitcast<__m512>(__left), __intrin_bitcast<__m512>(__right));
    }
    else if constexpr (_Is_pd_v<_DesiredType_>) {
        return _mm512_cmplt_pd_mask(__intrin_bitcast<__m512d>(__left), __intrin_bitcast<__m512d>(__right));
    }
    else {
        return _SimdToMask<_Generation, __register_policy, _DesiredType_>(
            _BlockwiseCompare<_DesiredType_, type_traits::less<>>(__left, __right));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512F, zmm512>::__simd_mask_type<_DesiredType_> 
    __simd_compare_implementation<arch::CpuFeature::AVX512F, zmm512>::_MaskCompareGreater(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    return _MaskCompareLess<_DesiredType_>(__right, __left);
}


template <
    class   _DesiredType_,
    class   _CompareType_,
    class   _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512BW, zmm512>::__simd_mask_type<_DesiredType_> 
    __simd_compare_implementation<arch::CpuFeature::AVX512BW, zmm512>::_MaskCompare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    if constexpr (std::is_same_v<_CompareType_, type_traits::equal_to<>>)
        return _MaskCompareEqual<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::not_equal_to<>>)
        return ~_MaskCompareEqual<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::less<>>)
        return _MaskCompareLess<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::less_equal<>>)
        return ~_MaskCompareGreater<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::greater<>>)
        return _MaskCompareGreater<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::greater_equal<>>)
        return ~_MaskCompareLess<_DesiredType_>(__left, __right);
}

template <
    typename _DesiredType_,
    class    _CompareType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_compare_implementation<arch::CpuFeature::AVX512BW, zmm512>::_Compare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    return _SimdToVector<_Generation, __register_policy, _VectorType_>(_MaskCompare<_DesiredType_, _CompareType_>(__left, __right));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512BW, zmm512>::__simd_mask_type<_DesiredType_> 
    __simd_compare_implementation<arch::CpuFeature::AVX512BW, zmm512>::_MaskCompareEqual(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    if constexpr (_Is_pd_v<_DesiredType_>)
        return _mm512_cmpeq_pd_mask(__intrin_bitcast<__m512d>(__left), __intrin_bitcast<__m512d>(__right));

    else if constexpr (_Is_ps_v<_DesiredType_>)
        return _mm512_cmpeq_ps_mask(__intrin_bitcast<__m512>(__left), __intrin_bitcast<__m512>(__right));

    else if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>)
        return _mm512_cmpeq_epi64_mask(__intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right));

    else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
        return _mm512_cmpeq_epi32_mask(__intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right));

    else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>)
        return _mm512_cmpeq_epi16_mask(__intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right));

    else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>)
        return _mm512_cmpeq_epi8_mask(__intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right));
}

template <
    typename _DesiredType_,
    class    _CompareType_,
    typename _VectorType_>
simd_stl_always_inline auto __simd_compare_implementation<arch::CpuFeature::AVX512BW, zmm512>::_NativeCompare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    return _MaskCompare<_DesiredType_, _CompareType_>(__left, __right);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512BW, zmm512>::__simd_mask_type<_DesiredType_> 
    __simd_compare_implementation<arch::CpuFeature::AVX512BW, zmm512>::_MaskCompareLess(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    return _MaskCompareGreater<_DesiredType_>(__right, __left);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512BW, zmm512>::__simd_mask_type<_DesiredType_> 
    __simd_compare_implementation<arch::CpuFeature::AVX512BW, zmm512>::_MaskCompareGreater(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    if constexpr (_Is_epi64_v<_DesiredType_>) {
        return _mm512_cmpgt_epi64_mask(__intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right));
    }
    else if constexpr (_Is_epu64_v<_DesiredType_>) {
        return _mm512_cmpgt_epu64_mask(__intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right));
    }
    else if constexpr (_Is_epi32_v<_DesiredType_>) {
        return _mm512_cmpgt_epi32_mask(__intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right));
    }
    else if constexpr (_Is_epu32_v<_DesiredType_>) {
        return _mm512_cmpgt_epu32_mask(__intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right));
    }
    else if constexpr (_Is_epi16_v<_DesiredType_>) {
        return _mm512_cmpgt_epi16_mask(__intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right));
    }
    else if constexpr (_Is_epu16_v<_DesiredType_>) {
        return _mm512_cmpgt_epu16_mask(__intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right));
    }
    else if constexpr (_Is_epi8_v<_DesiredType_>) {
        return _mm512_cmpgt_epi8_mask(__intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right));
    }
    else if constexpr (_Is_epu8_v<_DesiredType_>) {
        return _mm512_cmpgt_epu8_mask(__intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right));
    }
    else if constexpr (_Is_ps_v<_DesiredType_>) {
        return _mm512_cmplt_ps_mask(__intrin_bitcast<__m512>(__right), __intrin_bitcast<__m512>(__left));
    }
    else if constexpr (_Is_pd_v<_DesiredType_>) {
        return _mm512_cmplt_pd_mask(__intrin_bitcast<__m512d>(__right), __intrin_bitcast<__m512d>(__left));
    }
}

template <
    class   _DesiredType_,
    class   _CompareType_,
    class   _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512VLF, ymm256>::__simd_mask_type<_DesiredType_>
    __simd_compare_implementation<arch::CpuFeature::AVX512VLF, ymm256>::_MaskCompare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    if constexpr (std::is_same_v<_CompareType_, type_traits::equal_to<>>)
        return _MaskCompareEqual<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::not_equal_to<>>)
        return ~_MaskCompareEqual<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::less<>>)
        return _MaskCompareLess<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::less_equal<>>)
        return ~_MaskCompareGreater<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::greater<>>)
        return _MaskCompareGreater<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::greater_equal<>>)
        return ~_MaskCompareLess<_DesiredType_>(__left, __right);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512VLF, ymm256>::__simd_mask_type<_DesiredType_> 
    __simd_compare_implementation<arch::CpuFeature::AVX512VLF, ymm256>::_MaskCompareEqual(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    if constexpr (_Is_pd_v<_DesiredType_>)
        return _mm256_cmp_pd_mask(__intrin_bitcast<__m256d>(__left), __intrin_bitcast<__m256d>(__right), _CMP_EQ_OQ);

    else if constexpr (_Is_ps_v<_DesiredType_>)
        return _mm256_cmp_ps_mask(__intrin_bitcast<__m256>(__left), __intrin_bitcast<__m256>(__right), _CMP_EQ_OQ);

    else if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>)
        return _mm256_cmpeq_epi64_mask(__intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right));

    else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
        return _mm256_cmpeq_epi32_mask(__intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right));

    else
        return _SimdMaskCompare<arch::CpuFeature::AVX2, __register_policy, _DesiredType_, type_traits::equal_to<>>(__left, __right);
}

template <
    typename _DesiredType_,
    class    _CompareType_,
    typename _VectorType_>
simd_stl_always_inline auto __simd_compare_implementation<arch::CpuFeature::AVX512VLF, ymm256>::_NativeCompare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (sizeof(_DesiredType_) >= 4)
        return _MaskCompare<_DesiredType_, _CompareType_>(__left, __right);
    else
        return _Compare<_DesiredType_, _CompareType_>(__left, __right);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512VLF, ymm256>::__simd_mask_type<_DesiredType_> 
    __simd_compare_implementation<arch::CpuFeature::AVX512VLF, ymm256>::_MaskCompareLess(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    return _MaskCompareGreater<_DesiredType_>(__right, __left);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512VLF, ymm256>::__simd_mask_type<_DesiredType_> 
    __simd_compare_implementation<arch::CpuFeature::AVX512VLF, ymm256>::_MaskCompareGreater(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    if constexpr (_Is_epi64_v<_DesiredType_>) {
        return _mm256_cmpgt_epi64_mask(__intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right));
    }
    else if constexpr (_Is_epu64_v<_DesiredType_>) {
        return _mm256_cmpgt_epu64_mask(__intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right));
    }
    else if constexpr (_Is_epi32_v<_DesiredType_>) {
        return _mm256_cmpgt_epi32_mask(__intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right));
    }
    else if constexpr (_Is_epu32_v<_DesiredType_>) {
        return _mm256_cmpgt_epu32_mask(__intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right));
    }
    else if constexpr (_Is_ps_v<_DesiredType_>) {
        return _mm256_cmp_ps_mask(__intrin_bitcast<__m256>(__left), __intrin_bitcast<__m256>(__right), _CMP_GT_OQ);
    }
    else if constexpr (_Is_pd_v<_DesiredType_>) {
        return _mm256_cmp_pd_mask(__intrin_bitcast<__m256d>(__left), __intrin_bitcast<__m256d>(__right), _CMP_GT_OQ);
    }
    else {
        return _SimdMaskCompare<arch::CpuFeature::AVX2, __register_policy, _DesiredType_, type_traits::greater<>>(__left, __right);
    }
}

template <
    class   _DesiredType_,
    class   _CompareType_,
    class   _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, ymm256>::__simd_mask_type<_DesiredType_> 
    __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, ymm256>::_MaskCompare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    if constexpr (std::is_same_v<_CompareType_, type_traits::equal_to<>>)
        return _MaskCompareEqual<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::not_equal_to<>>)
        return ~_MaskCompareEqual<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::less<>>)
        return _MaskCompareLess<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::less_equal<>>)
        return ~_MaskCompareGreater<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::greater<>>)
        return _MaskCompareGreater<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::greater_equal<>>)
        return ~_MaskCompareLess<_DesiredType_>(__left, __right);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, ymm256>::__simd_mask_type<_DesiredType_> 
    __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, ymm256>::_MaskCompareEqual(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    if constexpr (_Is_epi16_v<_DesiredType_>) {
        return _mm256_cmpeq_epi16_mask(__intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right));
    }
    else if constexpr (_Is_epu16_v<_DesiredType_>) {
        return _mm256_cmpeq_epu16_mask(__intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right));
    }
    else if constexpr (_Is_epi8_v<_DesiredType_>) {
        return _mm256_cmpeq_epi8_mask(__intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right));
    }
    else if constexpr (_Is_epu8_v<_DesiredType_>) {
        return _mm256_cmpeq_epu8_mask(__intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right));
    }
    else {
        return _SimdMaskCompare<arch::CpuFeature::AVX512VLF, __register_policy, _DesiredType_, type_traits::equal_to<>>(__left, __right);
    }
}

template <
    typename _DesiredType_,
    class    _CompareType_,
    typename _VectorType_>
simd_stl_always_inline auto __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, ymm256>::_NativeCompare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    return _MaskCompare<_DesiredType_, _CompareType_>(__left, __right);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, ymm256>::__simd_mask_type<_DesiredType_> 
    __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, ymm256>::_MaskCompareLess(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    return _MaskCompareGreater<_DesiredType_>(__right, __left);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, ymm256>::__simd_mask_type<_DesiredType_> 
    __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, ymm256>::_MaskCompareGreater(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    if constexpr (_Is_epi16_v<_DesiredType_>) {
        return _mm256_cmpgt_epi16_mask(__intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right));
    }
    else if constexpr (_Is_epu16_v<_DesiredType_>) {
        return _mm256_cmpgt_epu16_mask(__intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right));
    }
    else if constexpr (_Is_epi8_v<_DesiredType_>) {
        return _mm256_cmpgt_epi8_mask(__intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right));
    }
    else if constexpr (_Is_epu8_v<_DesiredType_>) {
        return _mm256_cmpgt_epu8_mask(__intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right));
    }
    else {
        return _SimdMaskCompare<arch::CpuFeature::AVX512VLF, __register_policy, _DesiredType_, type_traits::greater<>>(__left, __right);
    }
}

template <
    class   _DesiredType_,
    class   _CompareType_,
    class   _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512VLF, xmm128>::__simd_mask_type<_DesiredType_>
    __simd_compare_implementation<arch::CpuFeature::AVX512VLF, xmm128>::_MaskCompare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    if constexpr (std::is_same_v<_CompareType_, type_traits::equal_to<>>)
        return _MaskCompareEqual<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::not_equal_to<>>)
        return ~_MaskCompareEqual<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::less<>>)
        return _MaskCompareLess<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::less_equal<>>)
        return ~_MaskCompareGreater<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::greater<>>)
        return _MaskCompareGreater<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::greater_equal<>>)
        return ~_MaskCompareLess<_DesiredType_>(__left, __right);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512VLF, xmm128>::__simd_mask_type<_DesiredType_>
    __simd_compare_implementation<arch::CpuFeature::AVX512VLF, xmm128>::_MaskCompareEqual(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    if constexpr (_Is_pd_v<_DesiredType_>)
        return _mm_cmp_pd_mask(__intrin_bitcast<__m128d>(__left), __intrin_bitcast<__m128d>(__right), _CMP_EQ_OQ);

    else if constexpr (_Is_ps_v<_DesiredType_>)
        return _mm_cmp_ps_mask(__intrin_bitcast<__m128>(__left), __intrin_bitcast<__m128>(__right), _CMP_EQ_OQ);

    else if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>)
        return _mm_cmpeq_epi64_mask(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));

    else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
        return _mm_cmpeq_epi32_mask(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));

    else
        return _SimdMaskCompare<arch::CpuFeature::SSE42, __register_policy, _DesiredType_, type_traits::equal_to<>>(__left, __right);
}

template <
    typename _DesiredType_,
    class    _CompareType_,
    typename _VectorType_>
simd_stl_always_inline auto __simd_compare_implementation<arch::CpuFeature::AVX512VLF, xmm128>::_NativeCompare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (sizeof(_DesiredType_) >= 4)
        return _MaskCompare<_DesiredType_, _CompareType_>(__left, __right);
    else
        return _Compare<_DesiredType_, _CompareType_>(__left, __right);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512VLF, xmm128>::__simd_mask_type<_DesiredType_>
    __simd_compare_implementation<arch::CpuFeature::AVX512VLF, xmm128>::_MaskCompareLess(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    return _MaskCompareGreater<_DesiredType_>(__right, __left);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512VLF, xmm128>::__simd_mask_type<_DesiredType_>
    __simd_compare_implementation<arch::CpuFeature::AVX512VLF, xmm128>::_MaskCompareGreater(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    if constexpr (_Is_epi64_v<_DesiredType_>) {
        return _mm_cmpgt_epi64_mask(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));
    }
    else if constexpr (_Is_epu64_v<_DesiredType_>) {
        return _mm_cmpgt_epu64_mask(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));
    }
    else if constexpr (_Is_epi32_v<_DesiredType_>) {
        return _mm_cmpgt_epi32_mask(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));
    }
    else if constexpr (_Is_epu32_v<_DesiredType_>) {
        return _mm_cmpgt_epu32_mask(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));
    }
    else if constexpr (_Is_ps_v<_DesiredType_>) {
        return _mm_cmp_ps_mask(__intrin_bitcast<__m128>(__left), __intrin_bitcast<__m128>(__right), _CMP_GT_OQ);
    }
    else if constexpr (_Is_pd_v<_DesiredType_>) {
        return _mm_cmp_pd_mask(__intrin_bitcast<__m128d>(__left), __intrin_bitcast<__m128d>(__right), _CMP_GT_OQ);
    }
    else {
        return _SimdMaskCompare<arch::CpuFeature::SSE42, __register_policy, _DesiredType_, type_traits::greater<>>(__left, __right);
    }
}

template <
    class   _DesiredType_,
    class   _CompareType_,
    class   _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, xmm128>::__simd_mask_type<_DesiredType_>
    __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, xmm128>::_MaskCompare(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    if constexpr (std::is_same_v<_CompareType_, type_traits::equal_to<>>)
        return _MaskCompareEqual<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::not_equal_to<>>)
        return ~_MaskCompareEqual<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::less<>>)
        return _MaskCompareLess<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::less_equal<>>)
        return ~_MaskCompareGreater<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::greater<>>)
        return _MaskCompareGreater<_DesiredType_>(__left, __right);

    else if constexpr (std::is_same_v<_CompareType_, type_traits::greater_equal<>>)
        return ~_MaskCompareLess<_DesiredType_>(__left, __right);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, xmm128>::__simd_mask_type<_DesiredType_>
    __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, xmm128>::_MaskCompareEqual(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    if constexpr (_Is_epi16_v<_DesiredType_>) {
        return _mm_cmpeq_epi16_mask(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));
    }
    else if constexpr (_Is_epu16_v<_DesiredType_>) {
        return _mm_cmpeq_epu16_mask(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));
    }
    else if constexpr (_Is_epi8_v<_DesiredType_>) {
        return _mm_cmpeq_epi8_mask(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));
    }
    else if constexpr (_Is_epu8_v<_DesiredType_>) {
        return _mm_cmpeq_epu8_mask(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));
    }
    else {
        return _SimdMaskCompare<arch::CpuFeature::AVX512VLF, __register_policy, _DesiredType_, type_traits::equal_to<>>(__left, __right);
    }
}

template <
    typename _DesiredType_,
    class    _CompareType_,
    typename _VectorType_>
simd_stl_always_inline auto __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, xmm128>::_NativeCompare(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    return _MaskCompare<_DesiredType_, _CompareType_>(__left, __right);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, xmm128>::__simd_mask_type<_DesiredType_>
    __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, xmm128>::_MaskCompareLess(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    return _MaskCompareGreater<_DesiredType_>(__right, __left);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, xmm128>::__simd_mask_type<_DesiredType_>
    __simd_compare_implementation<arch::CpuFeature::AVX512VLBW, xmm128>::_MaskCompareGreater(
        _VectorType_ __left,
        _VectorType_ __right) noexcept
{
    if constexpr (_Is_epi16_v<_DesiredType_>) {
        return _mm_cmpgt_epi16_mask(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));
    }
    else if constexpr (_Is_epu16_v<_DesiredType_>) {
        return _mm_cmpgt_epu16_mask(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));
    }
    else if constexpr (_Is_epi8_v<_DesiredType_>) {
        return _mm_cmpgt_epi8_mask(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));
    }
    else if constexpr (_Is_epu8_v<_DesiredType_>) {
        return _mm_cmpgt_epu8_mask(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right));
    }
    else {
        return _SimdMaskCompare<arch::CpuFeature::AVX512VLF, __register_policy, _DesiredType_, type_traits::greater<>>(__left, __right);
    }
}


#pragma endregion 

__SIMD_STL_NUMERIC_NAMESPACE_END
