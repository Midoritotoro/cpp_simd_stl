#pragma once 


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

#pragma region Sse2-Sse4.2 Simd arithmetic

template <typename _Type_>
struct _Reduce_type_helper {
    static constexpr auto _Size = sizeof(_Type_);

    using type =
        std::conditional_t<
            _Size == 1,
            std::conditional_t<std::is_unsigned_v<_Type_>, uint32, int32>,
        std::conditional_t<
            _Size == 2,
            std::conditional_t<std::is_unsigned_v<_Type_>, uint64, int64>,
        std::conditional_t<
            _Size == 4,
            std::conditional_t<std::is_floating_point_v<_Type_>, double,
                std::conditional_t<std::is_unsigned_v<_Type_>, uint64, int64>>,
        std::conditional_t<
            _Size == 8,
            std::conditional_t<std::is_floating_point_v<_Type_>, double,
                std::conditional_t<std::is_unsigned_v<_Type_>, uint64, int64>>,
        int64>>>>;
    
    static_assert(!std::is_same_v<type, void>, "Unsupported type size for _Reduce_type_helper");
};

template <typename _Type_>
using _Reduce_type = typename _Reduce_type_helper<_Type_>::type;

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::SSE2, xmm128>::_VerticalMin(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if constexpr (_Is_epi16_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_min_epi16(
            __intrin_bitcast<__m128i>(_Left), __intrin_bitcast<__m128i>(_Right)));
    }
    else if constexpr (_Is_epu8_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_min_epu8(
            __intrin_bitcast<__m128i>(_Left), __intrin_bitcast<__m128i>(_Right)));
    }
    else if constexpr (_Is_ps_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_min_ps(
            __intrin_bitcast<__m128>(_Left), __intrin_bitcast<__m128>(_Right)));
    }
    else if constexpr (_Is_pd_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_min_pd(
            __intrin_bitcast<__m128d>(_Left), __intrin_bitcast<__m128d>(_Right)));
    }
    else {
        const auto _Mask = _SimdCompare<_Generation, _RegisterPolicy, _DesiredType_, type_traits::less<>>(_Left, _Right);
        return _SimdBlend<_Generation, _RegisterPolicy, _DesiredType_>(_Left, _Right, _Mask);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_,
    typename _ReduceBinaryFunction_>
simd_stl_always_inline _DesiredType_ _SimdArithmetic<arch::CpuFeature::SSE2, xmm128>::_HorizontalFold(
    _VectorType_            _Vector,
    _ReduceBinaryFunction_  _Reduce) noexcept
{
    if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_> || _Is_pd_v<_DesiredType_>) {
        auto _HorizontalFoldedValues = __intrin_bitcast<__m128d>(_Vector);

        const auto _Shuffled    = _mm_shuffle_pd(_HorizontalFoldedValues, _HorizontalFoldedValues, 1);
        _HorizontalFoldedValues = _Reduce(_Shuffled, _HorizontalFoldedValues);

        if constexpr (_Is_pd_v<_DesiredType_>)
            return _mm_cvtsd_f64(_HorizontalFoldedValues);
        else
            return _mm_cvtsi128_si64(__intrin_bitcast<__m128i>(_HorizontalFoldedValues));
    }
    else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_> || _Is_ps_v<_DesiredType_>) {
        auto _HorizontalFoldedValues = __intrin_bitcast<__m128i>(_Vector);

        const auto _Shuffled1   = _mm_shuffle_epi32(_HorizontalFoldedValues, 0x4E);
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, _Shuffled1);

        const auto _Shuffled2   = _mm_shuffle_epi32(_HorizontalFoldedValues, 0xB1);
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, _Shuffled2);

        if constexpr (_Is_ps_v<_DesiredType_>)
            return _mm_cvtss_f32(__intrin_bitcast<__m128>(_HorizontalFoldedValues));
        else
            return _mm_cvtsi128_si32(_HorizontalFoldedValues);
    }
    else {
        constexpr auto _Length = sizeof(_VectorType_) / sizeof(_DesiredType_);

        _DesiredType_ _Array[_Length];
        _SimdStoreUnaligned<_Generation, _RegisterPolicy>(_Array, _Vector);

        _DesiredType_ _Folded = _Array[0];

        for (auto _Index = 1; _Index < _Length; ++_Index)
            if (_Reduce(_Array[_Index], _Folded))
                _Folded = _Array[_Index];

        return _Folded;
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_ _SimdArithmetic<arch::CpuFeature::SSE2, xmm128>::_HorizontalMin(_VectorType_ _Vector) noexcept {
    if constexpr (sizeof(_DesiredType_) >= 4)
        return _HorizontalFold<_DesiredType_>(_Vector, _VerticalMinWrapper<_Generation, _RegisterPolicy, _DesiredType_>{});
    else
        return _HorizontalFold<_DesiredType_>(_Vector, [](auto&& _Left, auto&& _Right) { return _Left < _Right; });
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::SSE2, xmm128>::_VerticalMax(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if constexpr (_Is_epi16_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_max_epi16(
            __intrin_bitcast<__m128i>(_Left), __intrin_bitcast<__m128i>(_Right)));
    }
    else if constexpr (_Is_epu8_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_max_epu8(
            __intrin_bitcast<__m128i>(_Left), __intrin_bitcast<__m128i>(_Right)));
    }
    else if constexpr (_Is_ps_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_max_ps(
            __intrin_bitcast<__m128>(_Left), __intrin_bitcast<__m128>(_Right)));
    }
    else if constexpr (_Is_pd_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_max_pd(
            __intrin_bitcast<__m128d>(_Left), __intrin_bitcast<__m128d>(_Right)));
    }
    else {
        const auto _Mask = _SimdCompare<_Generation, _RegisterPolicy, _DesiredType_, type_traits::greater<>>(_Left, _Right);
        return _SimdBlend<_Generation, _RegisterPolicy, _DesiredType_>(_Left, _Right, _Mask);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_ _SimdArithmetic<arch::CpuFeature::SSE2, xmm128>::_HorizontalMax(_VectorType_ _Vector) noexcept {
    if constexpr (sizeof(_DesiredType_) >= 4)
        return _HorizontalFold<_DesiredType_>(_Vector, _VerticalMaxWrapper<_Generation, _RegisterPolicy, _DesiredType_>{});
    else
        return _HorizontalFold<_DesiredType_>(_Vector, [] (auto&& _Left, auto&& _Right) { return _Left > _Right; });
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::SSE2, xmm128>::_Abs(_VectorType_ _Vector) noexcept {
    if constexpr (std::is_unsigned_v<_DesiredType_>) {
        return _Vector;
    }
    else if constexpr (_Is_epi64_v<_DesiredType_>) {
        const auto _HighSign    = _mm_srai_epi32(__intrin_bitcast<__m128i>(_Vector), 31);
        const auto _Sign        = _mm_shuffle_epi32(_HighSign, 0xF5);

        const auto _Invert  = _mm_xor_si128(__intrin_bitcast<__m128i>(_Vector), _Sign);
        return __intrin_bitcast<_VectorType_>(_mm_sub_epi64(_Invert, _Sign));
    }
    else if constexpr (_Is_epi32_v<_DesiredType_>) {
        const auto _Sign    = _mm_srai_epi32(__intrin_bitcast<__m128i>(_Vector), 31);
        const auto _Invert  = _mm_xor_si128(__intrin_bitcast<__m128i>(_Vector), _Sign);

        return __intrin_bitcast<_VectorType_>(_mm_sub_epi32(_Invert, _Sign));
    }
    else if constexpr (_Is_epi16_v<_DesiredType_>) {
        const auto _Negate = _mm_sub_epi16(_mm_setzero_si128(), __intrin_bitcast<__m128i>(_Vector));
        return _mm_max_epi16(__intrin_bitcast<__m128i>(_Vector), _Negate);
    }
    else if constexpr (_Is_epi8_v<_DesiredType_>) {
        const auto _Negate = _mm_sub_epi8(_mm_setzero_si128(), __intrin_bitcast<__m128i>(_Vector));
        return __intrin_bitcast<_VectorType_>(_mm_min_epu8(__intrin_bitcast<__m128i>(_Vector), _Negate));
    }
    else if constexpr (_Is_ps_v<_DesiredType_>) {
        const auto _Mask = _mm_set1_epi32(0x7FFFFFFF);
        return __intrin_bitcast<_VectorType_>(_mm_and_ps(__intrin_bitcast<__m128>(_Vector), __intrin_bitcast<__m128>(_Mask)));
    }
    else if constexpr (_Is_pd_v<_DesiredType_>) {
        const auto _Mask = _mm_set_epi32(0xFFFFFFFFu, 0x7FFFFFFFu, 0xFFFFFFFFu, 0x7FFFFFFFu);
        return __intrin_bitcast<_VectorType_>(_mm_and_pd(__intrin_bitcast<__m128d>(_Vector), __intrin_bitcast<__m128d>(_Mask)));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::SSE2, xmm128>
    ::_AdjustToUnsigned(_VectorType_ _Vector) noexcept
{

}


template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline auto _SimdArithmetic<arch::CpuFeature::SSE2, xmm128>::_Reduce(_VectorType_ _Vector) noexcept {
    using _ReduceType = _Reduce_type<_DesiredType_>;

    if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>) {
#if defined(simd_stl_processor_x86_32)
        return static_cast<_ReduceType>(_mm_cvtsi128_si32(_IntrinBitcast<__m128i>(_Vector)) + _SimdExtract<_Generation, _RegisterPolicy, int32>(_Vector, 2));
#else 
        return static_cast<_ReduceType>(_mm_cvtsi128_si64(__intrin_bitcast<__m128i>(_Vector)) +
            _SimdExtract<_Generation, _RegisterPolicy, int64>(_Vector, 1));
#endif // defined(simd_stl_processor_x86_32)
    }
    else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>) {
        const auto _FirstReduce = _mm_sad_epu8(__intrin_bitcast<__m128i>(_Vector), _mm_setzero_si128());
#if defined(simd_stl_processor_x86_32)
        return static_cast<_ReduceType>(_mm_cvtsi128_si32(_FirstReduce) 
            + _SimdExtract<_Generation, _RegisterPolicy, int32>(_FirstReduce, 2));
#else
        return static_cast<_ReduceType>(_mm_cvtsi128_si64(__intrin_bitcast<__m128i>(_FirstReduce))
            + _SimdExtract<_Generation, _RegisterPolicy, int64>(_FirstReduce, 1));
#endif // defined(simd_stl_processor_x86_32)
    }
    else
{
        constexpr auto _Length = sizeof(_VectorType_) / sizeof(_DesiredType_);

        _DesiredType_ _Array[_Length];
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&_Array), __intrin_bitcast<__m128i>(_Vector));

        _ReduceType _Sum = 0;

        for (auto _Index = 0; _Index < _Length; ++_Index)
            _Sum += _Array[_Index];

        return _Sum;
    }
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::SSE2, xmm128>::_ShiftRightVector(
    _VectorType_    _Vector,
    uint32          _ByteShift) noexcept
{
    return __intrin_bitcast<_VectorType_>(_mm_srli_si128(__intrin_bitcast<__m128i>(_Vector), _ByteShift));
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::SSE2, xmm128>::_ShiftLeftVector(
    _VectorType_    _Vector,
    uint32          _ByteShift) noexcept
{
    return __intrin_bitcast<_VectorType_>(_mm_slli_si128(__intrin_bitcast<__m128i>(_Vector), _ByteShift));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::SSE2, xmm128>::_ShiftRightElements(
    _VectorType_    _Vector,
    uint32          _BitShift) noexcept
{
    if      constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_srli_epi64(__intrin_bitcast<__m128i>(_Vector), _BitShift));

    else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_srli_epi32(__intrin_bitcast<__m128i>(_Vector), _BitShift));

    else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_srli_epi16(__intrin_bitcast<__m128i>(_Vector), _BitShift));

    else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>) {
        const auto _EvenVector = _mm_sra_epi16(_mm_slli_epi16(__intrin_bitcast<__m128i>(_Vector), 8), _mm_cvtsi32_si128(_BitShift + 8));
        const auto _OddVector = _mm_sra_epi16(__intrin_bitcast<__m128i>(_Vector), _mm_cvtsi32_si128(_BitShift));

        const auto _Mask = _mm_set1_epi32(0x00FF00FF);
        return __intrin_bitcast<_VectorType_>(_mm_or_si128(_mm_and_si128(_Mask, _EvenVector), _mm_andnot_si128(_Mask, _OddVector)));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::SSE2, xmm128>::_ShiftLeftElements(
    _VectorType_    _Vector,
    uint32          _BitShift) noexcept
{
    if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_slli_epi64(__intrin_bitcast<__m128i>(_Vector), _BitShift));

    else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_slli_epi32(__intrin_bitcast<__m128i>(_Vector), _BitShift));

    else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_slli_epi16(__intrin_bitcast<__m128i>(_Vector), _BitShift));

    else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>) {
        const auto _AndMask = _mm_and_si128(__intrin_bitcast<__m128i>(_Vector),
            _mm_set1_epi8(static_cast<int8>(0xFFu >> _BitShift)));

        return __intrin_bitcast<_VectorType_>(_mm_sll_epi16(_AndMask, _mm_cvtsi32_si128(_BitShift)));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::SSE2, xmm128>::_Negate(_VectorType_ _Vector) noexcept {
    if      constexpr (_Is_ps_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_xor_ps(_Vector, _mm_set1_ps(-0.0f)));

    else if constexpr (_Is_pd_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_xor_pd(_Vector,
            __intrin_bitcast<__m128d>(_mm_setr_epi32(0, 0x80000000, 0, 0x80000000))));

    else
        return _Substract<_DesiredType_>(_SimdBroadcastZeros<_Generation, _RegisterPolicy, _VectorType_>(), _Vector);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::SSE2, xmm128>::_Add(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_add_epi64(__intrin_bitcast<__m128i>(_Left), __intrin_bitcast<__m128i>(_Right)));

    else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_add_epi32(__intrin_bitcast<__m128i>(_Left), __intrin_bitcast<__m128i>(_Right)));

    else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_add_epi16(__intrin_bitcast<__m128i>(_Left), __intrin_bitcast<__m128i>(_Right)));

    else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_add_epi8(__intrin_bitcast<__m128i>(_Left), __intrin_bitcast<__m128i>(_Right)));

    else if constexpr (_Is_ps_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_add_ps(__intrin_bitcast<__m128>(_Left), __intrin_bitcast<__m128>(_Right)));

    else if constexpr (_Is_pd_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_add_pd(__intrin_bitcast<__m128d>(_Left), __intrin_bitcast<__m128d>(_Right)));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::SSE2, xmm128>::_Substract(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_sub_epi64(__intrin_bitcast<__m128i>(_Left), __intrin_bitcast<__m128i>(_Right)));

    else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_sub_epi32(__intrin_bitcast<__m128i>(_Left), __intrin_bitcast<__m128i>(_Right)));

    else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_sub_epi16(__intrin_bitcast<__m128i>(_Left), __intrin_bitcast<__m128i>(_Right)));

    else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_sub_epi8(__intrin_bitcast<__m128i>(_Left), __intrin_bitcast<__m128i>(_Right)));

    else if constexpr (_Is_ps_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_sub_ps(__intrin_bitcast<__m128>(_Left), __intrin_bitcast<__m128>(_Right)));

    else if constexpr (_Is_pd_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_sub_pd(__intrin_bitcast<__m128d>(_Left), __intrin_bitcast<__m128d>(_Right)));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::SSE2, xmm128>::_Multiply(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if constexpr (_Is_epi32_v<_DesiredType_>) {
        const auto _ShuffledLeft = _mm_shuffle_epi32(__intrin_bitcast<__m128i>(_Left), 0xF5);
        const auto _ShuffledRight = _mm_shuffle_epi32(__intrin_bitcast<__m128i>(_Right), 0xF5);

        const auto _ProductEvenIndices = _mm_mul_epu32(_Left, _Right);
        const auto _ProductOddIndices = _mm_mul_epu32(_ShuffledLeft, _ShuffledRight);

        const auto _ProductLowPair = _mm_unpacklo_epi32(_ProductEvenIndices, _ProductOddIndices);
        const auto _ProductHighPair = _mm_unpackhi_epi32(_ProductEvenIndices, _ProductOddIndices);

        return __intrin_bitcast<_VectorType_>(_mm_unpacklo_epi64(_ProductLowPair, _ProductHighPair));
    }

    else if constexpr (_Is_epu32_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_mul_epu32(__intrin_bitcast<__m128i>(_Left), __intrin_bitcast<__m128i>(_Right)));

    else if constexpr (_Is_ps_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_mul_ps(__intrin_bitcast<__m128>(_Left), __intrin_bitcast<__m128>(_Right)));

    else if constexpr (_Is_pd_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_mul_pd(__intrin_bitcast<__m128d>(_Left), __intrin_bitcast<__m128d>(_Right)));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::SSE2, xmm128>::_Divide(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if constexpr (_Is_pd_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_div_pd(__intrin_bitcast<__m128d>(_Left), __intrin_bitcast<__m128d>(_Right)));

    else if constexpr (_Is_ps_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_div_ps(__intrin_bitcast<__m128>(_Left), __intrin_bitcast<__m128>(_Right)));
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::SSE2, xmm128>::_BitNot(_VectorType_ _Vector) noexcept {
    if      constexpr (std::is_same_v<_VectorType_, __m128d>)
        return _mm_xor_pd(_Vector, _mm_cmpeq_pd(_Vector, _Vector));

    else if constexpr (std::is_same_v<_VectorType_, __m128i>)
        return _mm_xor_si128(_Vector, _mm_cmpeq_epi32(_Vector, _Vector));

    else if constexpr (std::is_same_v<_VectorType_, __m128>)
        return _mm_xor_ps(_Vector, _mm_cmpeq_ps(_Vector, _Vector));
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::SSE2, xmm128>::_BitXor(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if      constexpr (std::is_same_v<_VectorType_, __m128d>)
        return _mm_xor_pd(_Left, _Right);

    else if constexpr (std::is_same_v<_VectorType_, __m128i>)
        return _mm_xor_si128(_Left, _Right);

    else if constexpr (std::is_same_v<_VectorType_, __m128>)
        return _mm_xor_ps(_Left, _Right);
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::SSE2, xmm128>::_BitAnd(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if      constexpr (std::is_same_v<_VectorType_, __m128d>)
        return _mm_and_pd(_Left, _Right);

    else if constexpr (std::is_same_v<_VectorType_, __m128i>)
        return _mm_and_si128(_Left, _Right);

    else if constexpr (std::is_same_v<_VectorType_, __m128>)
        return _mm_and_ps(_Left, _Right);
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::SSE2, xmm128>::_BitOr(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if      constexpr (std::is_same_v<_VectorType_, __m128d>)
        return _mm_or_pd(_Left, _Right);

    else if constexpr (std::is_same_v<_VectorType_, __m128i>)
        return _mm_or_si128(_Left, _Right);

    else if constexpr (std::is_same_v<_VectorType_, __m128>)
        return _mm_or_ps(_Left, _Right);

}

template <
    typename _DesiredType_,
    typename _VectorType_,
    typename _ReduceBinaryFunction_>
simd_stl_always_inline _DesiredType_ _SimdArithmetic<arch::CpuFeature::SSSE3, xmm128>::_HorizontalFold(
    _VectorType_            _Vector,
    _ReduceBinaryFunction_  _Reduce) noexcept
{
    if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_> || _Is_pd_v<_DesiredType_>) {
        auto _HorizontalFoldedValues = __intrin_bitcast<__m128d>(_Vector);

        const auto _Shuffled    = _mm_shuffle_pd(_HorizontalFoldedValues, _HorizontalFoldedValues, 1);
        _HorizontalFoldedValues = _Reduce(_Shuffled, _HorizontalFoldedValues);

        if constexpr (_Is_pd_v<_DesiredType_>)
            return _mm_cvtsd_f64(_HorizontalFoldedValues);
        else
            return _mm_cvtsi128_si64(__intrin_bitcast<__m128i>(_HorizontalFoldedValues));
    }
    else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_> || _Is_ps_v<_DesiredType_>) {
        auto _HorizontalFoldedValues = __intrin_bitcast<__m128i>(_Vector);

        const auto _Shuffled1   = _mm_shuffle_epi32(_HorizontalFoldedValues, 0x4E);
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, _Shuffled1);

        const auto _Shuffled2   = _mm_shuffle_epi32(_HorizontalFoldedValues, 0xB1);
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, _Shuffled2);

        if constexpr (_Is_ps_v<_DesiredType_>)
            return _mm_cvtss_f32(__intrin_bitcast<__m128>(_HorizontalFoldedValues));
        else
            return _mm_cvtsi128_si32(_HorizontalFoldedValues);
    }
    else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>) {
        const auto _ShuffleWords = _mm_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2);

        auto _HorizontalFoldedValues = __intrin_bitcast<__m128i>(_Vector);
        
        const auto _Shuffled1   = _mm_shuffle_epi32(_HorizontalFoldedValues, 0x4E);
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, _Shuffled1);

        const auto _Shuffle2    = _mm_shuffle_epi32(_HorizontalFoldedValues, 0xB1);
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, _Shuffle2);

        const auto _Shuffle3    = _mm_shuffle_epi8(_HorizontalFoldedValues, _ShuffleWords);
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, _Shuffle3);

        return _mm_cvtsi128_si32(_HorizontalFoldedValues);
    }
    else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>) {
        const auto _ShuffleBytes = _mm_set_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
        const auto _ShuffleWords = _mm_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2);

        auto _HorizontalFoldedValues = __intrin_bitcast<__m128i>(_Vector);

        const auto _Shuffled1   = _mm_shuffle_epi32(_HorizontalFoldedValues, 0x4E);
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, _Shuffled1);

        const auto _Shuffled2   = _mm_shuffle_epi32(_HorizontalFoldedValues, 0xB1);
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, _Shuffled2);

        const auto _Shuffled3   = _mm_shuffle_epi8(_HorizontalFoldedValues, _ShuffleWords);
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, _Shuffled3);

        const auto _Shuffled4   = _mm_shuffle_epi8(_HorizontalFoldedValues, _ShuffleBytes);
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, _Shuffled4);

        return _mm_cvtsi128_si32(_HorizontalFoldedValues);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_ _SimdArithmetic<arch::CpuFeature::SSSE3, xmm128>::_HorizontalMin(_VectorType_ _Vector) noexcept {
    return _HorizontalFold<_DesiredType_>(_Vector, _VerticalMinWrapper<_Generation, _RegisterPolicy, _DesiredType_>{});
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_ _SimdArithmetic<arch::CpuFeature::SSSE3, xmm128>::_HorizontalMax(_VectorType_ _Vector) noexcept {
    return _HorizontalFold<_DesiredType_>(_Vector, _VerticalMaxWrapper<_Generation, _RegisterPolicy, _DesiredType_>{});
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline auto _SimdArithmetic<arch::CpuFeature::SSSE3, xmm128>::_Reduce(_VectorType_ _Vector) noexcept {
    using _ReduceType = _Reduce_type<_DesiredType_>;

    if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>) {
        const auto _Zeros = _mm_setzero_si128();

        const auto _Reduce4 = _mm_hadd_epi32(__intrin_bitcast<__m128i>(_Vector), _Zeros); // (0+1),(2+3),0,0
        const auto _Reduce5 = _mm_hadd_epi32(_Reduce4, _Zeros);                         // (0+...+3),0,0,0

        return static_cast<_ReduceType>(_mm_cvtsi128_si32(_Reduce5));
    }
    else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>) {
        const auto _Zeros = _mm_setzero_si128();

        const auto _Reduce2 = _mm_hadd_epi16(__intrin_bitcast<__m128i>(_Vector), _Zeros);
        const auto _Reduce3 = _mm_unpacklo_epi16(_Reduce2, _Zeros);

        const auto _Reduce4 = _mm_hadd_epi32(_Reduce3, _Zeros); // (0+1),(2+3),0,0
        const auto _Reduce5 = _mm_hadd_epi32(_Reduce4, _Zeros); // (0+...+3),0,0,0

        return static_cast<_ReduceType>(_mm_cvtsi128_si32(_Reduce5));
    }
    else {
        return _SimdReduce<arch::CpuFeature::SSE2, _RegisterPolicy, _DesiredType_>(_Vector);
    }
}


template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::SSE41, xmm128>::_Multiply(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if constexpr (_Is_epi32_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_mul_epi32(__intrin_bitcast<__m128i>(_Left), __intrin_bitcast<__m128i>(_Right)));

    else if constexpr (_Is_epu32_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_mul_epu32(__intrin_bitcast<__m128i>(_Left), __intrin_bitcast<__m128i>(_Right)));

    else if constexpr (_Is_ps_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_mul_ps(__intrin_bitcast<__m128>(_Left), __intrin_bitcast<__m128>(_Right)));

    else if constexpr (_Is_pd_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_mul_pd(__intrin_bitcast<__m128d>(_Left), __intrin_bitcast<__m128d>(_Right)));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_ _SimdArithmetic<arch::CpuFeature::SSE41, xmm128>::_HorizontalMin(_VectorType_ _Vector) noexcept {
    return _HorizontalFold<_DesiredType_>(_Vector, _VerticalMinWrapper<_Generation, _RegisterPolicy, _DesiredType_>{});
}

template <
    typename _DesiredType_,
    typename _VectorType_>
static simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::SSE41, xmm128>::_VerticalMin(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if constexpr (_Is_epi32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_min_epi32(__intrin_bitcast<__m128i>(_Left), __intrin_bitcast<__m128i>(_Right)));
    }
    else if constexpr (_Is_epu32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_min_epu32(__intrin_bitcast<__m128i>(_Left), __intrin_bitcast<__m128i>(_Right)));
    }
    else if constexpr (_Is_epu16_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_min_epu16(__intrin_bitcast<__m128i>(_Left), __intrin_bitcast<__m128i>(_Right)));
    }
    else if constexpr (_Is_epi8_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_min_epi8(__intrin_bitcast<__m128i>(_Left), __intrin_bitcast<__m128i>(_Right)));
    }
    else {
        return _SimdVerticalMin<arch::CpuFeature::SSE2, _RegisterPolicy, _DesiredType_>(_Left, _Right);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
static simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::SSE41, xmm128>::_VerticalMax(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if constexpr (_Is_epi32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_max_epi32(__intrin_bitcast<__m128i>(_Left), __intrin_bitcast<__m128i>(_Right)));
    }
    else if constexpr (_Is_epu32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_max_epu32(__intrin_bitcast<__m128i>(_Left), __intrin_bitcast<__m128i>(_Right)));
    }
    else if constexpr (_Is_epu16_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_max_epu16(__intrin_bitcast<__m128i>(_Left), __intrin_bitcast<__m128i>(_Right)));
    }
    else if constexpr (_Is_epi8_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_max_epi8(__intrin_bitcast<__m128i>(_Left), __intrin_bitcast<__m128i>(_Right)));
    }
    else {
        return _SimdVerticalMax<arch::CpuFeature::SSE2, _RegisterPolicy, _DesiredType_>(_Left, _Right);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_ _SimdArithmetic<arch::CpuFeature::SSE41, xmm128>::_HorizontalMax(_VectorType_ _Vector) noexcept {
    return _HorizontalFold<_DesiredType_>(_Vector, _VerticalMaxWrapper<_Generation, _RegisterPolicy, _DesiredType_>{});
}


#pragma endregion 

#pragma region Avx Simd arithmetic

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX2, ymm256>::_Abs(_VectorType_ _Vector) noexcept {
    if constexpr (std::is_unsigned_v<_DesiredType_>) {
        return _Vector;
    }
    else if constexpr (_Is_epi64_v<_DesiredType_>) {
        const auto _Sign        = _mm256_cmpgt_epi64(_mm256_setzero_si256(), _Vector);
        const auto _Inverted    = _mm256_xor_si256(__intrin_bitcast<__m256i>(_Vector), _Sign);

        return __intrin_bitcast<_VectorType_>(_mm256_sub_epi64(_Inverted, _Sign));
    }
    else if constexpr (_Is_epi32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_abs_epi32(__intrin_bitcast<__m256i>(_Vector)));
    }
    else if constexpr (_Is_epi16_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_abs_epi16(__intrin_bitcast<__m256i>(_Vector)));
    }
    else if constexpr (_Is_epi8_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_abs_epi8(__intrin_bitcast<__m256i>(_Vector)));
    }
    else if constexpr (_Is_pd_v<_DesiredType_>) {
        const auto _Mask = _mm256_set_epi32(0xFFFFFFFFu, 0x7FFFFFFFu, 0xFFFFFFFFu, 0x7FFFFFFFu,
            0xFFFFFFFFu, 0x7FFFFFFFu, 0xFFFFFFFFu, 0x7FFFFFFFu);
        return __intrin_bitcast<_VectorType_>(_mm256_and_pd(__intrin_bitcast<__m256d>(_Vector), __intrin_bitcast<__m256d>(_Mask)));
    }
    else if constexpr (_Is_ps_v<_DesiredType_>) {
        const auto _Mask = _mm256_set1_epi32(0x7FFFFFFFu);
        return __intrin_bitcast<_VectorType_>(_mm256_and_ps(__intrin_bitcast<__m256>(_Vector), __intrin_bitcast<__m256>(_Mask)));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX2, ymm256>
    ::_AdjustToUnsigned(_VectorType_ _Vector) noexcept
{

}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX2, ymm256>::_VerticalMin(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if constexpr (_Is_epi32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_min_epi32(
            __intrin_bitcast<__m256i>(_Left), __intrin_bitcast<__m256i>(_Right)));
    }
    else if constexpr (_Is_epu32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_min_epu32(
            __intrin_bitcast<__m256i>(_Left), __intrin_bitcast<__m256i>(_Right)));
    }
    else if constexpr (_Is_epi16_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_min_epi16(
            __intrin_bitcast<__m256i>(_Left), __intrin_bitcast<__m256i>(_Right)));
    }
    else if constexpr (_Is_epu16_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_min_epu16(
            __intrin_bitcast<__m256i>(_Left), __intrin_bitcast<__m256i>(_Right)));
    }
    else if constexpr (_Is_epi8_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_min_epi8(
            __intrin_bitcast<__m256i>(_Left), __intrin_bitcast<__m256i>(_Right)));
    }
    else if constexpr (_Is_epu8_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_min_epu8(
            __intrin_bitcast<__m256i>(_Left), __intrin_bitcast<__m256i>(_Right)));
    }
    else if constexpr (_Is_ps_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_min_ps(
            __intrin_bitcast<__m256>(_Left), __intrin_bitcast<__m256>(_Right)));
    }
    else if constexpr (_Is_ps_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_min_pd(
            __intrin_bitcast<__m256d>(_Left), __intrin_bitcast<__m256d>(_Right)));
    }
    else {
        const auto _Mask = _SimdCompare<_Generation, _RegisterPolicy, _DesiredType_, type_traits::less<>>(_Left, _Right);
        return _SimdBlend<_Generation, _RegisterPolicy, _DesiredType_>(_Left, _Right, _Mask);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_ _SimdArithmetic<arch::CpuFeature::AVX2, ymm256>::_HorizontalMin(_VectorType_ _Vector) noexcept {
    return _HorizontalFold<_DesiredType_>(_Vector, _VerticalMinWrapper<_Generation, _RegisterPolicy, _DesiredType_>{});
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX2, ymm256>::_VerticalMax(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if constexpr (_Is_epi32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_max_epi32(
            __intrin_bitcast<__m256i>(_Left), __intrin_bitcast<__m256i>(_Right)));
    }
    else if constexpr (_Is_epu32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_max_epu32(
            __intrin_bitcast<__m256i>(_Left), __intrin_bitcast<__m256i>(_Right)));
    }
    else if constexpr (_Is_epi16_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_max_epi16(
            __intrin_bitcast<__m256i>(_Left), __intrin_bitcast<__m256i>(_Right)));
    }
    else if constexpr (_Is_epu16_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_max_epu16(
            __intrin_bitcast<__m256i>(_Left), __intrin_bitcast<__m256i>(_Right)));
    }
    else if constexpr (_Is_epi8_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_max_epi8(
            __intrin_bitcast<__m256i>(_Left), __intrin_bitcast<__m256i>(_Right)));
    }
    else if constexpr (_Is_epu8_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_max_epu8(
            __intrin_bitcast<__m256i>(_Left), __intrin_bitcast<__m256i>(_Right)));
    }
    else if constexpr (_Is_ps_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_max_ps(
            __intrin_bitcast<__m256>(_Left), __intrin_bitcast<__m256>(_Right)));
    }
    else if constexpr (_Is_ps_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_max_pd(
            __intrin_bitcast<__m256d>(_Left), __intrin_bitcast<__m256d>(_Right)));
    }
    else {
        const auto _Mask = _SimdCompare<_Generation, _RegisterPolicy, _DesiredType_, type_traits::greater<>>(_Left, _Right);
        return _SimdBlend<_Generation, _RegisterPolicy, _DesiredType_>(_Left, _Right, _Mask);
    }
}


template <
    typename _DesiredType_,
    typename _VectorType_,
    typename _ReduceBinaryFunction_>
simd_stl_always_inline _DesiredType_ _SimdArithmetic<arch::CpuFeature::AVX2, ymm256>::_HorizontalFold(
    _VectorType_            _Vector,
    _ReduceBinaryFunction_  _Reduce) noexcept
{
    if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_> || _Is_pd_v<_DesiredType_>) {
        auto _HorizontalFoldedValues = __intrin_bitcast<__m256d>(_Vector);
        
        const auto _Shuffled1   = _mm256_shuffle_pd(_HorizontalFoldedValues, _HorizontalFoldedValues, 5);
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, _Shuffled1);

        const auto _Shuffled2   = _mm256_permute4x64_epi64(__intrin_bitcast<__m256i>(_HorizontalFoldedValues), 0x1B);
        _HorizontalFoldedValues = __intrin_bitcast<__m256d>(_Reduce(__intrin_bitcast<__m256i>(_HorizontalFoldedValues), __intrin_bitcast<__m256i>(_Shuffled2)));

        if constexpr (_Is_pd_v<_DesiredType_>)
            return _mm256_cvtsd_f64(_HorizontalFoldedValues);
        else 
            return _mm256_cvtsi256_si64(__intrin_bitcast<__m256i>(_HorizontalFoldedValues));
    }
    else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_> || _Is_ps_v<_DesiredType_>) {
        auto _HorizontalMaximumValues = __intrin_bitcast<__m256i>(_Vector);

        const auto _Shuffle1        = _mm256_permute4x64_epi64(_HorizontalMaximumValues, 0x4E);
        _HorizontalMaximumValues    = _Reduce(_HorizontalMaximumValues, _Shuffle1);

        const auto _Shuffle2        = _mm256_shuffle_epi32(_HorizontalMaximumValues, 0x4E);
        _HorizontalMaximumValues    = _Reduce(_HorizontalMaximumValues, _Shuffle2);

        const auto _Shuffle3        = _mm256_shuffle_epi32(_HorizontalMaximumValues, 0xB1);
        _HorizontalMaximumValues    = _Reduce(_HorizontalMaximumValues, _Shuffle3);

        if constexpr (_Is_ps_v<_DesiredType_>)
            return _mm256_cvtss_f32(__intrin_bitcast<__m256>(_HorizontalMaximumValues));
        else 
            return _mm256_cvtsi256_si32(_HorizontalMaximumValues);
    }
    else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>) {
        const auto _ShuffleWords = _mm256_broadcastsi128_si256(_mm_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2));

        auto _HorizontalMaximumValues = __intrin_bitcast<__m256i>(_Vector);

        const auto _Shuffle1        = _mm256_permute4x64_epi64(_HorizontalMaximumValues, 0x4E);
        _HorizontalMaximumValues    = _Reduce(_HorizontalMaximumValues, _Shuffle1);

        const auto _Shuffle2        = _mm256_shuffle_epi32(_HorizontalMaximumValues, 0x4E);
        _HorizontalMaximumValues    = _Reduce(_HorizontalMaximumValues, _Shuffle2);

        const auto _Shuffle3        = _mm256_shuffle_epi32(_HorizontalMaximumValues, 0xB1);
        _HorizontalMaximumValues    = _Reduce(_HorizontalMaximumValues, _Shuffle3);

        const auto _Shuffle4        = _mm256_shuffle_epi8(_HorizontalMaximumValues, _ShuffleWords);
        _HorizontalMaximumValues    = _Reduce(_HorizontalMaximumValues, _Shuffle4);

        return _mm256_cvtsi256_si32(_HorizontalMaximumValues);
    }
    else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>) {
        const auto _ShuffleWords = _mm256_broadcastsi128_si256(_mm_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2));
        const auto _ShuffleBytes = _mm256_broadcastsi128_si256(_mm_set_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1));

        auto _HorizontalMaximumValues = __intrin_bitcast<__m256i>(_Vector);
      
        const auto _Shuffle1        = _mm256_permute4x64_epi64(_HorizontalMaximumValues, 0x4E);
        _HorizontalMaximumValues    = _Reduce(_HorizontalMaximumValues, _Shuffle1);

        const auto _Shuffle2        = _mm256_shuffle_epi32(_HorizontalMaximumValues, 0x4E);
        _HorizontalMaximumValues    = _Reduce(_HorizontalMaximumValues, _Shuffle2);

        const auto _Shuffle3        = _mm256_shuffle_epi32(_HorizontalMaximumValues, 0xB1);
        _HorizontalMaximumValues    = _Reduce(_HorizontalMaximumValues, _Shuffle3);

        const auto _Shuffle4        = _mm256_shuffle_epi8(_HorizontalMaximumValues, _ShuffleWords);
        _HorizontalMaximumValues    = _Reduce(_HorizontalMaximumValues, _Shuffle4);

        const auto _Shuffle5        = _mm256_shuffle_epi8(_HorizontalMaximumValues, _ShuffleBytes);
        _HorizontalMaximumValues    = _Reduce(_HorizontalMaximumValues, _Shuffle5);

        return _mm256_cvtsi256_si32(_HorizontalMaximumValues);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_ _SimdArithmetic<arch::CpuFeature::AVX2, ymm256>::_HorizontalMax(_VectorType_ _Vector) noexcept {
    return _HorizontalFold<_DesiredType_>(_Vector, _VerticalMaxWrapper<_Generation, _RegisterPolicy, _DesiredType_>{});
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline auto _SimdArithmetic<arch::CpuFeature::AVX2, ymm256>::_Reduce(_VectorType_ _Vector) noexcept {
    using _ReduceType = _Reduce_type<_DesiredType_>;

    if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>) {
        const auto _Low64   = __intrin_bitcast<__m128i>(_Vector);
        const auto _High64  = _mm256_extracti128_si256(__intrin_bitcast<__m256i>(_Vector), 1);

        const auto _Reduce  = _mm_add_epi64(_Low64, _High64);
        return _SimdReduce<arch::CpuFeature::SSSE3, xmm128, _DesiredType_>(_Reduce);
    }
    if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>) {
        const auto _Zeros = _mm256_setzero_si256();

        const auto _Reduce4 = _mm256_hadd_epi32(__intrin_bitcast<__m256i>(_Vector), _Zeros); // (0+1),(2+3),0,0 per lane
        const auto _Reduce5 = _mm256_permute4x64_epi64(_Reduce4, 0xD8); // low lane  (0+1),(2+3),(4+5),(6+7)

        const auto _Reduce6 = _mm256_hadd_epi32(_Reduce5, _Zeros); // (0+...+3),(4+...+7),0,0
        const auto _Reduce7 = _mm256_hadd_epi32(_Reduce6, _Zeros); // (0+...+7),0,0,0

        return static_cast<_ReduceType>(_mm_cvtsi128_si32(__intrin_bitcast<__m128i>(_Reduce7)));
    }
    else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>) {
        const auto _Zeros = _mm256_setzero_si256();

        const auto _Reduce2 = _mm256_hadd_epi16(__intrin_bitcast<__m256i>(_Vector), _Zeros);
        const auto _Reduce3 = _mm256_unpacklo_epi16(_Reduce2, _Zeros);

        const auto _Reduce4 = _mm256_hadd_epi32(_Reduce3, _Zeros); // (0+1),(2+3),0,0 per lane
        const auto _Reduce5 = _mm256_permute4x64_epi64(_Reduce4, 0xD8); // low lane  (0+1),(2+3),(4+5),(6+7)

        const auto _Reduce6 = _mm256_hadd_epi32(_Reduce5, _Zeros); // (0+...+3),(4+...+7),0,0
        const auto _Reduce7 = _mm256_hadd_epi32(_Reduce6, _Zeros); // (0+...+7),0,0,0

        return static_cast<_ReduceType>(_mm_cvtsi128_si32(__intrin_bitcast<__m128i>(_Reduce7)));
    }
    else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>) {
        const auto _Reduce1 = _mm256_sad_epu8(__intrin_bitcast<__m256i>(_Vector), _mm256_setzero_si256());

        const auto _Low64 = _mm256_castsi256_si128(_Reduce1);
        const auto _High64 = _mm256_extracti128_si256(_Reduce1, 1);

        const auto _Reduce8 = _mm_add_epi64(_Low64, _High64);
        return _SimdReduce<arch::CpuFeature::SSSE3, xmm128, int64>(_Reduce8);
    }
    else {
        constexpr auto _Length = sizeof(_VectorType_) / sizeof(_DesiredType_);

        _DesiredType_ _Array[_Length];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&_Array), __intrin_bitcast<__m256i>(_Vector));

        _ReduceType _Sum = 0;

        for (auto _Index = 0; _Index < _Length; ++_Index)
            _Sum += _Array[_Index];

        return _Sum;
    }
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX2, ymm256>::_ShiftRightVector(
    _VectorType_    _Vector,
    uint32          _ByteShift) noexcept
{
    return __intrin_bitcast<_VectorType_>(_mm256_srli_si256(__intrin_bitcast<__m256i>(_Vector), _ByteShift));
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX2, ymm256>::_ShiftLeftVector(
    _VectorType_    _Vector,
    uint32          _ByteShift) noexcept
{
    return __intrin_bitcast<_VectorType_>(_mm256_slli_si256(__intrin_bitcast<__m256i>(_Vector), _ByteShift));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX2, ymm256>::_ShiftRightElements(
    _VectorType_    _Vector,
    uint32          _BitShift) noexcept
{
    if      constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_srli_epi64(__intrin_bitcast<__m256i>(_Vector), _BitShift));

    else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_srli_epi32(__intrin_bitcast<__m256i>(_Vector), _BitShift));

    else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_srli_epi16(__intrin_bitcast<__m256i>(_Vector), _BitShift));

    else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>) {
        const auto _EvenVector = _mm256_sra_epi16(_mm256_slli_epi16(
            __intrin_bitcast<__m256i>(_Vector), 8), _mm_cvtsi32_si128(_BitShift + 8));

        const auto _OddVector = _mm256_sra_epi16(__intrin_bitcast<__m256i>(_Vector), _mm_cvtsi32_si128(_BitShift));

        const auto _Mask = _mm256_set1_epi32(0x00FF00FF);
        return _SimdBlend<_Generation, _RegisterPolicy, _DesiredType_>(_EvenVector, _OddVector, _Mask);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX2, ymm256>::_ShiftLeftElements(
    _VectorType_    _Vector,
    uint32          _BitShift) noexcept
{
    if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_slli_epi64(__intrin_bitcast<__m256i>(_Vector), _BitShift));

    else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_slli_epi32(__intrin_bitcast<__m256i>(_Vector), _BitShift));

    else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_slli_epi16(__intrin_bitcast<__m256i>(_Vector), _BitShift));

    else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>) {
        const auto _AndMask = _mm256_and_si256(__intrin_bitcast<__m256i>(_Vector),
            _mm256_set1_epi8(static_cast<int8>(0xFFu >> _BitShift)));

        return __intrin_bitcast<_VectorType_>(_mm256_sll_epi16(_AndMask, _mm_cvtsi32_si128(_BitShift)));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX2, ymm256>::_Negate(_VectorType_ _Vector) noexcept {
    if      constexpr (_Is_ps_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_xor_ps(__intrin_bitcast<__m256>(_Vector), _mm256_set1_ps(-0.0f)));

    else if constexpr (_Is_pd_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_xor_pd(__intrin_bitcast<__m256d>(_Vector),
            __intrin_bitcast<__m256d>(_mm256_setr_epi32(0, 0x80000000, 0, 0x80000000, 0, 0x80000000, 0, 0x80000000))));

    else
        return _Substract<_DesiredType_>(_SimdBroadcastZeros<_Generation, _RegisterPolicy, _VectorType_>(), _Vector);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX2, ymm256>::_Add(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_add_epi64(__intrin_bitcast<__m256i>(_Left), __intrin_bitcast<__m256i>(_Right)));

    else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_add_epi32(__intrin_bitcast<__m256i>(_Left), __intrin_bitcast<__m256i>(_Right)));

    else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_add_epi16(__intrin_bitcast<__m256i>(_Left), __intrin_bitcast<__m256i>(_Right)));

    else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_add_epi8(__intrin_bitcast<__m256i>(_Left), __intrin_bitcast<__m256i>(_Right)));

    else if constexpr (_Is_ps_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_add_ps(__intrin_bitcast<__m256>(_Left), __intrin_bitcast<__m256>(_Right)));

    else if constexpr (_Is_pd_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_add_pd(__intrin_bitcast<__m256d>(_Left), __intrin_bitcast<__m256d>(_Right)));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX2, ymm256>::_Substract(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_sub_epi64(__intrin_bitcast<__m256i>(_Left), __intrin_bitcast<__m256i>(_Right)));

    else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_sub_epi32(__intrin_bitcast<__m256i>(_Left), __intrin_bitcast<__m256i>(_Right)));

    else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_sub_epi16(__intrin_bitcast<__m256i>(_Left), __intrin_bitcast<__m256i>(_Right)));

    else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_sub_epi8(__intrin_bitcast<__m256i>(_Left), __intrin_bitcast<__m256i>(_Right)));

    else if constexpr (_Is_ps_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_sub_ps(__intrin_bitcast<__m256>(_Left), __intrin_bitcast<__m256>(_Right)));

    else if constexpr (_Is_pd_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_sub_pd(__intrin_bitcast<__m256d>(_Left), __intrin_bitcast<__m256d>(_Right)));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX2, ymm256>::_Multiply(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if constexpr (_Is_epi32_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_mul_epi32(__intrin_bitcast<__m256i>(_Left), __intrin_bitcast<__m256i>(_Right)));

    else if constexpr (_Is_epu32_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_mul_epu32(__intrin_bitcast<__m256i>(_Left), __intrin_bitcast<__m256i>(_Right)));

    else if constexpr (_Is_ps_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_mul_ps(__intrin_bitcast<__m256>(_Left), __intrin_bitcast<__m256>(_Right)));

    else if constexpr (_Is_pd_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_mul_pd(__intrin_bitcast<__m256d>(_Left), __intrin_bitcast<__m256d>(_Right)));

    else {

    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX2, ymm256>::_Divide(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if constexpr (_Is_pd_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_div_pd(__intrin_bitcast<__m256d>(_Left), __intrin_bitcast<__m256d>(_Right)));

    else if constexpr (_Is_ps_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_div_ps(__intrin_bitcast<__m256>(_Left), __intrin_bitcast<__m256>(_Right)));
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX2, ymm256>::_BitNot(_VectorType_ _Vector) noexcept {
    if      constexpr (std::is_same_v<_VectorType_, __m256d>)
        return _mm256_xor_pd(_Vector, _mm256_cmp_pd(_Vector, _Vector, _CMP_EQ_OQ));

    else if constexpr (std::is_same_v<_VectorType_, __m256i>)
        return _mm256_xor_si256(_Vector, _mm256_cmpeq_epi32(_Vector, _Vector));

    else if constexpr (std::is_same_v<_VectorType_, __m256>)
        return _mm256_xor_ps(_Vector, _mm256_cmp_ps(_Vector, _Vector, _CMP_EQ_OQ));
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX2, ymm256>::_BitXor(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if      constexpr (std::is_same_v<_VectorType_, __m256d>)
        return _mm256_xor_pd(_Left, _Right);

    else if constexpr (std::is_same_v<_VectorType_, __m256i>)
        return _mm256_xor_si256(_Left, _Right);

    else if constexpr (std::is_same_v<_VectorType_, __m256>)
        return _mm256_xor_ps(_Left, _Right);
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX2, ymm256>::_BitAnd(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if      constexpr (std::is_same_v<_VectorType_, __m256d>)
        return _mm256_and_pd(_Left, _Right);

    else if constexpr (std::is_same_v<_VectorType_, __m256i>)
        return _mm256_and_si256(_Left, _Right);

    else if constexpr (std::is_same_v<_VectorType_, __m256>)
        return _mm256_and_ps(_Left, _Right);
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX2, ymm256>::_BitOr(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if      constexpr (std::is_same_v<_VectorType_, __m256d>)
        return _mm256_or_pd(_Left, _Right);

    else if constexpr (std::is_same_v<_VectorType_, __m256i>)
        return _mm256_or_si256(_Left, _Right);

    else if constexpr (std::is_same_v<_VectorType_, __m256>)
        return _mm256_or_ps(_Left, _Right);
}

#pragma endregion

#pragma region Avx512 Simd arithmetic

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX512F, zmm512>
    ::_AdjustToUnsigned(_VectorType_ _Vector) noexcept
{

}

template <
    typename _DesiredType_,
    typename _VectorType_,
    typename _ReduceBinaryFunction_>
simd_stl_always_inline _DesiredType_ _SimdArithmetic<arch::CpuFeature::AVX512F, zmm512>::_HorizontalFold(
    _VectorType_            _Vector,
    _ReduceBinaryFunction_  _Reduce) noexcept
{
    if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_> || _Is_pd_v<_DesiredType_>) {
        auto _HorizontalFoldedValues = __intrin_bitcast<__m512i>(_Vector);

        const auto _Shuffled1   = _mm512_permutexvar_epi64(_mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0), _HorizontalFoldedValues);
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, _Shuffled1);

        const auto _Shuffled2   = _mm512_permutex_epi64(_HorizontalFoldedValues, 0x4E);
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, _Shuffled2);

        const auto _Shuffled3   = _mm512_permutex_epi64(_HorizontalFoldedValues, 0xB1);
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, _Shuffled3);

        if constexpr (_Is_pd_v<_DesiredType_>)
            return _mm512_cvtsd_f64(__intrin_bitcast<__m512d>(_HorizontalFoldedValues));
        else
            return _mm512_cvtsi512_si64(_HorizontalFoldedValues);
    }
    else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_> || _Is_ps_v<_DesiredType_>) {
        auto _HorizontalFoldedValues = __intrin_bitcast<__m512i>(_Vector);

        const auto _Shuffled1   = _mm512_permutexvar_epi64(_mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0), _HorizontalFoldedValues);
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, _Shuffled1);

        const auto _Shuffled2   = _mm512_permutex_epi64(_HorizontalFoldedValues, 0x4E);
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, _Shuffled2);

        const auto _Shuffled3   = _mm512_permutex_epi64(_HorizontalFoldedValues, 0xB1);
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, _Shuffled3);

        const auto _Shuffled4   = __intrin_bitcast<__m512i>(_mm512_permute_ps(__intrin_bitcast<__m512>(_HorizontalFoldedValues), 0xB1));
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, _Shuffled4);

        if constexpr (_Is_ps_v<_DesiredType_>)
            return _mm512_cvtss_f32(__intrin_bitcast<__m512>(_HorizontalFoldedValues));
        else
            return _mm512_cvtsi512_si32(_HorizontalFoldedValues);
    }
    else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>) {
        const auto _ShuffleWords = _mm256_broadcastsi128_si256(_mm_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2));
        auto _HorizontalFoldedValues = __intrin_bitcast<__m512i>(_Vector);

        const auto _Shuffled1   = _mm512_permutexvar_epi64(_mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0), _HorizontalFoldedValues);
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, _Shuffled1);

        const auto _Shuffled2   = _mm512_permutex_epi64(_HorizontalFoldedValues, 0x4E);
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, _Shuffled2);

        const auto _Shuffled3   = _mm512_permutex_epi64(_HorizontalFoldedValues, 0xB1);
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, _Shuffled3);

        const auto _Shuffled4   = __intrin_bitcast<__m512i>(_mm512_permute_ps(__intrin_bitcast<__m512>(_HorizontalFoldedValues), 0xB1));
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, _Shuffled4);

        const auto _Shuffled5Low    = _mm256_shuffle_epi8(__intrin_bitcast<__m256i>(_HorizontalFoldedValues), _ShuffleWords);
        const auto _Shuffled5High   = _mm256_shuffle_epi8(__intrin_bitcast<__m256i>(_mm512_extractf64x4_pd(
            __intrin_bitcast<__m512d>(_HorizontalFoldedValues), 1)), _ShuffleWords);
        
        auto _Shuffled5 = __intrin_bitcast<__m512d>(_Shuffled5Low);
        _Shuffled5 = _mm512_insertf64x4(_Shuffled5, __intrin_bitcast<__m256d>(_Shuffled5High), 1);

        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, __intrin_bitcast<__m512i>(_Shuffled5));

        return _mm512_cvtsi512_si32(_HorizontalFoldedValues);
    }
    else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>) {
        const auto _ShuffleWords = _mm256_broadcastsi128_si256(_mm_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2));
        const auto _ShuffleBytes = _mm256_broadcastsi128_si256(_mm_set_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1));

        auto _HorizontalFoldedValues = __intrin_bitcast<__m512i>(_Vector);

        const auto _Shuffled1   = _mm512_permutexvar_epi64(_mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0), _HorizontalFoldedValues);
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, _Shuffled1);

        const auto _Shuffled2   = _mm512_permutex_epi64(_HorizontalFoldedValues, 0x4E);
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, _Shuffled2);

        const auto _Shuffled3   = _mm512_permutex_epi64(_HorizontalFoldedValues, 0xB1);
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, _Shuffled3);

        const auto _Shuffled4   = __intrin_bitcast<__m512i>(_mm512_permute_ps(__intrin_bitcast<__m512>(_HorizontalFoldedValues), 0xB1));
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, _Shuffled4);

        const auto _Shuffled5Low    = _mm256_shuffle_epi8(__intrin_bitcast<__m256i>(_HorizontalFoldedValues), _ShuffleWords);
        const auto _Shuffled5High   = _mm256_shuffle_epi8(__intrin_bitcast<__m256i>(_mm512_extractf64x4_pd(
            __intrin_bitcast<__m512d>(_HorizontalFoldedValues), 1)), _ShuffleWords);
        
        auto _Shuffled5 = __intrin_bitcast<__m512d>(_Shuffled5Low);
        _Shuffled5 = _mm512_insertf64x4(_Shuffled5, __intrin_bitcast<__m256d>(_Shuffled5High), 1);

        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, __intrin_bitcast<__m512i>(_Shuffled5));

        const auto _Shuffled6Low = _mm256_shuffle_epi8(__intrin_bitcast<__m256i>(_HorizontalFoldedValues), _ShuffleBytes);
        const auto _Shuffled6High = _mm256_shuffle_epi8(__intrin_bitcast<__m256i>(_mm512_extractf64x4_pd(
            __intrin_bitcast<__m512d>(_HorizontalFoldedValues), 1)), _ShuffleBytes);

        auto _Shuffled6 = __intrin_bitcast<__m512d>(_Shuffled6Low);
        _Shuffled6 = _mm512_insertf64x4(_Shuffled6, __intrin_bitcast<__m256d>(_Shuffled6High), 1);

        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, __intrin_bitcast<__m512i>(_Shuffled6));

        return _mm512_cvtsi512_si32(_HorizontalFoldedValues);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX512F, zmm512>::_VerticalMin(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if constexpr (_Is_epi64_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_min_epi64(__intrin_bitcast<__m512i>(_Left), __intrin_bitcast<__m512i>(_Right)));
    }
    else if constexpr (_Is_epu64_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_min_epu64(__intrin_bitcast<__m512i>(_Left), __intrin_bitcast<__m512i>(_Right)));
    }
    else if constexpr (_Is_epi32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_min_epi32(__intrin_bitcast<__m512i>(_Left), __intrin_bitcast<__m512i>(_Right)));
    }
    else if constexpr (_Is_epu32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_min_epu32(__intrin_bitcast<__m512i>(_Left), __intrin_bitcast<__m512i>(_Right)));
    }
    else if constexpr (_Is_ps_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_min_ps(__intrin_bitcast<__m512>(_Left), __intrin_bitcast<__m512>(_Right)));
    }
    else if constexpr (_Is_pd_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_min_pd(__intrin_bitcast<__m512d>(_Left), __intrin_bitcast<__m512d>(_Right)));
    }
    else {
        const auto _Mask = _SimdCompare<_Generation, _RegisterPolicy, _DesiredType_, type_traits::less<>>(_Left, _Right);
        return _SimdBlend<_Generation, _RegisterPolicy, _DesiredType_>(_Left, _Right, _Mask);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_ _SimdArithmetic<arch::CpuFeature::AVX512F, zmm512>::_HorizontalMin(_VectorType_ _Vector) noexcept {
    return _HorizontalFold<_DesiredType_>(_Vector, _VerticalMinWrapper<_Generation, _RegisterPolicy, _DesiredType_>{});
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX512F, zmm512>::_VerticalMax(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if constexpr (_Is_epi64_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_max_epi64(__intrin_bitcast<__m512i>(_Left), __intrin_bitcast<__m512i>(_Right)));
    }
    else if constexpr (_Is_epu64_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_max_epu64(__intrin_bitcast<__m512i>(_Left), __intrin_bitcast<__m512i>(_Right)));
    }
    else if constexpr (_Is_epi32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_max_epi32(__intrin_bitcast<__m512i>(_Left), __intrin_bitcast<__m512i>(_Right)));
    }
    else if constexpr (_Is_epu32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_max_epu32(__intrin_bitcast<__m512i>(_Left), __intrin_bitcast<__m512i>(_Right)));
    }
    else if constexpr (_Is_ps_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_max_ps(__intrin_bitcast<__m512>(_Left), __intrin_bitcast<__m512>(_Right)));
    }
    else if constexpr (_Is_pd_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_max_pd(__intrin_bitcast<__m512d>(_Left), __intrin_bitcast<__m512d>(_Right)));
    }
    else {
        const auto _Mask = _SimdCompare<_Generation, _RegisterPolicy, _DesiredType_, type_traits::greater<>>(_Left, _Right);
        return _SimdBlend<_Generation, _RegisterPolicy, _DesiredType_>(_Left, _Right, _Mask);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_ _SimdArithmetic<arch::CpuFeature::AVX512F, zmm512>::_HorizontalMax(_VectorType_ _Vector) noexcept {
    return _HorizontalFold<_DesiredType_>(_Vector, _VerticalMaxWrapper<_Generation, _RegisterPolicy, _DesiredType_>{});
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX512F, zmm512>::_Abs(_VectorType_ _Vector) noexcept {
    if constexpr (std::is_unsigned_v<_DesiredType_>) {
        return _Vector;
    }
    else if constexpr (_Is_epi64_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_abs_epi64(__intrin_bitcast<__m512i>(_Vector)));
    }
    else if constexpr (_Is_epi32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_abs_epi32(__intrin_bitcast<__m512i>(_Vector)));
    }
    else if constexpr (_Is_ps_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_abs_ps(__intrin_bitcast<__m512>(_Vector)));
    }
    else if constexpr (_Is_pd_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_abs_pd(__intrin_bitcast<__m512d>(_Vector)));
    }
    else {
        const auto _Low     = __intrin_bitcast<__m256i>(_Vector);
        const auto _High    = _mm512_extractf64x4_pd(__intrin_bitcast<__m512d>(_Vector), 1);

        auto _Result = __intrin_bitcast<__m512i>(_SimdAbs<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(_Low));
        _Result = __intrin_bitcast<__m512i>(_mm512_insertf64x4(__intrin_bitcast<__m512d>(_Result), 
            __intrin_bitcast<__m256d>(_SimdAbs<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(_High)), 1));

        return __intrin_bitcast<_VectorType_>(_Result);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline auto _SimdArithmetic<arch::CpuFeature::AVX512F, zmm512>::_Reduce(_VectorType_ _Vector) noexcept {
    if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>) {
        return _mm512_reduce_add_epi64(__intrin_bitcast<__m512i>(_Vector));
    }
    else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>) {
        return _mm512_reduce_add_epi32(__intrin_bitcast<__m512i>(_Vector));
    }
    else if constexpr (_Is_ps_v<_DesiredType_>) {
        return _mm512_reduce_add_ps(__intrin_bitcast<__m512>(_Vector));
    }
    else if constexpr (_Is_pd_v<_DesiredType_>) {
        return _mm512_reduce_add_pd(__intrin_bitcast<__m512d>(_Vector));
    }
    else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>) {
        return _SimdReduce<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(__intrin_bitcast<__m256i>(_Vector)) +
            _SimdReduce<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(_mm512_extractf64x4_pd(__intrin_bitcast<__m512d>(_Vector), 1));
    }
    else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>) {
        return _SimdReduce<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(__intrin_bitcast<__m256i>(_Vector)) +
            _SimdReduce<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(_mm512_extractf64x4_pd(__intrin_bitcast<__m512d>(_Vector), 1));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX512F, zmm512>::_ShiftRightVector(
    _VectorType_    _Vector,
    uint32          _ByteShift) noexcept
{
    auto _Low = __intrin_bitcast<__m256i>(_Vector);
    auto _High = _mm512_extracti64x4_epi64(__intrin_bitcast<__m512i>(_Vector), 1);

    if (_ByteShift >= 32) {
        _Low = _mm256_srli_si256(_High, _ByteShift - 32);
        _High = _mm256_setzero_si256();
    }
    else {
        const auto _LowYmmShift = _mm256_srli_si256(_Low, _ByteShift);
        const auto _Shifted = _mm256_alignr_epi8(_High, _Low, 32 - _ByteShift);

        _High = _Shifted;
        _Low = _LowYmmShift;
    }

    return __intrin_bitcast<_VectorType_>(_mm512_inserti64x4(__intrin_bitcast<__m512i>(_Low), _High, 1));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX512F, zmm512>::_ShiftLeftVector(
    _VectorType_    _Vector,
    uint32          _ByteShift) noexcept
{
    auto _Low = __intrin_bitcast<__m256i>(_Vector);
    auto _High = _mm512_extracti64x4_epi64(__intrin_bitcast<__m512i>(_Vector), 1);

    if (_ByteShift >= 32) {

    }
    else {
        const auto _LowYmmShift = _mm256_srli_si256(_Low, _ByteShift);
        const auto _Shifted = _mm256_alignr_epi8(_High, _Low, 32 - _ByteShift);

        _High = _Shifted;
        _Low = _LowYmmShift;
    }

    return __intrin_bitcast<_VectorType_>(_mm512_inserti64x4(__intrin_bitcast<__m512i>(_Low), _High, 1));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX512F, zmm512>::_ShiftRightElements(
    _VectorType_    _Vector,
    uint32          _BitShift) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        return __intrin_bitcast<_VectorType_>(_mm512_srli_epi64(__intrin_bitcast<__m512i>(_Vector), _BitShift));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        return __intrin_bitcast<_VectorType_>(_mm512_srli_epi32(__intrin_bitcast<__m512i>(_Vector), _BitShift));
    }
    else {
        const auto _Low = _SimdShiftRightElements<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(
            __intrin_bitcast<__m256i>(_Vector), _BitShift);

        const auto _High = _SimdShiftRightElements<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(
            _mm512_extracti64x4_epi64(__intrin_bitcast<__m512i>(_Vector), 1), _BitShift);

        return __intrin_bitcast<_VectorType_>(_mm512_inserti64x4(__intrin_bitcast<__m512i>(_Low), _High, 1));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX512F, zmm512>::_ShiftLeftElements(
    _VectorType_    _Vector,
    uint32          _BitShift) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        return __intrin_bitcast<_VectorType_>(_mm512_slli_epi64(__intrin_bitcast<__m512i>(_Vector), _BitShift));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        return __intrin_bitcast<_VectorType_>(_mm512_slli_epi32(__intrin_bitcast<__m512i>(_Vector), _BitShift));
    }
    else {
        const auto _Low = _SimdShiftLeftElements<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(
            __intrin_bitcast<__m256i>(_Vector), _BitShift);

        const auto _High = _SimdShiftLeftElements<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(
            _mm512_extracti64x4_epi64(__intrin_bitcast<__m512i>(_Vector), 1), _BitShift);

        return __intrin_bitcast<_VectorType_>(_mm512_inserti64x4(__intrin_bitcast<__m512i>(_Low), _High, 1));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX512F, zmm512>::_Negate(_VectorType_ _Vector) noexcept {
    if      constexpr (_Is_ps_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm512_xor_ps(_Vector, _mm512_set1_ps(-0.0f)));

    else if constexpr (_Is_pd_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm512_xor_pd(_Vector,
            __intrin_bitcast<__m512d>(_mm512_setr_epi32(0, 0x80000000, 0, 0x80000000, 0, 0x80000000,
                0, 0x80000000, 0, 0x80000000, 0, 0x80000000, 0, 0x80000000, 0, 0x80000000))));

    else
        return _Substract<_DesiredType_>(_SimdBroadcastZeros<_Generation, _RegisterPolicy, _VectorType_>(), _Vector);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX512F, zmm512>::_Add(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if constexpr (_Is_pd_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_add_pd(__intrin_bitcast<__m512d>(_Left), __intrin_bitcast<__m512d>(_Right)));
    }
    else if constexpr (_Is_ps_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_add_ps(__intrin_bitcast<__m512>(_Left), __intrin_bitcast<__m512>(_Right)));
    }
    else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_add_epi32(__intrin_bitcast<__m512i>(_Left), __intrin_bitcast<__m512i>(_Right)));
    }
    else if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_add_epi64(__intrin_bitcast<__m512i>(_Left), __intrin_bitcast<__m512i>(_Right)));
    }
    else {
        const auto _Low = _SimdAdd<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(
            __intrin_bitcast<__m256i>(_Left), __intrin_bitcast<__m256i>(_Right));

        const auto _High = _SimdAdd<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(
            _mm512_extracti64x4_epi64(__intrin_bitcast<__m512i>(_Left), 1),
            _mm512_extracti64x4_epi64(__intrin_bitcast<__m512i>(_Right), 1));

        return __intrin_bitcast<_VectorType_>(_mm512_inserti64x4(__intrin_bitcast<__m512i>(_Low), _High, 1));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX512F, zmm512>::_Substract(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if constexpr (_Is_pd_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_sub_pd(__intrin_bitcast<__m512d>(_Left), __intrin_bitcast<__m512d>(_Right)));
    }
    else if constexpr (_Is_ps_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_sub_ps(__intrin_bitcast<__m512>(_Left), __intrin_bitcast<__m512>(_Right)));
    }
    else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_sub_epi32(__intrin_bitcast<__m512i>(_Left), __intrin_bitcast<__m512i>(_Right)));
    }
    else if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_sub_epi64(__intrin_bitcast<__m512i>(_Left), __intrin_bitcast<__m512i>(_Right)));
    }
    else {
        const auto _Low = _SimdSubstract<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(
            __intrin_bitcast<__m256i>(_Left), __intrin_bitcast<__m256i>(_Right));

        const auto _High = _SimdSubstract<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(
            _mm512_extracti64x4_epi64(__intrin_bitcast<__m512i>(_Left), 1),
            _mm512_extracti64x4_epi64(__intrin_bitcast<__m512i>(_Right), 1));

        return __intrin_bitcast<_VectorType_>(_mm512_inserti64x4(__intrin_bitcast<__m512i>(_Low), _High, 1));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX512F, zmm512>::_Multiply(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    return _Left;
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX512F, zmm512>::_Divide(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    return _Left;
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX512F, zmm512>::_BitNot(_VectorType_ _Vector) noexcept {
    if      constexpr (std::is_same_v<_VectorType_, __m512d>)
        return _mm512_xor_pd(_Vector, _mm512_set1_pd(-1));

    else if constexpr (std::is_same_v<_VectorType_, __m512i>)
        return _mm512_xor_si512(_Vector, _mm512_set1_epi32(-1));

    else if constexpr (std::is_same_v<_VectorType_, __m512>)
        return _mm512_xor_ps(_Vector, _mm512_set1_ps(-1));
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX512F, zmm512>::_BitXor(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if      constexpr (std::is_same_v<_VectorType_, __m512d>)
        return _mm512_xor_pd(_Left, _Right);

    else if constexpr (std::is_same_v<_VectorType_, __m512i>)
        return _mm512_xor_si512(_Left, _Right);

    else if constexpr (std::is_same_v<_VectorType_, __m512>)
        return _mm512_xor_ps(_Left, _Right);
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX512F, zmm512>::_BitAnd(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if      constexpr (std::is_same_v<_VectorType_, __m512d>)
        return _mm512_and_pd(_Left, _Right);

    else if constexpr (std::is_same_v<_VectorType_, __m512i>)
        return _mm512_and_si512(_Left, _Right);

    else if constexpr (std::is_same_v<_VectorType_, __m512>)
        return _mm512_and_ps(_Left, _Right);
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX512F, zmm512>::_BitOr(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if      constexpr (std::is_same_v<_VectorType_, __m512d>)
        return _mm512_or_pd(_Left, _Right);

    else if constexpr (std::is_same_v<_VectorType_, __m512i>)
        return _mm512_or_si512(_Left, _Right);

    else if constexpr (std::is_same_v<_VectorType_, __m512>)
        return _mm512_or_ps(_Left, _Right);
}

template <
    typename _DesiredType_,
    typename _VectorType_,
    typename _ReduceBinaryFunction_>
simd_stl_always_inline _DesiredType_ _SimdArithmetic<arch::CpuFeature::AVX512BW, zmm512>::_HorizontalFold(
    _VectorType_            _Vector,
    _ReduceBinaryFunction_  _Reduce) noexcept
{
    if constexpr (sizeof(_DesiredType_) >= 4) {
        return _SimdHorizontalFold<arch::CpuFeature::AVX512F, _RegisterPolicy, _DesiredType_>(_Vector, _Reduce);
    }
    else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>) {
        const auto _ShuffleWords = _mm512_set_epi8(
            61, 60, 63, 62, 57, 56, 59, 58, 53, 52, 55, 54, 49, 48, 51, 50,
            45, 44, 47, 46, 41, 40, 43, 42, 37, 36, 39, 38, 33, 32, 35, 34,
            29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 18,
            13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2);

        auto _HorizontalFoldedValues = __intrin_bitcast<__m512i>(_Vector);

        const auto _Shuffled1   = _mm512_permutexvar_epi64(_mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0), _HorizontalFoldedValues);
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, _Shuffled1);

        const auto _Shuffled2   = _mm512_permutex_epi64(_HorizontalFoldedValues, 0x4E);
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, _Shuffled2);

        const auto _Shuffled3   = _mm512_permutex_epi64(_HorizontalFoldedValues, 0xB1);
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, _Shuffled3);

        const auto _Shuffled4   = __intrin_bitcast<__m512i>(_mm512_permute_ps(__intrin_bitcast<__m512>(_HorizontalFoldedValues), 0xB1));
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, _Shuffled4);

        const auto _Shuffled5   = _mm512_shuffle_epi8(_HorizontalFoldedValues, _ShuffleWords);
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, __intrin_bitcast<__m512i>(_Shuffled5));

        return _mm512_cvtsi512_si32(_HorizontalFoldedValues);
    }
    else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>) {
        const auto _ShuffleWords = _mm512_set_epi8(
            61, 60, 63, 62, 57, 56, 59, 58, 53, 52, 55, 54, 49, 48, 51, 50,
            45, 44, 47, 46, 41, 40, 43, 42, 37, 36, 39, 38, 33, 32, 35, 34,
            29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 18,
            13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2);

        const auto _ShuffleBytes = _mm512_set_epi8(
            62, 63, 60, 61, 58, 59, 56, 57, 54, 55, 52, 53, 50, 51, 48, 49,
            46, 47, 44, 45, 42, 43, 40, 41, 38, 39, 36, 37, 34, 35, 32, 33,
            30, 31, 28, 29, 26, 27, 24, 25, 22, 23, 20, 21, 18, 19, 16, 17,
            14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);

        auto _HorizontalFoldedValues = __intrin_bitcast<__m512i>(_Vector);

        const auto _Shuffled1   = _mm512_permutexvar_epi64(_mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0), _HorizontalFoldedValues);
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, _Shuffled1);

        const auto _Shuffled2   = _mm512_permutex_epi64(_HorizontalFoldedValues, 0x4E);
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, _Shuffled2);

        const auto _Shuffled3   = _mm512_permutex_epi64(_HorizontalFoldedValues, 0xB1);
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, _Shuffled3);

        const auto _Shuffled4   = __intrin_bitcast<__m512i>(_mm512_permute_ps(__intrin_bitcast<__m512>(_HorizontalFoldedValues), 0xB1));
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, _Shuffled4);

        const auto _Shuffled5   = _mm512_shuffle_epi8(_HorizontalFoldedValues, _ShuffleWords);
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, __intrin_bitcast<__m512i>(_Shuffled5));

        const auto _Shuffled6   = _mm512_shuffle_epi8(_HorizontalFoldedValues, _ShuffleBytes);
        _HorizontalFoldedValues = _Reduce(_HorizontalFoldedValues, __intrin_bitcast<__m512i>(_Shuffled6));

        return _mm512_cvtsi512_si32(_HorizontalFoldedValues);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX512BW, zmm512>::_VerticalMin(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if constexpr (_Is_epi8_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm512_min_epi8(__intrin_bitcast<__m512i>(_Left), __intrin_bitcast<__m512i>(_Right)));
    else if constexpr (_Is_epu8_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm512_min_epu8(__intrin_bitcast<__m512i>(_Left), __intrin_bitcast<__m512i>(_Right)));
    else if constexpr (_Is_epi16_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm512_min_epi16(__intrin_bitcast<__m512i>(_Left), __intrin_bitcast<__m512i>(_Right)));
    else if constexpr (_Is_epu16_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm512_min_epu16(__intrin_bitcast<__m512i>(_Left), __intrin_bitcast<__m512i>(_Right)));
    else
        return _SimdVerticalMin<arch::CpuFeature::AVX512F, zmm512, _DesiredType_>(_Left, _Right);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_ _SimdArithmetic<arch::CpuFeature::AVX512BW, zmm512>::_HorizontalMin(_VectorType_ _Vector) noexcept {
    return _HorizontalFold<_DesiredType_>(_Vector, _VerticalMinWrapper<_Generation, _RegisterPolicy, _DesiredType_>{});
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX512BW, zmm512>::_VerticalMax(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if constexpr (_Is_epi8_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm512_max_epi8(__intrin_bitcast<__m512i>(_Left), __intrin_bitcast<__m512i>(_Right)));
    else if constexpr (_Is_epu8_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm512_max_epu8(__intrin_bitcast<__m512i>(_Left), __intrin_bitcast<__m512i>(_Right)));
    else if constexpr (_Is_epi16_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm512_max_epi16(__intrin_bitcast<__m512i>(_Left), __intrin_bitcast<__m512i>(_Right)));
    else if constexpr (_Is_epu16_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm512_max_epu16(__intrin_bitcast<__m512i>(_Left), __intrin_bitcast<__m512i>(_Right)));
    else
        return _SimdVerticalMax<arch::CpuFeature::AVX512F, zmm512, _DesiredType_>(_Left, _Right);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_ _SimdArithmetic<arch::CpuFeature::AVX512BW, zmm512>::_HorizontalMax(_VectorType_ _Vector) noexcept {
    return _HorizontalFold<_DesiredType_>(_Vector, _VerticalMaxWrapper<_Generation, _RegisterPolicy, _DesiredType_>{});
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline auto _SimdArithmetic<arch::CpuFeature::AVX512BW, zmm512>::_Reduce(_VectorType_ _Vector) noexcept {
    if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>) {
        return _mm512_reduce_add_epi64(__intrin_bitcast<__m512i>(_Vector));
    }
    else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>) {
        return _mm512_reduce_add_epi32(__intrin_bitcast<__m512i>(_Vector));
    }
    else if constexpr (_Is_ps_v<_DesiredType_>) {
        return _mm512_reduce_add_ps(__intrin_bitcast<__m512>(_Vector));
    }
    else if constexpr (_Is_pd_v<_DesiredType_>) {
        return _mm512_reduce_add_pd(__intrin_bitcast<__m512d>(_Vector));
    }
    else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>) {
        return _mm512_reduce_add_epi64(_mm512_sad_epu8(__intrin_bitcast<__m512i>(_Vector), _mm512_setzero_si512()));
    }
    else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>) {
        return _SimdReduce<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(__intrin_bitcast<__m256i>(_Vector)) +
            _SimdReduce<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(_mm512_extractf64x4_pd(__intrin_bitcast<__m512d>(_Vector), 1));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX512BW, zmm512>::_Abs(_VectorType_ _Vector) noexcept {
    if constexpr (_Is_epi16_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_abs_epi16(__intrin_bitcast<__m512i>(_Vector)));
    }
    else if constexpr (_Is_epi8_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_abs_epi8(__intrin_bitcast<__m512i>(_Vector)));
    }
    else {
        return _SimdAbs<arch::CpuFeature::AVX512F, zmm512, _DesiredType_>(_Vector);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX512VLF, ymm256>::_VerticalMin(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if constexpr (_Is_epi64_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_min_epi64(__intrin_bitcast<__m256i>(_Left), __intrin_bitcast<__m256i>(_Right)));
    }
    else if constexpr (_Is_epu64_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_min_epu64(__intrin_bitcast<__m256i>(_Left), __intrin_bitcast<__m256i>(_Right)));
    }
    else {
        return _SimdVerticalMin<arch::CpuFeature::AVX2, _RegisterPolicy, _DesiredType_, _VectorType_>(_Left, _Right);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX512VLF, ymm256>::_VerticalMax(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if constexpr (_Is_epi64_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_max_epi64(__intrin_bitcast<__m256i>(_Left), __intrin_bitcast<__m256i>(_Right)));
    }
    else if constexpr (_Is_epu64_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_max_epu64(__intrin_bitcast<__m256i>(_Left), __intrin_bitcast<__m256i>(_Right)));
    }
    else {
        return _SimdVerticalMax<arch::CpuFeature::AVX2, _RegisterPolicy, _DesiredType_, _VectorType_>(_Left, _Right);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX512VLF, ymm256>::_Abs(_VectorType_ _Vector) noexcept {
    if constexpr (_Is_epi64_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_abs_epi64(__intrin_bitcast<__m256i>(_Vector)));
    else
        return _SimdAbs<arch::CpuFeature::AVX2, _RegisterPolicy, _DesiredType_>(_Vector);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX512VLF, xmm128>::_VerticalMin(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if constexpr (_Is_epi64_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_min_epi64(__intrin_bitcast<__m128i>(_Left), __intrin_bitcast<__m128i>(_Right)));
    }
    else if constexpr (_Is_epu64_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_min_epu64(__intrin_bitcast<__m128i>(_Left), __intrin_bitcast<__m128i>(_Right)));
    }
    else {
        return _SimdVerticalMin<arch::CpuFeature::SSE42, _RegisterPolicy, _DesiredType_, _VectorType_>(_Left, _Right);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX512VLF, xmm128>::_VerticalMax(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if constexpr (_Is_epi64_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_max_epi64(__intrin_bitcast<__m128i>(_Left), __intrin_bitcast<__m128i>(_Right)));
    }
    else if constexpr (_Is_epu64_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_max_epu64(__intrin_bitcast<__m128i>(_Left), __intrin_bitcast<__m128i>(_Right)));
    }
    else {
        return _SimdVerticalMax<arch::CpuFeature::SSE42, _RegisterPolicy, _DesiredType_, _VectorType_>(_Left, _Right);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX512VLF, xmm128>::_Abs(_VectorType_ _Vector) noexcept {
    if constexpr (_Is_epi64_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_abs_epi64(__intrin_bitcast<__m128i>(_Vector)));
    else
        return _SimdAbs<arch::CpuFeature::SSE42, _RegisterPolicy, _DesiredType_>(_Vector);
}


#pragma endregion

__SIMD_STL_NUMERIC_NAMESPACE_END
