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
simd_stl_always_inline auto _SimdArithmetic<arch::CpuFeature::SSE2, xmm128>::_Reduce(_VectorType_ _Vector) noexcept {
    using _ReduceType = _Reduce_type<_DesiredType_>;

    if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>) {
#if defined(simd_stl_processor_x86_32)
        return static_cast<_ReduceType>(_mm_cvtsi128_si32(_IntrinBitcast<__m128i>(_Vector)) + _SimdExtract<_Generation, _RegisterPolicy, int32>(_Vector, 2));
#else 
        return static_cast<_ReduceType>(_mm_cvtsi128_si64(_IntrinBitcast<__m128i>(_Vector)) +
            _SimdExtract<_Generation, _RegisterPolicy, int64>(_Vector, 1));
#endif // defined(simd_stl_processor_x86_32)
    }
    else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>) {
        const auto _FirstReduce = _mm_sad_epu8(_IntrinBitcast<__m128i>(_Vector), _mm_setzero_si128());
#if defined(simd_stl_processor_x86_32)
        return static_cast<_ReduceType>(_mm_cvtsi128_si32(_FirstReduce) 
            + _SimdExtract<_Generation, _RegisterPolicy, int32>(_FirstReduce, 2));
#else
        return static_cast<_ReduceType>(_mm_cvtsi128_si64(_IntrinBitcast<__m128i>(_FirstReduce))
            + _SimdExtract<_Generation, _RegisterPolicy, int64>(_FirstReduce, 1));
#endif // defined(simd_stl_processor_x86_32)
    }
    else
{
        constexpr auto _Length = sizeof(_VectorType_) / sizeof(_DesiredType_);

        _DesiredType_ _Array[_Length];
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&_Array), _IntrinBitcast<__m128i>(_Vector));

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
    return _IntrinBitcast<_VectorType_>(_mm_srli_si128(_IntrinBitcast<__m128i>(_Vector), _ByteShift));
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::SSE2, xmm128>::_ShiftLeftVector(
    _VectorType_    _Vector,
    uint32          _ByteShift) noexcept
{
    return _IntrinBitcast<_VectorType_>(_mm_slli_si128(_IntrinBitcast<__m128i>(_Vector), _ByteShift));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::SSE2, xmm128>::_ShiftRightElements(
    _VectorType_    _Vector,
    uint32          _BitShift) noexcept
{
    if      constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm_srli_epi64(_IntrinBitcast<__m128i>(_Vector), _BitShift));

    else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm_srli_epi32(_IntrinBitcast<__m128i>(_Vector), _BitShift));

    else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm_srli_epi16(_IntrinBitcast<__m128i>(_Vector), _BitShift));

    else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>) {
        const auto _EvenVector = _mm_sra_epi16(_mm_slli_epi16(_IntrinBitcast<__m128i>(_Vector), 8), _mm_cvtsi32_si128(_BitShift + 8));
        const auto _OddVector = _mm_sra_epi16(_IntrinBitcast<__m128i>(_Vector), _mm_cvtsi32_si128(_BitShift));

        const auto _Mask = _mm_set1_epi32(0x00FF00FF);
        return _IntrinBitcast<_VectorType_>(_mm_or_si128(_mm_and_si128(_Mask, _EvenVector), _mm_andnot_si128(_Mask, _OddVector)));
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
        return _IntrinBitcast<_VectorType_>(_mm_slli_epi64(_IntrinBitcast<__m128i>(_Vector), _BitShift));

    else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm_slli_epi32(_IntrinBitcast<__m128i>(_Vector), _BitShift));

    else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm_slli_epi16(_IntrinBitcast<__m128i>(_Vector), _BitShift));

    else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>) {
        const auto _AndMask = _mm_and_si128(_IntrinBitcast<__m128i>(_Vector),
            _mm_set1_epi8(static_cast<int8>(0xFFu >> _BitShift)));

        return _IntrinBitcast<_VectorType_>(_mm_sll_epi16(_AndMask, _mm_cvtsi32_si128(_BitShift)));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::SSE2, xmm128>::_Negate(_VectorType_ _Vector) noexcept {
    if      constexpr (_Is_ps_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm_xor_ps(_Vector, _mm_set1_ps(-0.0f)));

    else if constexpr (_Is_pd_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm_xor_pd(_Vector,
            _IntrinBitcast<__m128d>(_mm_setr_epi32(0, 0x80000000, 0, 0x80000000))));

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
        return _IntrinBitcast<_VectorType_>(_mm_add_epi64(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

    else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm_add_epi32(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

    else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm_add_epi16(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

    else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm_add_epi8(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

    else if constexpr (_Is_ps_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm_add_ps(_IntrinBitcast<__m128>(_Left), _IntrinBitcast<__m128>(_Right)));

    else if constexpr (_Is_pd_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm_add_pd(_IntrinBitcast<__m128d>(_Left), _IntrinBitcast<__m128d>(_Right)));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::SSE2, xmm128>::_Substract(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm_sub_epi64(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

    else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm_sub_epi32(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

    else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm_sub_epi16(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

    else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm_sub_epi8(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

    else if constexpr (_Is_ps_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm_sub_ps(_IntrinBitcast<__m128>(_Left), _IntrinBitcast<__m128>(_Right)));

    else if constexpr (_Is_pd_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm_sub_pd(_IntrinBitcast<__m128d>(_Left), _IntrinBitcast<__m128d>(_Right)));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::SSE2, xmm128>::_Multiply(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if constexpr (_Is_epi32_v<_DesiredType_>) {
        const auto _ShuffledLeft = _mm_shuffle_epi32(_IntrinBitcast<__m128i>(_Left), 0xF5);
        const auto _ShuffledRight = _mm_shuffle_epi32(_IntrinBitcast<__m128i>(_Right), 0xF5);

        const auto _ProductEvenIndices = _mm_mul_epu32(_Left, _Right);
        const auto _ProductOddIndices = _mm_mul_epu32(_ShuffledLeft, _ShuffledRight);

        const auto _ProductLowPair = _mm_unpacklo_epi32(_ProductEvenIndices, _ProductOddIndices);
        const auto _ProductHighPair = _mm_unpackhi_epi32(_ProductEvenIndices, _ProductOddIndices);

        return _IntrinBitcast<_VectorType_>(_mm_unpacklo_epi64(_ProductLowPair, _ProductHighPair));
    }

    else if constexpr (_Is_epu32_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm_mul_epu32(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

    else if constexpr (_Is_ps_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm_mul_ps(_IntrinBitcast<__m128>(_Left), _IntrinBitcast<__m128>(_Right)));

    else if constexpr (_Is_pd_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm_mul_pd(_IntrinBitcast<__m128d>(_Left), _IntrinBitcast<__m128d>(_Right)));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::SSE2, xmm128>::_Divide(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if constexpr (_Is_pd_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm_div_pd(_IntrinBitcast<__m128d>(_Left), _IntrinBitcast<__m128d>(_Right)));

    else if constexpr (_Is_ps_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm_div_ps(_IntrinBitcast<__m128>(_Left), _IntrinBitcast<__m128>(_Right)));
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
    typename _VectorType_>
simd_stl_always_inline auto _SimdArithmetic<arch::CpuFeature::SSSE3, xmm128>::_Reduce(_VectorType_ _Vector) noexcept {
    using _ReduceType = _Reduce_type<_DesiredType_>;

    if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>) {
        const auto _Zeros = _mm_setzero_si128();

        const auto _Reduce4 = _mm_hadd_epi32(_IntrinBitcast<__m128i>(_Vector), _Zeros); // (0+1),(2+3),0,0
        const auto _Reduce5 = _mm_hadd_epi32(_Reduce4, _Zeros);                         // (0+...+3),0,0,0

        return static_cast<_ReduceType>(_mm_cvtsi128_si32(_Reduce5));
    }
    else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>) {
        const auto _Zeros = _mm_setzero_si128();

        const auto _Reduce2 = _mm_hadd_epi16(_IntrinBitcast<__m128i>(_Vector), _Zeros);
        const auto _Reduce3 = _mm_unpacklo_epi16(_Reduce2, _Zeros);

        const auto _Reduce4 = _mm_hadd_epi32(_Reduce3, _Zeros); // (0+1),(2+3),0,0
        const auto _Reduce5 = _mm_hadd_epi32(_Reduce4, _Zeros); // (0+...+3),0,0,0

        return static_cast<_ReduceType>(_mm_cvtsi128_si32(_Reduce5));
    }
    else 
    {
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
        return _IntrinBitcast<_VectorType_>(_mm_mul_epi32(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

    else if constexpr (_Is_epu32_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm_mul_epu32(_IntrinBitcast<__m128i>(_Left), _IntrinBitcast<__m128i>(_Right)));

    else if constexpr (_Is_ps_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm_mul_ps(_IntrinBitcast<__m128>(_Left), _IntrinBitcast<__m128>(_Right)));

    else if constexpr (_Is_pd_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm_mul_pd(_IntrinBitcast<__m128d>(_Left), _IntrinBitcast<__m128d>(_Right)));
}

#pragma endregion 

#pragma region Avx Simd arithmetic

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline auto _SimdArithmetic<arch::CpuFeature::AVX2, ymm256>::_Reduce(_VectorType_ _Vector) noexcept {
    using _ReduceType = _Reduce_type<_DesiredType_>;

    if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>) {
        const auto _Low64   = _IntrinBitcast<__m128i>(_Vector);
        const auto _High64  = _mm256_extracti128_si256(_IntrinBitcast<__m256i>(_Vector), 1);

        const auto _Reduce  = _mm_add_epi64(_Low64, _High64);
        return _SimdReduce<arch::CpuFeature::SSSE3, xmm128, _DesiredType_>(_Reduce);
    }
    if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>) {
        const auto _Zeros = _mm256_setzero_si256();

        const auto _Reduce4 = _mm256_hadd_epi32(_IntrinBitcast<__m256i>(_Vector), _Zeros); // (0+1),(2+3),0,0 per lane
        const auto _Reduce5 = _mm256_permute4x64_epi64(_Reduce4, 0xD8); // low lane  (0+1),(2+3),(4+5),(6+7)

        const auto _Reduce6 = _mm256_hadd_epi32(_Reduce5, _Zeros); // (0+...+3),(4+...+7),0,0
        const auto _Reduce7 = _mm256_hadd_epi32(_Reduce6, _Zeros); // (0+...+7),0,0,0

        return static_cast<_ReduceType>(_mm_cvtsi128_si32(_IntrinBitcast<__m128i>(_Reduce7)));
    }
    else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>) {
        const auto _Zeros = _mm256_setzero_si256();

        const auto _Reduce2 = _mm256_hadd_epi16(_IntrinBitcast<__m256i>(_Vector), _Zeros);
        const auto _Reduce3 = _mm256_unpacklo_epi16(_Reduce2, _Zeros);

        const auto _Reduce4 = _mm256_hadd_epi32(_Reduce3, _Zeros); // (0+1),(2+3),0,0 per lane
        const auto _Reduce5 = _mm256_permute4x64_epi64(_Reduce4, 0xD8); // low lane  (0+1),(2+3),(4+5),(6+7)

        const auto _Reduce6 = _mm256_hadd_epi32(_Reduce5, _Zeros); // (0+...+3),(4+...+7),0,0
        const auto _Reduce7 = _mm256_hadd_epi32(_Reduce6, _Zeros); // (0+...+7),0,0,0

        return static_cast<_ReduceType>(_mm_cvtsi128_si32(_IntrinBitcast<__m128i>(_Reduce7)));
    }
    else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>) {
        const auto _Reduce1 = _mm256_sad_epu8(_IntrinBitcast<__m256i>(_Vector), _mm256_setzero_si256());

        const auto _Low64 = _mm256_castsi256_si128(_Reduce1);
        const auto _High64 = _mm256_extracti128_si256(_Reduce1, 1);

        const auto _Reduce8 = _mm_add_epi64(_Low64, _High64);
        return _SimdReduce<arch::CpuFeature::SSSE3, xmm128, int64>(_Reduce8);
    }
    else {
        constexpr auto _Length = sizeof(_VectorType_) / sizeof(_DesiredType_);

        _DesiredType_ _Array[_Length];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&_Array), _IntrinBitcast<__m256i>(_Vector));

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
    return _IntrinBitcast<_VectorType_>(_mm256_srli_si256(_IntrinBitcast<__m256i>(_Vector), _ByteShift));
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX2, ymm256>::_ShiftLeftVector(
    _VectorType_    _Vector,
    uint32          _ByteShift) noexcept
{
    return _IntrinBitcast<_VectorType_>(_mm256_slli_si256(_IntrinBitcast<__m256i>(_Vector), _ByteShift));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX2, ymm256>::_ShiftRightElements(
    _VectorType_    _Vector,
    uint32          _BitShift) noexcept
{
    if      constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm256_srli_epi64(_IntrinBitcast<__m256i>(_Vector), _BitShift));

    else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm256_srli_epi32(_IntrinBitcast<__m256i>(_Vector), _BitShift));

    else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm256_srli_epi16(_IntrinBitcast<__m256i>(_Vector), _BitShift));

    else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>) {
        const auto _EvenVector = _mm256_sra_epi16(_mm256_slli_epi16(
            _IntrinBitcast<__m256i>(_Vector), 8), _mm_cvtsi32_si128(_BitShift + 8));

        const auto _OddVector = _mm256_sra_epi16(_IntrinBitcast<__m256i>(_Vector), _mm_cvtsi32_si128(_BitShift));

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
        return _IntrinBitcast<_VectorType_>(_mm256_slli_epi64(_IntrinBitcast<__m256i>(_Vector), _BitShift));

    else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm256_slli_epi32(_IntrinBitcast<__m256i>(_Vector), _BitShift));

    else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm256_slli_epi16(_IntrinBitcast<__m256i>(_Vector), _BitShift));

    else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>) {
        const auto _AndMask = _mm256_and_si256(_IntrinBitcast<__m256i>(_Vector),
            _mm256_set1_epi8(static_cast<int8>(0xFFu >> _BitShift)));

        return _IntrinBitcast<_VectorType_>(_mm256_sll_epi16(_AndMask, _mm_cvtsi32_si128(_BitShift)));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX2, ymm256>::_Negate(_VectorType_ _Vector) noexcept {
    if      constexpr (_Is_ps_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm256_xor_ps(_IntrinBitcast<__m256>(_Vector), _mm256_set1_ps(-0.0f)));

    else if constexpr (_Is_pd_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm256_xor_pd(_IntrinBitcast<__m256d>(_Vector),
            _IntrinBitcast<__m256d>(_mm256_setr_epi32(0, 0x80000000, 0, 0x80000000, 0, 0x80000000, 0, 0x80000000))));

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
        return _IntrinBitcast<_VectorType_>(_mm256_add_epi64(_IntrinBitcast<__m256i>(_Left), _IntrinBitcast<__m256i>(_Right)));

    else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm256_add_epi32(_IntrinBitcast<__m256i>(_Left), _IntrinBitcast<__m256i>(_Right)));

    else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm256_add_epi16(_IntrinBitcast<__m256i>(_Left), _IntrinBitcast<__m256i>(_Right)));

    else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm256_add_epi8(_IntrinBitcast<__m256i>(_Left), _IntrinBitcast<__m256i>(_Right)));

    else if constexpr (_Is_ps_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm256_add_ps(_IntrinBitcast<__m256>(_Left), _IntrinBitcast<__m256>(_Right)));

    else if constexpr (_Is_pd_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm256_add_pd(_IntrinBitcast<__m256d>(_Left), _IntrinBitcast<__m256d>(_Right)));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX2, ymm256>::_Substract(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm256_sub_epi64(_IntrinBitcast<__m256i>(_Left), _IntrinBitcast<__m256i>(_Right)));

    else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm256_sub_epi32(_IntrinBitcast<__m256i>(_Left), _IntrinBitcast<__m256i>(_Right)));

    else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm256_sub_epi16(_IntrinBitcast<__m256i>(_Left), _IntrinBitcast<__m256i>(_Right)));

    else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm256_sub_epi8(_IntrinBitcast<__m256i>(_Left), _IntrinBitcast<__m256i>(_Right)));

    else if constexpr (_Is_ps_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm256_sub_ps(_IntrinBitcast<__m256>(_Left), _IntrinBitcast<__m256>(_Right)));

    else if constexpr (_Is_pd_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm256_sub_pd(_IntrinBitcast<__m256d>(_Left), _IntrinBitcast<__m256d>(_Right)));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX2, ymm256>::_Multiply(
    _VectorType_ _Left,
    _VectorType_ _Right) noexcept
{
    if constexpr (_Is_epi32_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm256_mul_epi32(_IntrinBitcast<__m256i>(_Left), _IntrinBitcast<__m256i>(_Right)));

    else if constexpr (_Is_epu32_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm256_mul_epu32(_IntrinBitcast<__m256i>(_Left), _IntrinBitcast<__m256i>(_Right)));

    else if constexpr (_Is_ps_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm256_mul_ps(_IntrinBitcast<__m256>(_Left), _IntrinBitcast<__m256>(_Right)));

    else if constexpr (_Is_pd_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm256_mul_pd(_IntrinBitcast<__m256d>(_Left), _IntrinBitcast<__m256d>(_Right)));

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
        return _IntrinBitcast<_VectorType_>(_mm256_div_pd(_IntrinBitcast<__m256d>(_Left), _IntrinBitcast<__m256d>(_Right)));

    else if constexpr (_Is_ps_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm256_div_ps(_IntrinBitcast<__m256>(_Left), _IntrinBitcast<__m256>(_Right)));
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
simd_stl_always_inline auto _SimdArithmetic<arch::CpuFeature::AVX512F, zmm512>::_Reduce(_VectorType_ _Vector) noexcept {
    if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>) {
        return _mm512_reduce_add_epi64(_IntrinBitcast<__m512i>(_Vector));
    }
    else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>) {
        return _mm512_reduce_add_epi32(_IntrinBitcast<__m512i>(_Vector));
    }
    else if constexpr (_Is_ps_v<_DesiredType_>) {
        return _mm512_reduce_add_ps(_IntrinBitcast<__m512>(_Vector));
    }
    else if constexpr (_Is_pd_v<_DesiredType_>) {
        return _mm512_reduce_add_pd(_IntrinBitcast<__m512d>(_Vector));
    }
    else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>) {
        return _SimdReduce<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(_IntrinBitcast<__m256i>(_Vector)) +
            _SimdReduce<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(_mm512_extractf64x4_pd(_IntrinBitcast<__m512d>(_Vector), 1));
    }
    else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>) {
        return _SimdReduce<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(_IntrinBitcast<__m256i>(_Vector)) +
            _SimdReduce<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(_mm512_extractf64x4_pd(_IntrinBitcast<__m512d>(_Vector), 1));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX512F, zmm512>::_ShiftRightVector(
    _VectorType_    _Vector,
    uint32          _ByteShift) noexcept
{
    auto _Low = _IntrinBitcast<__m256i>(_Vector);
    auto _High = _mm512_extracti64x4_epi64(_IntrinBitcast<__m512i>(_Vector), 1);

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

    return _IntrinBitcast<_VectorType_>(_mm512_inserti64x4(_IntrinBitcast<__m512i>(_Low), _High, 1));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX512F, zmm512>::_ShiftLeftVector(
    _VectorType_    _Vector,
    uint32          _ByteShift) noexcept
{
    auto _Low = _IntrinBitcast<__m256i>(_Vector);
    auto _High = _mm512_extracti64x4_epi64(_IntrinBitcast<__m512i>(_Vector), 1);

    if (_ByteShift >= 32) {

    }
    else {
        const auto _LowYmmShift = _mm256_srli_si256(_Low, _ByteShift);
        const auto _Shifted = _mm256_alignr_epi8(_High, _Low, 32 - _ByteShift);

        _High = _Shifted;
        _Low = _LowYmmShift;
    }

    return _IntrinBitcast<_VectorType_>(_mm512_inserti64x4(_IntrinBitcast<__m512i>(_Low), _High, 1));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX512F, zmm512>::_ShiftRightElements(
    _VectorType_    _Vector,
    uint32          _BitShift) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        return _IntrinBitcast<_VectorType_>(_mm512_srli_epi64(_IntrinBitcast<__m512i>(_Vector), _BitShift));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        return _IntrinBitcast<_VectorType_>(_mm512_srli_epi32(_IntrinBitcast<__m512i>(_Vector), _BitShift));
    }
    else {
        const auto _Low = _SimdShiftRightElements<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(
            _IntrinBitcast<__m256i>(_Vector), _BitShift);

        const auto _High = _SimdShiftRightElements<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(
            _mm512_extracti64x4_epi64(_IntrinBitcast<__m512i>(_Vector), 1), _BitShift);

        return _IntrinBitcast<_VectorType_>(_mm512_inserti64x4(_IntrinBitcast<__m512i>(_Low), _High, 1));
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
        return _IntrinBitcast<_VectorType_>(_mm512_slli_epi64(_IntrinBitcast<__m512i>(_Vector), _BitShift));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        return _IntrinBitcast<_VectorType_>(_mm512_slli_epi32(_IntrinBitcast<__m512i>(_Vector), _BitShift));
    }
    else {
        const auto _Low = _SimdShiftLeftElements<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(
            _IntrinBitcast<__m256i>(_Vector), _BitShift);

        const auto _High = _SimdShiftLeftElements<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(
            _mm512_extracti64x4_epi64(_IntrinBitcast<__m512i>(_Vector), 1), _BitShift);

        return _IntrinBitcast<_VectorType_>(_mm512_inserti64x4(_IntrinBitcast<__m512i>(_Low), _High, 1));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ _SimdArithmetic<arch::CpuFeature::AVX512F, zmm512>::_Negate(_VectorType_ _Vector) noexcept {
    if      constexpr (_Is_ps_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm512_xor_ps(_Vector, _mm512_set1_ps(-0.0f)));

    else if constexpr (_Is_pd_v<_DesiredType_>)
        return _IntrinBitcast<_VectorType_>(_mm512_xor_pd(_Vector,
            _IntrinBitcast<__m512d>(_mm512_setr_epi32(0, 0x80000000, 0, 0x80000000, 0, 0x80000000,
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
        return _IntrinBitcast<_VectorType_>(_mm512_add_pd(_IntrinBitcast<__m512d>(_Left), _IntrinBitcast<__m512d>(_Right)));
    }
    else if constexpr (_Is_ps_v<_DesiredType_>) {
        return _IntrinBitcast<_VectorType_>(_mm512_add_ps(_IntrinBitcast<__m512>(_Left), _IntrinBitcast<__m512>(_Right)));
    }
    else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>) {
        return _IntrinBitcast<_VectorType_>(_mm512_add_epi32(_IntrinBitcast<__m512i>(_Left), _IntrinBitcast<__m512i>(_Right)));
    }
    else if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>) {
        return _IntrinBitcast<_VectorType_>(_mm512_add_epi64(_IntrinBitcast<__m512i>(_Left), _IntrinBitcast<__m512i>(_Right)));
    }
    else {
        const auto _Low = _SimdAdd<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(
            _IntrinBitcast<__m256i>(_Left), _IntrinBitcast<__m256i>(_Right));

        const auto _High = _SimdAdd<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(
            _mm512_extracti64x4_epi64(_IntrinBitcast<__m512i>(_Left), 1),
            _mm512_extracti64x4_epi64(_IntrinBitcast<__m512i>(_Right), 1));

        return _IntrinBitcast<_VectorType_>(_mm512_inserti64x4(_IntrinBitcast<__m512i>(_Low), _High, 1));
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
        return _IntrinBitcast<_VectorType_>(_mm512_sub_pd(_IntrinBitcast<__m512d>(_Left), _IntrinBitcast<__m512d>(_Right)));
    }
    else if constexpr (_Is_ps_v<_DesiredType_>) {
        return _IntrinBitcast<_VectorType_>(_mm512_sub_ps(_IntrinBitcast<__m512>(_Left), _IntrinBitcast<__m512>(_Right)));
    }
    else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>) {
        return _IntrinBitcast<_VectorType_>(_mm512_sub_epi32(_IntrinBitcast<__m512i>(_Left), _IntrinBitcast<__m512i>(_Right)));
    }
    else if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>) {
        return _IntrinBitcast<_VectorType_>(_mm512_sub_epi64(_IntrinBitcast<__m512i>(_Left), _IntrinBitcast<__m512i>(_Right)));
    }
    else {
        const auto _Low = _SimdSubstract<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(
            _IntrinBitcast<__m256i>(_Left), _IntrinBitcast<__m256i>(_Right));

        const auto _High = _SimdSubstract<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(
            _mm512_extracti64x4_epi64(_IntrinBitcast<__m512i>(_Left), 1),
            _mm512_extracti64x4_epi64(_IntrinBitcast<__m512i>(_Right), 1));

        return _IntrinBitcast<_VectorType_>(_mm512_inserti64x4(_IntrinBitcast<__m512i>(_Low), _High, 1));
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
    typename _VectorType_>
simd_stl_always_inline auto _SimdArithmetic<arch::CpuFeature::AVX512BW, zmm512>::_Reduce(_VectorType_ _Vector) noexcept {
    if constexpr (_Is_epi64_v<_DesiredType_> || _Is_epu64_v<_DesiredType_>) {
        return _mm512_reduce_add_epi64(_IntrinBitcast<__m512i>(_Vector));
    }
    else if constexpr (_Is_epi32_v<_DesiredType_> || _Is_epu32_v<_DesiredType_>) {
        return _mm512_reduce_add_epi32(_IntrinBitcast<__m512i>(_Vector));
    }
    else if constexpr (_Is_ps_v<_DesiredType_>) {
        return _mm512_reduce_add_ps(_IntrinBitcast<__m512>(_Vector));
    }
    else if constexpr (_Is_pd_v<_DesiredType_>) {
        return _mm512_reduce_add_pd(_IntrinBitcast<__m512d>(_Vector));
    }
    else if constexpr (_Is_epi8_v<_DesiredType_> || _Is_epu8_v<_DesiredType_>) {
        return _mm512_reduce_add_epi64(_mm512_sad_epu8(_IntrinBitcast<__m512i>(_Vector), _mm512_setzero_si512()));
    }
    else if constexpr (_Is_epi16_v<_DesiredType_> || _Is_epu16_v<_DesiredType_>) {
        return _SimdReduce<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(_IntrinBitcast<__m256i>(_Vector)) +
            _SimdReduce<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(_mm512_extractf64x4_pd(_IntrinBitcast<__m512d>(_Vector), 1));
    }
}

#pragma endregion

__SIMD_STL_NUMERIC_NAMESPACE_END
