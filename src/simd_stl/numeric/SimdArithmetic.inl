#pragma once 


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

#pragma region Sse2-Sse4.2 Simd arithmetic

template <typename _Type_>
struct __reduce_type_helper {
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
    
    static_assert(!std::is_same_v<type, void>, "Unsupported type size for __reduce_type_helper");
};

template <typename _Type_>
using __reduce_type = typename __reduce_type_helper<_Type_>::type;

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::SSE2, xmm128>::__vertical_min(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (__is_epi16_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_min_epi16(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else if constexpr (__is_epu8_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_min_epu8(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else if constexpr (__is_ps_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_min_ps(
            __intrin_bitcast<__m128>(__left), __intrin_bitcast<__m128>(__right)));
    }
    else if constexpr (__is_pd_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_min_pd(
            __intrin_bitcast<__m128d>(__left), __intrin_bitcast<__m128d>(__right)));
    }
    else {
        const auto __mask = __simd_compare<__generation, __register_policy, _DesiredType_, __simd_comparison::less>(__left, __right);
        return __simd_blend<__generation, __register_policy, _DesiredType_>(__left, __right, __mask);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_,
    typename _ReduceBinaryFunction_>
simd_stl_always_inline _DesiredType_ __simd_arithmetic<arch::CpuFeature::SSE2, xmm128>::__horizontal_fold(
    _VectorType_            __vector,
    _ReduceBinaryFunction_  __reduce) noexcept
{
    if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_> || __is_pd_v<_DesiredType_>) {
        auto __horizontal_folded_values = __intrin_bitcast<__m128d>(__vector);

        const auto __shuffled       = _mm_shuffle_pd(__horizontal_folded_values, __horizontal_folded_values, 1);
        __horizontal_folded_values  = __reduce(__shuffled, __horizontal_folded_values);

        if constexpr (__is_pd_v<_DesiredType_>)
            return _mm_cvtsd_f64(__horizontal_folded_values);
        else
            return _mm_cvtsi128_si64(__intrin_bitcast<__m128i>(__horizontal_folded_values));
    }
    else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_> || __is_ps_v<_DesiredType_>) {
        auto __horizontal_folded_values = __intrin_bitcast<__m128i>(__vector);

        const auto __shuffled1      = _mm_shuffle_epi32(__horizontal_folded_values, 0x4E);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffled1);

        const auto __shuffled2      = _mm_shuffle_epi32(__horizontal_folded_values, 0xB1);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffled2);

        if constexpr (__is_ps_v<_DesiredType_>)
            return _mm_cvtss_f32(__intrin_bitcast<__m128>(__horizontal_folded_values));
        else
            return _mm_cvtsi128_si32(__horizontal_folded_values);
    }
    else if constexpr (__is_epi16_v<_DesiredType_> || __is_epu16_v<_DesiredType_>) {
        auto __horizontal_folded_values = __intrin_bitcast<__m128i>(__vector);
        
        __horizontal_folded_values = __reduce(__horizontal_folded_values, _mm_srli_si128(__horizontal_folded_values, 8));
        __horizontal_folded_values = __reduce(__horizontal_folded_values, _mm_srli_si128(__horizontal_folded_values, 4));

        __horizontal_folded_values = __reduce(__horizontal_folded_values, _mm_srli_si128(__horizontal_folded_values, 2));

        return _mm_cvtsi128_si32(__intrin_bitcast<__m128i>(__horizontal_folded_values));
    } 
    else if constexpr (__is_epi8_v<_DesiredType_> || __is_epu8_v<_DesiredType_>) {
        auto __horizontal_folded_values = __intrin_bitcast<__m128i>(__vector);

        __horizontal_folded_values = __reduce(__horizontal_folded_values, _mm_srli_si128(__horizontal_folded_values, 8));
        __horizontal_folded_values = __reduce(__horizontal_folded_values, _mm_srli_si128(__horizontal_folded_values, 4));

        __horizontal_folded_values = __reduce(__horizontal_folded_values, _mm_srli_si128(__horizontal_folded_values, 2));
        __horizontal_folded_values = __reduce(__horizontal_folded_values, _mm_srli_si128(__horizontal_folded_values, 1));

        return _mm_cvtsi128_si32(__intrin_bitcast<__m128i>(__horizontal_folded_values));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_ __simd_arithmetic<arch::CpuFeature::SSE2, xmm128>::__horizontal_min(_VectorType_ __vector) noexcept {
    return __horizontal_fold<_DesiredType_>(__vector, __vertical_min_wrapper<__generation, __register_policy, _DesiredType_>{});
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::SSE2, xmm128>::__vertical_max(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (__is_epi16_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_max_epi16(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else if constexpr (__is_epu8_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_max_epu8(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else if constexpr (__is_ps_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_max_ps(
            __intrin_bitcast<__m128>(__left), __intrin_bitcast<__m128>(__right)));
    }
    else if constexpr (__is_pd_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_max_pd(
            __intrin_bitcast<__m128d>(__left), __intrin_bitcast<__m128d>(__right)));
    }
    else {
        const auto __mask = __simd_compare<__generation, __register_policy, _DesiredType_, __simd_comparison::greater>(__left, __right);
        return __simd_blend<__generation, __register_policy, _DesiredType_>(__left, __right, __mask);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_ __simd_arithmetic<arch::CpuFeature::SSE2, xmm128>::__horizontal_max(_VectorType_ __vector) noexcept {
    return __horizontal_fold<_DesiredType_>(__vector, __vertical_max_wrapper<__generation, __register_policy, _DesiredType_>{});
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::SSE2, xmm128>::__abs(_VectorType_ __vector) noexcept {
    if constexpr (std::is_unsigned_v<_DesiredType_>) {
        return __vector;
    }
    else if constexpr (__is_epi64_v<_DesiredType_>) {
        const auto __high_sign  = _mm_srai_epi32(__intrin_bitcast<__m128i>(__vector), 31);
        const auto __sign       = _mm_shuffle_epi32(__high_sign, 0xF5);

        const auto __invert     = _mm_xor_si128(__intrin_bitcast<__m128i>(__vector), __sign);
        return __intrin_bitcast<_VectorType_>(_mm_sub_epi64(__invert, __sign));
    }
    else if constexpr (__is_epi32_v<_DesiredType_>) {
        const auto __sign   = _mm_srai_epi32(__intrin_bitcast<__m128i>(__vector), 31);
        const auto __invert = _mm_xor_si128(__intrin_bitcast<__m128i>(__vector), __sign);

        return __intrin_bitcast<_VectorType_>(_mm_sub_epi32(__invert, __sign));
    }
    else if constexpr (__is_epi16_v<_DesiredType_>) {
        const auto __negate = _mm_sub_epi16(_mm_setzero_si128(), __intrin_bitcast<__m128i>(__vector));
        return _mm_max_epi16(__intrin_bitcast<__m128i>(__vector), __negate);
    }
    else if constexpr (__is_epi8_v<_DesiredType_>) {
        const auto __negate = _mm_sub_epi8(_mm_setzero_si128(), __intrin_bitcast<__m128i>(__vector));
        return __intrin_bitcast<_VectorType_>(_mm_min_epu8(__intrin_bitcast<__m128i>(__vector), __negate));
    }
    else if constexpr (__is_ps_v<_DesiredType_>) {
        const auto __mask = _mm_set1_epi32(0x7FFFFFFF);
        return __intrin_bitcast<_VectorType_>(_mm_and_ps(__intrin_bitcast<__m128>(__vector), __intrin_bitcast<__m128>(__mask)));
    }
    else if constexpr (__is_pd_v<_DesiredType_>) {
        const auto __mask = _mm_set_epi32(0xFFFFFFFFu, 0x7FFFFFFFu, 0xFFFFFFFFu, 0x7FFFFFFFu);
        return __intrin_bitcast<_VectorType_>(_mm_and_pd(__intrin_bitcast<__m128d>(__vector), __intrin_bitcast<__m128d>(__mask)));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline auto __simd_arithmetic<arch::CpuFeature::SSE2, xmm128>::__reduce(_VectorType_ __vector) noexcept {
    using _ReduceType = __reduce_type<_DesiredType_>;

    if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>) {
#if defined(simd_stl_processor_x86_32)
        return static_cast<_ReduceType>(_mm_cvtsi128_si32(_IntrinBitcast<__m128i>(__vector)) + 
            __simd_extract<__generation, __register_policy, int32>(__vector, 2));
#else 
        return static_cast<_ReduceType>(_mm_cvtsi128_si64(__intrin_bitcast<__m128i>(__vector)) +
            __simd_extract<__generation, __register_policy, int64>(__vector, 1));
#endif // defined(simd_stl_processor_x86_32)
    }
    else if constexpr (__is_epi8_v<_DesiredType_> || __is_epu8_v<_DesiredType_>) {
        const auto __first_reduce = _mm_sad_epu8(__intrin_bitcast<__m128i>(__vector), _mm_setzero_si128());
#if defined(simd_stl_processor_x86_32)
        return static_cast<_ReduceType>(_mm_cvtsi128_si32(__first_reduce)
            + __simd_extract<__generation, __register_policy, int32>(__first_reduce, 2));
#else
        return static_cast<_ReduceType>(_mm_cvtsi128_si64(__intrin_bitcast<__m128i>(__first_reduce))
            + __simd_extract<__generation, __register_policy, int64>(__first_reduce, 1));
#endif // defined(simd_stl_processor_x86_32)
    }
    else {
        constexpr auto __length = sizeof(_VectorType_) / sizeof(_DesiredType_);

        _DesiredType_ __array[__length];
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&__array), __intrin_bitcast<__m128i>(__vector));

        auto __sum = _ReduceType(0);

        for (auto __index = 0; __index < __length; ++__index)
            __sum += __array[__index];

        return __sum;
    }
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::SSE2, xmm128>::__shift_right_vector(
    _VectorType_    __vector,
    uint32          __byte_shift) noexcept
{
    return __intrin_bitcast<_VectorType_>(_mm_srli_si128(__intrin_bitcast<__m128i>(__vector), __byte_shift));
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::SSE2, xmm128>::__shift_left_vector(
    _VectorType_    __vector,
    uint32          __byte_shift) noexcept
{
    return __intrin_bitcast<_VectorType_>(_mm_slli_si128(__intrin_bitcast<__m128i>(__vector), __byte_shift));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::SSE2, xmm128>::__shift_right_elements(
    _VectorType_    __vector,
    uint32          __bit_shift) noexcept
{
    if      constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_srli_epi64(__intrin_bitcast<__m128i>(__vector), __bit_shift));

    else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_srli_epi32(__intrin_bitcast<__m128i>(__vector), __bit_shift));

    else if constexpr (__is_epi16_v<_DesiredType_> || __is_epu16_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_srli_epi16(__intrin_bitcast<__m128i>(__vector), __bit_shift));

    else if constexpr (__is_epi8_v<_DesiredType_> || __is_epu8_v<_DesiredType_>) {
        const auto __even_vector    = _mm_sra_epi16(_mm_slli_epi16(
            __intrin_bitcast<__m128i>(__vector), 8), _mm_cvtsi32_si128(__bit_shift + 8));
        const auto __odd_vector     = _mm_sra_epi16(
            __intrin_bitcast<__m128i>(__vector), _mm_cvtsi32_si128(__bit_shift));

        const auto __mask = _mm_set1_epi32(0x00FF00FF);
        return __intrin_bitcast<_VectorType_>(_mm_or_si128(
            _mm_and_si128(__mask, __even_vector), _mm_andnot_si128(__mask, __odd_vector)));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::SSE2, xmm128>::__shift_left_elements(
    _VectorType_    __vector,
    uint32          __bit_shift) noexcept
{
    if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_slli_epi64(__intrin_bitcast<__m128i>(__vector), __bit_shift));

    else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_slli_epi32(__intrin_bitcast<__m128i>(__vector), __bit_shift));

    else if constexpr (__is_epi16_v<_DesiredType_> || __is_epu16_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_slli_epi16(__intrin_bitcast<__m128i>(__vector), __bit_shift));

    else if constexpr (__is_epi8_v<_DesiredType_> || __is_epu8_v<_DesiredType_>) {
        const auto __and_mask = _mm_and_si128(__intrin_bitcast<__m128i>(__vector),
            _mm_set1_epi8(static_cast<int8>(0xFFu >> __bit_shift)));

        return __intrin_bitcast<_VectorType_>(_mm_sll_epi16(__and_mask, _mm_cvtsi32_si128(__bit_shift)));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::SSE2, xmm128>::__negate(_VectorType_ __vector) noexcept {
    if      constexpr (__is_ps_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_xor_ps(__vector, _mm_set1_ps(-0.0f)));

    else if constexpr (__is_pd_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_xor_pd(__vector,
            __intrin_bitcast<__m128d>(_mm_setr_epi32(0, 0x80000000, 0, 0x80000000))));

    else
        return __substract<_DesiredType_>(_SimdBroadcastZeros<__generation, __register_policy, _VectorType_>(), __vector);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::SSE2, xmm128>::__add(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_add_epi64(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));

    else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_add_epi32(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));

    else if constexpr (__is_epi16_v<_DesiredType_> || __is_epu16_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_add_epi16(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));

    else if constexpr (__is_epi8_v<_DesiredType_> || __is_epu8_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_add_epi8(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));

    else if constexpr (__is_ps_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_add_ps(__intrin_bitcast<__m128>(__left), __intrin_bitcast<__m128>(__right)));

    else if constexpr (__is_pd_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_add_pd(__intrin_bitcast<__m128d>(__left), __intrin_bitcast<__m128d>(__right)));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::SSE2, xmm128>::__substract(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_sub_epi64(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));

    else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_sub_epi32(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));

    else if constexpr (__is_epi16_v<_DesiredType_> || __is_epu16_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_sub_epi16(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));

    else if constexpr (__is_epi8_v<_DesiredType_> || __is_epu8_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_sub_epi8(__intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));

    else if constexpr (__is_ps_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_sub_ps(__intrin_bitcast<__m128>(__left), __intrin_bitcast<__m128>(__right)));

    else if constexpr (__is_pd_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_sub_pd(__intrin_bitcast<__m128d>(__left), __intrin_bitcast<__m128d>(__right)));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::SSE2, xmm128>::__multiply(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (__is_epi32_v<_DesiredType_>) {
        const auto __shuffled_left  = _mm_shuffle_epi32(__intrin_bitcast<__m128i>(__left), 0xF5);
        const auto __shuffled_right = _mm_shuffle_epi32(__intrin_bitcast<__m128i>(__right), 0xF5);

        const auto __product_even_indices   = _mm_mul_epu32(__left, __right);
        const auto __product_odd_indices    = _mm_mul_epu32(__shuffled_left, __shuffled_right);

        const auto __product_low_pair   = _mm_unpacklo_epi32(__product_even_indices, __product_odd_indices);
        const auto __product_high_pair  = _mm_unpackhi_epi32(__product_even_indices, __product_odd_indices);

        return __intrin_bitcast<_VectorType_>(_mm_unpacklo_epi64(__product_low_pair, __product_high_pair));
    }
    else if constexpr (__is_epu32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_mul_epu32(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else if constexpr (__is_ps_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_mul_ps(
            __intrin_bitcast<__m128>(__left), __intrin_bitcast<__m128>(__right)));
    }
    else if constexpr (__is_pd_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_mul_pd(
            __intrin_bitcast<__m128d>(__left), __intrin_bitcast<__m128d>(__right)));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::SSE2, xmm128>::__divide(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (__is_pd_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_div_pd(
            __intrin_bitcast<__m128d>(__left), __intrin_bitcast<__m128d>(__right)));

    else if constexpr (__is_ps_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_div_ps(
            __intrin_bitcast<__m128>(__left), __intrin_bitcast<__m128>(__right)));
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::SSE2, xmm128>::__bit_not(_VectorType_ __vector) noexcept {
    if      constexpr (std::is_same_v<_VectorType_, __m128d>)
        return _mm_xor_pd(__vector, _mm_cmpeq_pd(__vector, __vector));

    else if constexpr (std::is_same_v<_VectorType_, __m128i>)
        return _mm_xor_si128(__vector, _mm_cmpeq_epi32(__vector, __vector));

    else if constexpr (std::is_same_v<_VectorType_, __m128>)
        return _mm_xor_ps(__vector, _mm_cmpeq_ps(__vector, __vector));
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::SSE2, xmm128>::__bit_xor(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if      constexpr (std::is_same_v<_VectorType_, __m128d>)
        return _mm_xor_pd(__left, __right);

    else if constexpr (std::is_same_v<_VectorType_, __m128i>)
        return _mm_xor_si128(__left, __right);

    else if constexpr (std::is_same_v<_VectorType_, __m128>)
        return _mm_xor_ps(__left, __right);
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::SSE2, xmm128>::__bit_and(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if      constexpr (std::is_same_v<_VectorType_, __m128d>)
        return _mm_and_pd(__left, __right);

    else if constexpr (std::is_same_v<_VectorType_, __m128i>)
        return _mm_and_si128(__left, __right);

    else if constexpr (std::is_same_v<_VectorType_, __m128>)
        return _mm_and_ps(__left, __right);
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::SSE2, xmm128>::__bit_or(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if      constexpr (std::is_same_v<_VectorType_, __m128d>)
        return _mm_or_pd(__left, __right);

    else if constexpr (std::is_same_v<_VectorType_, __m128i>)
        return _mm_or_si128(__left, __right);

    else if constexpr (std::is_same_v<_VectorType_, __m128>)
        return _mm_or_ps(__left, __right);
}

template <
    typename _DesiredType_,
    typename _VectorType_,
    typename _ReduceBinaryFunction_>
simd_stl_always_inline _DesiredType_ __simd_arithmetic<arch::CpuFeature::SSSE3, xmm128>::__horizontal_fold(
    _VectorType_            __vector,
    _ReduceBinaryFunction_  __reduce) noexcept
{
    if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_> || __is_pd_v<_DesiredType_>) {
        auto __horizontal_folded_values = __intrin_bitcast<__m128d>(__vector);

        const auto __shuffled       = _mm_shuffle_pd(__horizontal_folded_values, __horizontal_folded_values, 1);
        __horizontal_folded_values  = __reduce(__shuffled, __horizontal_folded_values);

        if constexpr (__is_pd_v<_DesiredType_>)
            return _mm_cvtsd_f64(__horizontal_folded_values);
        else
            return _mm_cvtsi128_si64(__intrin_bitcast<__m128i>(__horizontal_folded_values));
    }
    else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_> || __is_ps_v<_DesiredType_>) {
        auto __horizontal_folded_values = __intrin_bitcast<__m128i>(__vector);

        const auto __shuffled1      = _mm_shuffle_epi32(__horizontal_folded_values, 0x4E);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffled1);

        const auto __shuffled2      = _mm_shuffle_epi32(__horizontal_folded_values, 0xB1);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffled2);

        if constexpr (__is_ps_v<_DesiredType_>)
            return _mm_cvtss_f32(__intrin_bitcast<__m128>(__horizontal_folded_values));
        else
            return _mm_cvtsi128_si32(__horizontal_folded_values);
    }
    else if constexpr (__is_epi16_v<_DesiredType_> || __is_epu16_v<_DesiredType_>) {
        const auto __shuffle_words = _mm_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2);

        auto __horizontal_folded_values = __intrin_bitcast<__m128i>(__vector);
        
        const auto __shuffled1      = _mm_shuffle_epi32(__horizontal_folded_values, 0x4E);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffled1);

        const auto __shuffle2       = _mm_shuffle_epi32(__horizontal_folded_values, 0xB1);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffle2);

        const auto __shuffle3       = _mm_shuffle_epi8(__horizontal_folded_values, __shuffle_words);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffle3);

        return _mm_cvtsi128_si32(__horizontal_folded_values);
    }
    else if constexpr (__is_epi8_v<_DesiredType_> || __is_epu8_v<_DesiredType_>) {
        const auto __shuffle_bytes = _mm_set_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
        const auto __shuffle_words = _mm_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2);

        auto __horizontal_folded_values = __intrin_bitcast<__m128i>(__vector);

        const auto __shuffled1      = _mm_shuffle_epi32(__horizontal_folded_values, 0x4E);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffled1);

        const auto __shuffled2      = _mm_shuffle_epi32(__horizontal_folded_values, 0xB1);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffled2);

        const auto __shuffled3      = _mm_shuffle_epi8(__horizontal_folded_values, __shuffle_words);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffled3);

        const auto __shuffled4      = _mm_shuffle_epi8(__horizontal_folded_values, __shuffle_bytes);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffled4);

        return _mm_cvtsi128_si32(__horizontal_folded_values);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_ __simd_arithmetic<arch::CpuFeature::SSSE3, xmm128>::__horizontal_min(_VectorType_ __vector) noexcept {
    return __horizontal_fold<_DesiredType_>(__vector, __vertical_min_wrapper<__generation, __register_policy, _DesiredType_>{});
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_ __simd_arithmetic<arch::CpuFeature::SSSE3, xmm128>::__horizontal_max(_VectorType_ __vector) noexcept {
    return __horizontal_fold<_DesiredType_>(__vector, __vertical_max_wrapper<__generation, __register_policy, _DesiredType_>{});
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline auto __simd_arithmetic<arch::CpuFeature::SSSE3, xmm128>::__reduce(_VectorType_ __vector) noexcept {
    using _ReduceType = __reduce_type<_DesiredType_>;

    if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>) {
        const auto __zeros = _mm_setzero_si128();

        const auto __reduce4 = _mm_hadd_epi32(__intrin_bitcast<__m128i>(__vector), __zeros); // (0+1),(2+3),0,0
        const auto __reduce5 = _mm_hadd_epi32(__reduce4, __zeros);                         // (0+...+3),0,0,0

        return static_cast<_ReduceType>(_mm_cvtsi128_si32(__reduce5));
    }
    else if constexpr (__is_epi16_v<_DesiredType_> || __is_epu16_v<_DesiredType_>) {
        const auto __zeros = _mm_setzero_si128();

        const auto __reduce2 = _mm_hadd_epi16(__intrin_bitcast<__m128i>(__vector), __zeros);
        const auto __reduce3 = _mm_unpacklo_epi16(__reduce2, __zeros);

        const auto __reduce4 = _mm_hadd_epi32(__reduce3, __zeros); // (0+1),(2+3),0,0
        const auto __reduce5 = _mm_hadd_epi32(__reduce4, __zeros); // (0+...+3),0,0,0

        return static_cast<_ReduceType>(_mm_cvtsi128_si32(__reduce5));
    }
    else {
        return __simd_reduce<arch::CpuFeature::SSE2, __register_policy, _DesiredType_>(__vector);
    }
}


template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::SSE41, xmm128>::__multiply(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (__is_epi32_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_mul_epi32(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));

    else if constexpr (__is_epu32_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_mul_epu32(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));

    else if constexpr (__is_ps_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_mul_ps(
            __intrin_bitcast<__m128>(__left), __intrin_bitcast<__m128>(__right)));

    else if constexpr (__is_pd_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_mul_pd(
            __intrin_bitcast<__m128d>(__left), __intrin_bitcast<__m128d>(__right)));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_ __simd_arithmetic<arch::CpuFeature::SSE41, xmm128>::__horizontal_min(_VectorType_ __vector) noexcept {
    return __horizontal_fold<_DesiredType_>(__vector, __vertical_min_wrapper<__generation, __register_policy, _DesiredType_>{});
}

template <
    typename _DesiredType_,
    typename _VectorType_>
static simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::SSE41, xmm128>::__vertical_min(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (__is_epi32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_min_epi32(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else if constexpr (__is_epu32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_min_epu32(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else if constexpr (__is_epu16_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_min_epu16(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else if constexpr (__is_epi8_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_min_epi8(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else {
        return __simd_vertical_min<arch::CpuFeature::SSE2, __register_policy, _DesiredType_>(__left, __right);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
static simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::SSE41, xmm128>::__vertical_max(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (__is_epi32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_max_epi32(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else if constexpr (__is_epu32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_max_epu32(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else if constexpr (__is_epu16_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_max_epu16(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else if constexpr (__is_epi8_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_max_epi8(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else {
        return __simd_vertical_max<arch::CpuFeature::SSE2, __register_policy, _DesiredType_>(__left, __right);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_ __simd_arithmetic<arch::CpuFeature::SSE41, xmm128>::__horizontal_max(_VectorType_ __vector) noexcept {
    return __horizontal_fold<_DesiredType_>(__vector, __vertical_max_wrapper<__generation, __register_policy, _DesiredType_>{});
}

#pragma endregion 

#pragma region Avx Simd arithmetic

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX2, ymm256>::__abs(_VectorType_ __vector) noexcept {
    if constexpr (std::is_unsigned_v<_DesiredType_>) {
        return __vector;
    }
    else if constexpr (__is_epi64_v<_DesiredType_>) {
        const auto __sign        = _mm256_cmpgt_epi64(_mm256_setzero_si256(), __vector);
        const auto __inverted    = _mm256_xor_si256(__intrin_bitcast<__m256i>(__vector), __sign);

        return __intrin_bitcast<_VectorType_>(_mm256_sub_epi64(__inverted, __sign));
    }
    else if constexpr (__is_epi32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_abs_epi32(__intrin_bitcast<__m256i>(__vector)));
    }
    else if constexpr (__is_epi16_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_abs_epi16(__intrin_bitcast<__m256i>(__vector)));
    }
    else if constexpr (__is_epi8_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_abs_epi8(__intrin_bitcast<__m256i>(__vector)));
    }
    else if constexpr (__is_pd_v<_DesiredType_>) {
        const auto __mask = _mm256_set_epi32(0xFFFFFFFFu, 0x7FFFFFFFu, 0xFFFFFFFFu, 0x7FFFFFFFu,
            0xFFFFFFFFu, 0x7FFFFFFFu, 0xFFFFFFFFu, 0x7FFFFFFFu);
        return __intrin_bitcast<_VectorType_>(_mm256_and_pd(
            __intrin_bitcast<__m256d>(__vector), __intrin_bitcast<__m256d>(__mask)));
    }
    else if constexpr (__is_ps_v<_DesiredType_>) {
        const auto __mask = _mm256_set1_epi32(0x7FFFFFFFu);
        return __intrin_bitcast<_VectorType_>(_mm256_and_ps(
            __intrin_bitcast<__m256>(__vector), __intrin_bitcast<__m256>(__mask)));
    }
}


template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX2, ymm256>::__vertical_min(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (__is_epi32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_min_epi32(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));
    }
    else if constexpr (__is_epu32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_min_epu32(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));
    }
    else if constexpr (__is_epi16_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_min_epi16(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));
    }
    else if constexpr (__is_epu16_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_min_epu16(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));
    }
    else if constexpr (__is_epi8_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_min_epi8(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));
    }
    else if constexpr (__is_epu8_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_min_epu8(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));
    }
    else if constexpr (__is_ps_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_min_ps(
            __intrin_bitcast<__m256>(__left), __intrin_bitcast<__m256>(__right)));
    }
    else if constexpr (__is_ps_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_min_pd(
            __intrin_bitcast<__m256d>(__left), __intrin_bitcast<__m256d>(__right)));
    }
    else {
        const auto __mask = __simd_compare<__generation, __register_policy, _DesiredType_, __simd_comparison::less>(__left, __right);
        return __simd_blend<__generation, __register_policy, _DesiredType_>(__left, __right, __mask);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_ __simd_arithmetic<arch::CpuFeature::AVX2, ymm256>::__horizontal_min(_VectorType_ __vector) noexcept {
    return __horizontal_fold<_DesiredType_>(__vector, __vertical_min_wrapper<__generation, __register_policy, _DesiredType_>{});
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX2, ymm256>::__vertical_max(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (__is_epi32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_max_epi32(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));
    }
    else if constexpr (__is_epu32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_max_epu32(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));
    }
    else if constexpr (__is_epi16_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_max_epi16(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));
    }
    else if constexpr (__is_epu16_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_max_epu16(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));
    }
    else if constexpr (__is_epi8_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_max_epi8(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));
    }
    else if constexpr (__is_epu8_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_max_epu8(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));
    }
    else if constexpr (__is_ps_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_max_ps(
            __intrin_bitcast<__m256>(__left), __intrin_bitcast<__m256>(__right)));
    }
    else if constexpr (__is_ps_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_max_pd(
            __intrin_bitcast<__m256d>(__left), __intrin_bitcast<__m256d>(__right)));
    }
    else {
        const auto __mask = __simd_compare<__generation, __register_policy, _DesiredType_, __simd_comparison::greater>(__left, __right);
        return __simd_blend<__generation, __register_policy, _DesiredType_>(__left, __right, __mask);
    }
}


template <
    typename _DesiredType_,
    typename _VectorType_,
    typename _ReduceBinaryFunction_>
simd_stl_always_inline _DesiredType_ __simd_arithmetic<arch::CpuFeature::AVX2, ymm256>::__horizontal_fold(
    _VectorType_            __vector,
    _ReduceBinaryFunction_  __reduce) noexcept
{
    if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_> || __is_pd_v<_DesiredType_>) {
        auto __horizontal_folded_values = __intrin_bitcast<__m256d>(__vector);
        
        const auto __shuffle1       = _mm256_shuffle_pd(__horizontal_folded_values, __horizontal_folded_values, 5);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffle1);

        const auto __shuffle2       = _mm256_permute4x64_epi64(__intrin_bitcast<__m256i>(__horizontal_folded_values), 0x1B);
        __horizontal_folded_values  = __intrin_bitcast<__m256d>(__reduce(
            __intrin_bitcast<__m256i>(__horizontal_folded_values), __intrin_bitcast<__m256i>(__shuffle2)));

        if constexpr (__is_pd_v<_DesiredType_>)
            return _mm256_cvtsd_f64(__horizontal_folded_values);
        else 
            return _mm256_cvtsi256_si64(__intrin_bitcast<__m256i>(__horizontal_folded_values));
    }
    else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_> || __is_ps_v<_DesiredType_>) {
        auto __horizontal_folded_values = __intrin_bitcast<__m256i>(__vector);

        const auto __shuffle1       = _mm256_permute4x64_epi64(__horizontal_folded_values, 0x4E);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffle1);

        const auto __shuffle2       = _mm256_shuffle_epi32(__horizontal_folded_values, 0x4E);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffle2);

        const auto __shuffle3       = _mm256_shuffle_epi32(__horizontal_folded_values, 0xB1);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffle3);

        if constexpr (__is_ps_v<_DesiredType_>)
            return _mm256_cvtss_f32(__intrin_bitcast<__m256>(__horizontal_folded_values));
        else 
            return _mm256_cvtsi256_si32(__horizontal_folded_values);
    }
    else if constexpr (__is_epi16_v<_DesiredType_> || __is_epu16_v<_DesiredType_>) {
        const auto __shuffle_words = _mm256_broadcastsi128_si256(_mm_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2));

        auto __horizontal_folded_values = __intrin_bitcast<__m256i>(__vector);

        const auto __shuffle1       = _mm256_permute4x64_epi64(__horizontal_folded_values, 0x4E);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffle1);

        const auto __shuffle2       = _mm256_shuffle_epi32(__horizontal_folded_values, 0x4E);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffle2);

        const auto __shuffle3       = _mm256_shuffle_epi32(__horizontal_folded_values, 0xB1);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffle3);

        const auto __shuffle4       = _mm256_shuffle_epi8(__horizontal_folded_values, __shuffle_words);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffle4);

        return _mm256_cvtsi256_si32(__horizontal_folded_values);
    }
    else if constexpr (__is_epi8_v<_DesiredType_> || __is_epu8_v<_DesiredType_>) {
        const auto __shuffle_words = _mm256_broadcastsi128_si256(_mm_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2));
        const auto __shuffle_bytes = _mm256_broadcastsi128_si256(_mm_set_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1));

        auto __horizontal_folded_values = __intrin_bitcast<__m256i>(__vector);
      
        const auto __shuffle1       = _mm256_permute4x64_epi64(__horizontal_folded_values, 0x4E);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffle1);

        const auto __shuffle2       = _mm256_shuffle_epi32(__horizontal_folded_values, 0x4E);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffle2);

        const auto __shuffle3       = _mm256_shuffle_epi32(__horizontal_folded_values, 0xB1);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffle3);

        const auto __shuffle4       = _mm256_shuffle_epi8(__horizontal_folded_values, __shuffle_words);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffle4);

        const auto __shuffle5       = _mm256_shuffle_epi8(__horizontal_folded_values, __shuffle_bytes);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffle5);

        return _mm256_cvtsi256_si32(__horizontal_folded_values);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_ __simd_arithmetic<arch::CpuFeature::AVX2, ymm256>::__horizontal_max(_VectorType_ __vector) noexcept {
    return __horizontal_fold<_DesiredType_>(__vector, __vertical_max_wrapper<__generation, __register_policy, _DesiredType_>{});
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline auto __simd_arithmetic<arch::CpuFeature::AVX2, ymm256>::__reduce(_VectorType_ __vector) noexcept {
    using _ReduceType = __reduce_type<_DesiredType_>;

    if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>) {
        const auto __low64   = __intrin_bitcast<__m128i>(__vector);
        const auto __high64  = _mm256_extracti128_si256(__intrin_bitcast<__m256i>(__vector), 1);

        const auto __reduce  = _mm_add_epi64(__low64, __high64);
        return __simd_reduce<arch::CpuFeature::SSSE3, xmm128, _DesiredType_>(__reduce);
    }
    if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>) {
        const auto __zeros = _mm256_setzero_si256();

        const auto __reduce4 = _mm256_hadd_epi32(__intrin_bitcast<__m256i>(__vector), __zeros); // (0+1),(2+3),0,0
        const auto __reduce5 = _mm256_permute4x64_epi64(__reduce4, 0xD8); // low lane  (0+1),(2+3),(4+5),(6+7)

        const auto __reduce6 = _mm256_hadd_epi32(__reduce5, __zeros); // (0+...+3),(4+...+7),0,0
        const auto __reduce7 = _mm256_hadd_epi32(__reduce6, __zeros); // (0+...+7),0,0,0

        return static_cast<_ReduceType>(_mm_cvtsi128_si32(__intrin_bitcast<__m128i>(__reduce7)));
    }
    else if constexpr (__is_epi16_v<_DesiredType_> || __is_epu16_v<_DesiredType_>) {
        const auto __zeros = _mm256_setzero_si256();

        const auto __reduce2 = _mm256_hadd_epi16(__intrin_bitcast<__m256i>(__vector), __zeros);
        const auto __reduce3 = _mm256_unpacklo_epi16(__reduce2, __zeros);

        const auto __reduce4 = _mm256_hadd_epi32(__reduce3, __zeros); // (0+1),(2+3),0,0
        const auto __reduce5 = _mm256_permute4x64_epi64(__reduce4, 0xD8); // (0+1),(2+3),(4+5),(6+7)

        const auto __reduce6 = _mm256_hadd_epi32(__reduce5, __zeros); // (0+...+3),(4+...+7),0,0
        const auto __reduce7 = _mm256_hadd_epi32(__reduce6, __zeros); // (0+...+7),0,0,0

        return static_cast<_ReduceType>(_mm_cvtsi128_si32(__intrin_bitcast<__m128i>(__reduce7)));
    }
    else if constexpr (__is_epi8_v<_DesiredType_> || __is_epu8_v<_DesiredType_>) {
        const auto __reduce1 = _mm256_sad_epu8(__intrin_bitcast<__m256i>(__vector), _mm256_setzero_si256());

        const auto __low64  = _mm256_castsi256_si128(__reduce1);
        const auto __high64 = _mm256_extracti128_si256(__reduce1, 1);

        const auto __reduce8 = _mm_add_epi64(__low64, __high64);
        return __simd_reduce<arch::CpuFeature::SSSE3, xmm128, int64>(__reduce8);
    }
    else {
        constexpr auto __length = sizeof(_VectorType_) / sizeof(_DesiredType_);

        _DesiredType_ __array[__length];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&__array), __intrin_bitcast<__m256i>(__vector));

        auto __sum = _ReduceType(0);

        for (auto __index = 0; __index < __length; ++__index)
            __sum += __array[__index];

        return __sum;
    }
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX2, ymm256>::__shift_right_vector(
    _VectorType_    __vector,
    uint32          __byte_shift) noexcept
{
    return __intrin_bitcast<_VectorType_>(_mm256_srli_si256(__intrin_bitcast<__m256i>(__vector), __byte_shift));
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX2, ymm256>::__shift_left_vector(
    _VectorType_    __vector,
    uint32          __byte_shift) noexcept
{
    return __intrin_bitcast<_VectorType_>(_mm256_slli_si256(__intrin_bitcast<__m256i>(__vector), __byte_shift));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX2, ymm256>::__shift_right_elements(
    _VectorType_    __vector,
    uint32          __bit_shift) noexcept
{
    if      constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_srli_epi64(__intrin_bitcast<__m256i>(__vector), __bit_shift));

    else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_srli_epi32(__intrin_bitcast<__m256i>(__vector), __bit_shift));

    else if constexpr (__is_epi16_v<_DesiredType_> || __is_epu16_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_srli_epi16(__intrin_bitcast<__m256i>(__vector), __bit_shift));

    else if constexpr (__is_epi8_v<_DesiredType_> || __is_epu8_v<_DesiredType_>) {
        const auto __even_vector = _mm256_sra_epi16(_mm256_slli_epi16(
            __intrin_bitcast<__m256i>(__vector), 8), _mm_cvtsi32_si128(__bit_shift + 8));

        const auto __odd_vector = _mm256_sra_epi16(__intrin_bitcast<__m256i>(__vector), _mm_cvtsi32_si128(__bit_shift));

        const auto __mask = _mm256_set1_epi32(0x00FF00FF);
        return __simd_blend<__generation, __register_policy, _DesiredType_>(__even_vector, __odd_vector, __mask);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX2, ymm256>::__shift_left_elements(
    _VectorType_    __vector,
    uint32          __bit_shift) noexcept
{
    if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_slli_epi64(__intrin_bitcast<__m256i>(__vector), __bit_shift));
    }
    else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_slli_epi32(__intrin_bitcast<__m256i>(__vector), __bit_shift));
    }
    else if constexpr (__is_epi16_v<_DesiredType_> || __is_epu16_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_slli_epi16(__intrin_bitcast<__m256i>(__vector), __bit_shift));
    }
    else if constexpr (__is_epi8_v<_DesiredType_> || __is_epu8_v<_DesiredType_>) {
        const auto __and_mask = _mm256_and_si256(__intrin_bitcast<__m256i>(__vector),
            _mm256_set1_epi8(static_cast<int8>(0xFFu >> __bit_shift)));

        return __intrin_bitcast<_VectorType_>(_mm256_sll_epi16(__and_mask, _mm_cvtsi32_si128(__bit_shift)));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX2, ymm256>::__negate(_VectorType_ __vector) noexcept {
    if      constexpr (__is_ps_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_xor_ps(__intrin_bitcast<__m256>(__vector), _mm256_set1_ps(-0.0f)));

    else if constexpr (__is_pd_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_xor_pd(__intrin_bitcast<__m256d>(__vector),
            __intrin_bitcast<__m256d>(_mm256_setr_epi32(0, 0x80000000, 0, 0x80000000, 0, 0x80000000, 0, 0x80000000))));

    else
        return __substract<_DesiredType_>(__simd_broadcast_zeros<__generation, __register_policy, _VectorType_>(), __vector);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX2, ymm256>::__add(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_add_epi64(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));

    else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_add_epi32(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));

    else if constexpr (__is_epi16_v<_DesiredType_> || __is_epu16_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_add_epi16(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));

    else if constexpr (__is_epi8_v<_DesiredType_> || __is_epu8_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_add_epi8(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));

    else if constexpr (__is_ps_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_add_ps(
            __intrin_bitcast<__m256>(__left), __intrin_bitcast<__m256>(__right)));

    else if constexpr (__is_pd_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_add_pd(
            __intrin_bitcast<__m256d>(__left), __intrin_bitcast<__m256d>(__right)));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX2, ymm256>::__substract(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_sub_epi64(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));

    else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_sub_epi32(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));

    else if constexpr (__is_epi16_v<_DesiredType_> || __is_epu16_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_sub_epi16(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));

    else if constexpr (__is_epi8_v<_DesiredType_> || __is_epu8_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_sub_epi8(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));

    else if constexpr (__is_ps_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_sub_ps(
            __intrin_bitcast<__m256>(__left), __intrin_bitcast<__m256>(__right)));

    else if constexpr (__is_pd_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_sub_pd(
            __intrin_bitcast<__m256d>(__left), __intrin_bitcast<__m256d>(__right)));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX2, ymm256>::__multiply(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (__is_epi32_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_mul_epi32(__intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));

    else if constexpr (__is_epu32_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_mul_epu32(__intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));

    else if constexpr (__is_ps_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_mul_ps(__intrin_bitcast<__m256>(__left), __intrin_bitcast<__m256>(__right)));

    else if constexpr (__is_pd_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_mul_pd(__intrin_bitcast<__m256d>(__left), __intrin_bitcast<__m256d>(__right)));

    else {

    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX2, ymm256>::__divide(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (__is_pd_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_div_pd(__intrin_bitcast<__m256d>(__left), __intrin_bitcast<__m256d>(__right)));

    else if constexpr (__is_ps_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_div_ps(__intrin_bitcast<__m256>(__left), __intrin_bitcast<__m256>(__right)));
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX2, ymm256>::__bit_not(_VectorType_ __vector) noexcept {
    if      constexpr (std::is_same_v<_VectorType_, __m256d>)
        return _mm256_xor_pd(__vector, _mm256_cmp_pd(__vector, __vector, _CMP_EQ_OQ));

    else if constexpr (std::is_same_v<_VectorType_, __m256i>)
        return _mm256_xor_si256(__vector, _mm256_cmpeq_epi32(__vector, __vector));

    else if constexpr (std::is_same_v<_VectorType_, __m256>)
        return _mm256_xor_ps(__vector, _mm256_cmp_ps(__vector, __vector, _CMP_EQ_OQ));
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX2, ymm256>::__bit_xor(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if      constexpr (std::is_same_v<_VectorType_, __m256d>)
        return _mm256_xor_pd(__left, __right);

    else if constexpr (std::is_same_v<_VectorType_, __m256i>)
        return _mm256_xor_si256(__left, __right);

    else if constexpr (std::is_same_v<_VectorType_, __m256>)
        return _mm256_xor_ps(__left, __right);
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX2, ymm256>::__bit_and(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if      constexpr (std::is_same_v<_VectorType_, __m256d>)
        return _mm256_and_pd(__left, __right);

    else if constexpr (std::is_same_v<_VectorType_, __m256i>)
        return _mm256_and_si256(__left, __right);

    else if constexpr (std::is_same_v<_VectorType_, __m256>)
        return _mm256_and_ps(__left, __right);
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX2, ymm256>::__bit_or(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if      constexpr (std::is_same_v<_VectorType_, __m256d>)
        return _mm256_or_pd(__left, __right);

    else if constexpr (std::is_same_v<_VectorType_, __m256i>)
        return _mm256_or_si256(__left, __right);

    else if constexpr (std::is_same_v<_VectorType_, __m256>)
        return _mm256_or_ps(__left, __right);
}

#pragma endregion

#pragma region Avx512 Simd arithmetic

template <
    typename _DesiredType_,
    typename _VectorType_,
    typename _ReduceBinaryFunction_>
simd_stl_always_inline _DesiredType_ __simd_arithmetic<arch::CpuFeature::AVX512F, zmm512>::__horizontal_fold(
    _VectorType_            __vector,
    _ReduceBinaryFunction_  __reduce) noexcept
{
    if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_> || __is_pd_v<_DesiredType_>) {
        auto __horizontal_folded_values = __intrin_bitcast<__m512i>(__vector);

        const auto __shuffled1      = _mm512_permutexvar_epi64(_mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0), __horizontal_folded_values);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffled1);

        const auto __shuffled2      = _mm512_permutex_epi64(__horizontal_folded_values, 0x4E);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffled2);

        const auto __shuffled3      = _mm512_permutex_epi64(__horizontal_folded_values, 0xB1);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffled3);

        if constexpr (__is_pd_v<_DesiredType_>)
            return _mm512_cvtsd_f64(__intrin_bitcast<__m512d>(__horizontal_folded_values));
        else
            return _mm512_cvtsi512_si64(__horizontal_folded_values);
    }
    else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_> || __is_ps_v<_DesiredType_>) {
        auto __horizontal_folded_values = __intrin_bitcast<__m512i>(__vector);

        const auto __shuffled1      = _mm512_permutexvar_epi64(_mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0), __horizontal_folded_values);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffled1);

        const auto __shuffled2      = _mm512_permutex_epi64(__horizontal_folded_values, 0x4E);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffled2);

        const auto __shuffled3      = _mm512_permutex_epi64(__horizontal_folded_values, 0xB1);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffled3);

        const auto __shuffled4      = __intrin_bitcast<__m512i>(_mm512_permute_ps(__intrin_bitcast<__m512>(__horizontal_folded_values), 0xB1));
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffled4);

        if constexpr (__is_ps_v<_DesiredType_>)
            return _mm512_cvtss_f32(__intrin_bitcast<__m512>(__horizontal_folded_values));
        else
            return _mm512_cvtsi512_si32(__horizontal_folded_values);
    }
    else if constexpr (__is_epi16_v<_DesiredType_> || __is_epu16_v<_DesiredType_>) {
        const auto __shuffle_words = _mm256_broadcastsi128_si256(_mm_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2));
        auto __horizontal_folded_values = __intrin_bitcast<__m512i>(__vector);

        const auto __shuffled1      = _mm512_permutexvar_epi64(_mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0), __horizontal_folded_values);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffled1);

        const auto __shuffled2      = _mm512_permutex_epi64(__horizontal_folded_values, 0x4E);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffled2);

        const auto __shuffled3      = _mm512_permutex_epi64(__horizontal_folded_values, 0xB1);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffled3);

        const auto __shuffled4      = __intrin_bitcast<__m512i>(_mm512_permute_ps(__intrin_bitcast<__m512>(__horizontal_folded_values), 0xB1));
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffled4);

        const auto __shuffled5_low  = _mm256_shuffle_epi8(__intrin_bitcast<__m256i>(__horizontal_folded_values), __shuffle_words);
        const auto __shuffled5_high = _mm256_shuffle_epi8(__intrin_bitcast<__m256i>(_mm512_extractf64x4_pd(
            __intrin_bitcast<__m512d>(__horizontal_folded_values), 1)), __shuffle_words);
        
        auto __shuffled5    = __intrin_bitcast<__m512d>(__shuffled5_low);
        __shuffled5         = _mm512_insertf64x4(__shuffled5, __intrin_bitcast<__m256d>(__shuffled5_high), 1);

        __horizontal_folded_values = __reduce(__horizontal_folded_values, __intrin_bitcast<__m512i>(__shuffled5));

        return _mm512_cvtsi512_si32(__horizontal_folded_values);
    }
    else if constexpr (__is_epi8_v<_DesiredType_> || __is_epu8_v<_DesiredType_>) {
        const auto __shuffle_words = _mm256_broadcastsi128_si256(_mm_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2));
        const auto __shuffle_bytes = _mm256_broadcastsi128_si256(_mm_set_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1));

        auto __horizontal_folded_values = __intrin_bitcast<__m512i>(__vector);

        const auto __shuffled1      = _mm512_permutexvar_epi64(_mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0), __horizontal_folded_values);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffled1);

        const auto __shuffled2      = _mm512_permutex_epi64(__horizontal_folded_values, 0x4E);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffled2);

        const auto __shuffled3      = _mm512_permutex_epi64(__horizontal_folded_values, 0xB1);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffled3);

        const auto __shuffled4      = __intrin_bitcast<__m512i>(_mm512_permute_ps(__intrin_bitcast<__m512>(__horizontal_folded_values), 0xB1));
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffled4);

        const auto __shuffled5_low  = _mm256_shuffle_epi8(__intrin_bitcast<__m256i>(__horizontal_folded_values), __shuffle_words);
        const auto __shuffled5_high = _mm256_shuffle_epi8(__intrin_bitcast<__m256i>(_mm512_extractf64x4_pd(
            __intrin_bitcast<__m512d>(__horizontal_folded_values), 1)), __shuffle_words);
        
        auto __shuffled5    = __intrin_bitcast<__m512d>(__shuffled5_low);
        __shuffled5         = _mm512_insertf64x4(__shuffled5, __intrin_bitcast<__m256d>(__shuffled5_high), 1);

        __horizontal_folded_values = __reduce(__horizontal_folded_values, __intrin_bitcast<__m512i>(__shuffled5));

        const auto __shuffled6_low  = _mm256_shuffle_epi8(__intrin_bitcast<__m256i>(__horizontal_folded_values), __shuffle_bytes);
        const auto __shuffled6_high = _mm256_shuffle_epi8(__intrin_bitcast<__m256i>(_mm512_extractf64x4_pd(
            __intrin_bitcast<__m512d>(__horizontal_folded_values), 1)), __shuffle_bytes);

        auto __shuffled6    = __intrin_bitcast<__m512d>(__shuffled6_low);
        __shuffled6         = _mm512_insertf64x4(__shuffled6, __intrin_bitcast<__m256d>(__shuffled6_high), 1);

        __horizontal_folded_values = __reduce(__horizontal_folded_values, __intrin_bitcast<__m512i>(__shuffled6));

        return _mm512_cvtsi512_si32(__horizontal_folded_values);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX512F, zmm512>::__vertical_min(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (__is_epi64_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_min_epi64(
            __intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right)));
    }
    else if constexpr (__is_epu64_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_min_epu64(
            __intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right)));
    }
    else if constexpr (__is_epi32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_min_epi32(
            __intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right)));
    }
    else if constexpr (__is_epu32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_min_epu32(
            __intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right)));
    }
    else if constexpr (__is_ps_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_min_ps(
            __intrin_bitcast<__m512>(__left), __intrin_bitcast<__m512>(__right)));
    }
    else if constexpr (__is_pd_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_min_pd(
            __intrin_bitcast<__m512d>(__left), __intrin_bitcast<__m512d>(__right)));
    }
    else {
        const auto __mask = __simd_compare<__generation, __register_policy, _DesiredType_, __simd_comparison::less>(__left, __right);
        return __simd_blend<__generation, __register_policy, _DesiredType_>(__left, __right, __mask);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_ __simd_arithmetic<arch::CpuFeature::AVX512F, zmm512>::__horizontal_min(_VectorType_ __vector) noexcept {
    return __horizontal_fold<_DesiredType_>(__vector, __vertical_min_wrapper<__generation, __register_policy, _DesiredType_>{});
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX512F, zmm512>::__vertical_max(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (__is_epi64_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_max_epi64(
            __intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right)));
    }
    else if constexpr (__is_epu64_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_max_epu64(
            __intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right)));
    }
    else if constexpr (__is_epi32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_max_epi32(
            __intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right)));
    }
    else if constexpr (__is_epu32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_max_epu32(
            __intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right)));
    }
    else if constexpr (__is_ps_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_max_ps(
            __intrin_bitcast<__m512>(__left), __intrin_bitcast<__m512>(__right)));
    }
    else if constexpr (__is_pd_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_max_pd(
            __intrin_bitcast<__m512d>(__left), __intrin_bitcast<__m512d>(__right)));
    }
    else {
        const auto __mask = __simd_compare<__generation, __register_policy, _DesiredType_, __simd_comparison::greater>(__left, __right);
        return __simd_blend<__generation, __register_policy, _DesiredType_>(__left, __right, __mask);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_ __simd_arithmetic<arch::CpuFeature::AVX512F, zmm512>::__horizontal_max(_VectorType_ __vector) noexcept {
    return __horizontal_fold<_DesiredType_>(__vector, __vertical_max_wrapper<__generation, __register_policy, _DesiredType_>{});
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX512F, zmm512>::__abs(_VectorType_ __vector) noexcept {
    if constexpr (std::is_unsigned_v<_DesiredType_>) {
        return __vector;
    }
    else if constexpr (__is_epi64_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_abs_epi64(__intrin_bitcast<__m512i>(__vector)));
    }
    else if constexpr (__is_epi32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_abs_epi32(__intrin_bitcast<__m512i>(__vector)));
    }
    else if constexpr (__is_ps_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_abs_ps(__intrin_bitcast<__m512>(__vector)));
    }
    else if constexpr (__is_pd_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_abs_pd(__intrin_bitcast<__m512d>(__vector)));
    }
    else {
        const auto __low     = __intrin_bitcast<__m256i>(__vector);
        const auto __high    = _mm512_extractf64x4_pd(__intrin_bitcast<__m512d>(__vector), 1);

        auto __result = __intrin_bitcast<__m512i>(__simd_abs<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(__low));
        __result = __intrin_bitcast<__m512i>(_mm512_insertf64x4(__intrin_bitcast<__m512d>(__result), 
            __intrin_bitcast<__m256d>(__simd_abs<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(__high)), 1));

        return __intrin_bitcast<_VectorType_>(__result);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline auto __simd_arithmetic<arch::CpuFeature::AVX512F, zmm512>::__reduce(_VectorType_ __vector) noexcept {
    if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>) {
        return _mm512_reduce_add_epi64(__intrin_bitcast<__m512i>(__vector));
    }
    else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>) {
        return _mm512_reduce_add_epi32(__intrin_bitcast<__m512i>(__vector));
    }
    else if constexpr (__is_ps_v<_DesiredType_>) {
        return _mm512_reduce_add_ps(__intrin_bitcast<__m512>(__vector));
    }
    else if constexpr (__is_pd_v<_DesiredType_>) {
        return _mm512_reduce_add_pd(__intrin_bitcast<__m512d>(__vector));
    }
    else if constexpr (__is_epi8_v<_DesiredType_> || __is_epu8_v<_DesiredType_>) {
        return __simd_reduce<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(__intrin_bitcast<__m256i>(__vector)) +
            __simd_reduce<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(_mm512_extractf64x4_pd(__intrin_bitcast<__m512d>(__vector), 1));
    }
    else if constexpr (__is_epi16_v<_DesiredType_> || __is_epu16_v<_DesiredType_>) {
        return __simd_reduce<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(__intrin_bitcast<__m256i>(__vector)) +
            __simd_reduce<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(_mm512_extractf64x4_pd(__intrin_bitcast<__m512d>(__vector), 1));
    }
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX512F, zmm512>::__shift_right_vector(
    _VectorType_    __vector,
    uint32          __byte_shift) noexcept
{
    return __vector;
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX512F, zmm512>::__shift_left_vector(
    _VectorType_    __vector,
    uint32          __byte_shift) noexcept
{
    return __vector;
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX512F, zmm512>::__shift_right_elements(
    _VectorType_    __vector,
    uint32          __bit_shift) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        return __intrin_bitcast<_VectorType_>(_mm512_srli_epi64(__intrin_bitcast<__m512i>(__vector), __bit_shift));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        return __intrin_bitcast<_VectorType_>(_mm512_srli_epi32(__intrin_bitcast<__m512i>(__vector), __bit_shift));
    }
    else {
        const auto __low = __simd_shift_right_elements<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(
            __intrin_bitcast<__m256i>(__vector), __bit_shift);

        const auto __high = __simd_shift_right_elements<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(
            _mm512_extracti64x4_epi64(__intrin_bitcast<__m512i>(__vector), 1), __bit_shift);

        return __intrin_bitcast<_VectorType_>(_mm512_inserti64x4(__intrin_bitcast<__m512i>(__low), __high, 1));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX512F, zmm512>::__shift_left_elements(
    _VectorType_    __vector,
    uint32          __bit_shift) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        return __intrin_bitcast<_VectorType_>(_mm512_slli_epi64(__intrin_bitcast<__m512i>(__vector), __bit_shift));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        return __intrin_bitcast<_VectorType_>(_mm512_slli_epi32(__intrin_bitcast<__m512i>(__vector), __bit_shift));
    }
    else {
        const auto __low = __simd_shift_left_elements<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(
            __intrin_bitcast<__m256i>(__vector), __bit_shift);

        const auto __high = __simd_shift_left_elements<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(
            _mm512_extracti64x4_epi64(__intrin_bitcast<__m512i>(__vector), 1), __bit_shift);

        return __intrin_bitcast<_VectorType_>(_mm512_inserti64x4(__intrin_bitcast<__m512i>(__low), __high, 1));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX512F, zmm512>::__negate(_VectorType_ __vector) noexcept {
    if      constexpr (__is_ps_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm512_xor_ps(__vector, _mm512_set1_ps(-0.0f)));

    else if constexpr (__is_pd_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm512_xor_pd(__vector,
            __intrin_bitcast<__m512d>(_mm512_setr_epi32(0, 0x80000000, 0, 0x80000000, 0, 0x80000000,
                0, 0x80000000, 0, 0x80000000, 0, 0x80000000, 0, 0x80000000, 0, 0x80000000))));

    else
        return __substract<_DesiredType_>(__simd_broadcast_zeros<__generation, __register_policy, _VectorType_>(), __vector);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX512F, zmm512>::__add(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (__is_pd_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_add_pd(
            __intrin_bitcast<__m512d>(__left), __intrin_bitcast<__m512d>(__right)));
    }
    else if constexpr (__is_ps_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_add_ps(
            __intrin_bitcast<__m512>(__left), __intrin_bitcast<__m512>(__right)));
    }
    else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_add_epi32(
            __intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right)));
    }
    else if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_add_epi64(
            __intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right)));
    }
    else {
        const auto __low = __simd_add<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right));

        const auto __high = __simd_add<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(
            _mm512_extracti64x4_epi64(__intrin_bitcast<__m512i>(__left), 1),
            _mm512_extracti64x4_epi64(__intrin_bitcast<__m512i>(__right), 1));

        return __intrin_bitcast<_VectorType_>(_mm512_inserti64x4(__intrin_bitcast<__m512i>(__low), __high, 1));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX512F, zmm512>::__substract(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (__is_pd_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_sub_pd(
            __intrin_bitcast<__m512d>(__left), __intrin_bitcast<__m512d>(__right)));
    }
    else if constexpr (__is_ps_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_sub_ps(
            __intrin_bitcast<__m512>(__left), __intrin_bitcast<__m512>(__right)));
    }
    else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_sub_epi32(
            __intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right)));
    }
    else if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_sub_epi64(
            __intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right)));
    }
    else {
        const auto __low = __simd_substract<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right));

        const auto __high = __simd_substract<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(
            _mm512_extracti64x4_epi64(__intrin_bitcast<__m512i>(__left), 1),
            _mm512_extracti64x4_epi64(__intrin_bitcast<__m512i>(__right), 1));

        return __intrin_bitcast<_VectorType_>(_mm512_inserti64x4(__intrin_bitcast<__m512i>(__low), __high, 1));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX512F, zmm512>::__multiply(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    return __left;
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX512F, zmm512>::__divide(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    return __left;
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX512F, zmm512>::__bit_not(_VectorType_ __vector) noexcept {
    if      constexpr (std::is_same_v<_VectorType_, __m512d>)
        return _mm512_xor_pd(__vector, _mm512_set1_pd(-1));

    else if constexpr (std::is_same_v<_VectorType_, __m512i>)
        return _mm512_xor_si512(__vector, _mm512_set1_epi32(-1));

    else if constexpr (std::is_same_v<_VectorType_, __m512>)
        return _mm512_xor_ps(__vector, _mm512_set1_ps(-1));
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX512F, zmm512>::__bit_xor(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if      constexpr (std::is_same_v<_VectorType_, __m512d>)
        return _mm512_xor_pd(__left, __right);

    else if constexpr (std::is_same_v<_VectorType_, __m512i>)
        return _mm512_xor_si512(__left, __right);

    else if constexpr (std::is_same_v<_VectorType_, __m512>)
        return _mm512_xor_ps(__left, __right);
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX512F, zmm512>::__bit_and(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if      constexpr (std::is_same_v<_VectorType_, __m512d>)
        return _mm512_and_pd(__left, __right);

    else if constexpr (std::is_same_v<_VectorType_, __m512i>)
        return _mm512_and_si512(__left, __right);

    else if constexpr (std::is_same_v<_VectorType_, __m512>)
        return _mm512_and_ps(__left, __right);
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX512F, zmm512>::__bit_or(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if      constexpr (std::is_same_v<_VectorType_, __m512d>)
        return _mm512_or_pd(__left, __right);

    else if constexpr (std::is_same_v<_VectorType_, __m512i>)
        return _mm512_or_si512(__left, __right);

    else if constexpr (std::is_same_v<_VectorType_, __m512>)
        return _mm512_or_ps(__left, __right);
}

template <
    typename _DesiredType_,
    typename _VectorType_,
    typename _ReduceBinaryFunction_>
simd_stl_always_inline _DesiredType_ __simd_arithmetic<arch::CpuFeature::AVX512BW, zmm512>::__horizontal_fold(
    _VectorType_            __vector,
    _ReduceBinaryFunction_  __reduce) noexcept
{
    if constexpr (sizeof(_DesiredType_) >= 4) {
        return __simd_horizontal_fold<arch::CpuFeature::AVX512F, __register_policy, _DesiredType_>(__vector, __reduce);
    }
    else if constexpr (__is_epi16_v<_DesiredType_> || __is_epu16_v<_DesiredType_>) {
        const auto __shuffle_words = _mm512_set_epi8(
            61, 60, 63, 62, 57, 56, 59, 58, 53, 52, 55, 54, 49, 48, 51, 50,
            45, 44, 47, 46, 41, 40, 43, 42, 37, 36, 39, 38, 33, 32, 35, 34,
            29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 18,
            13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2);

        auto __horizontal_folded_values = __intrin_bitcast<__m512i>(__vector);

        const auto __shuffled1      = _mm512_permutexvar_epi64(_mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0), __horizontal_folded_values);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffled1);

        const auto __shuffled2      = _mm512_permutex_epi64(__horizontal_folded_values, 0x4E);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffled2);

        const auto __shuffled3      = _mm512_permutex_epi64(__horizontal_folded_values, 0xB1);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffled3);

        const auto __shuffled4      = __intrin_bitcast<__m512i>(_mm512_permute_ps(__intrin_bitcast<__m512>(__horizontal_folded_values), 0xB1));
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffled4);

        const auto __shuffled5      = _mm512_shuffle_epi8(__horizontal_folded_values, __shuffle_words);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __intrin_bitcast<__m512i>(__shuffled5));

        return _mm512_cvtsi512_si32(__horizontal_folded_values);
    }
    else if constexpr (__is_epi8_v<_DesiredType_> || __is_epu8_v<_DesiredType_>) {
        const auto __shuffle_words = _mm512_set_epi8(
            61, 60, 63, 62, 57, 56, 59, 58, 53, 52, 55, 54, 49, 48, 51, 50,
            45, 44, 47, 46, 41, 40, 43, 42, 37, 36, 39, 38, 33, 32, 35, 34,
            29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 18,
            13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2);

        const auto __shuffle_bytes = _mm512_set_epi8(
            62, 63, 60, 61, 58, 59, 56, 57, 54, 55, 52, 53, 50, 51, 48, 49,
            46, 47, 44, 45, 42, 43, 40, 41, 38, 39, 36, 37, 34, 35, 32, 33,
            30, 31, 28, 29, 26, 27, 24, 25, 22, 23, 20, 21, 18, 19, 16, 17,
            14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);

        auto __horizontal_folded_values = __intrin_bitcast<__m512i>(__vector);

        const auto __shuffled1      = _mm512_permutexvar_epi64(_mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0), __horizontal_folded_values);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffled1);

        const auto __shuffled2      = _mm512_permutex_epi64(__horizontal_folded_values, 0x4E);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffled2);

        const auto __shuffled3      = _mm512_permutex_epi64(__horizontal_folded_values, 0xB1);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffled3);

        const auto __shuffled4      = __intrin_bitcast<__m512i>(_mm512_permute_ps(__intrin_bitcast<__m512>(__horizontal_folded_values), 0xB1));
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __shuffled4);

        const auto __shuffled5      = _mm512_shuffle_epi8(__horizontal_folded_values, __shuffle_words);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __intrin_bitcast<__m512i>(__shuffled5));

        const auto __shuffled6      = _mm512_shuffle_epi8(__horizontal_folded_values, __shuffle_bytes);
        __horizontal_folded_values  = __reduce(__horizontal_folded_values, __intrin_bitcast<__m512i>(__shuffled6));

        return _mm512_cvtsi512_si32(__horizontal_folded_values);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX512BW, zmm512>::__vertical_min(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (__is_epi8_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm512_min_epi8(
            __intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right)));
    else if constexpr (__is_epu8_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm512_min_epu8(
            __intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right)));
    else if constexpr (__is_epi16_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm512_min_epi16(
            __intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right)));
    else if constexpr (__is_epu16_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm512_min_epu16(
            __intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right)));
    else
        return __simd_vertical_min<arch::CpuFeature::AVX512F, zmm512, _DesiredType_>(__left, __right);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_ __simd_arithmetic<arch::CpuFeature::AVX512BW, zmm512>::__horizontal_min(_VectorType_ __vector) noexcept {
    return __horizontal_fold<_DesiredType_>(__vector, __vertical_min_wrapper<__generation, __register_policy, _DesiredType_>{});
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX512BW, zmm512>::__vertical_max(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (__is_epi8_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm512_max_epi8(
            __intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right)));
    else if constexpr (__is_epu8_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm512_max_epu8(
            __intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right)));
    else if constexpr (__is_epi16_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm512_max_epi16(
            __intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right)));
    else if constexpr (__is_epu16_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm512_max_epu16(
            __intrin_bitcast<__m512i>(__left), __intrin_bitcast<__m512i>(__right)));
    else
        return __simd_vertical_max<arch::CpuFeature::AVX512F, zmm512, _DesiredType_>(__left, __right);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_ __simd_arithmetic<arch::CpuFeature::AVX512BW, zmm512>::__horizontal_max(_VectorType_ __vector) noexcept {
    return __horizontal_fold<_DesiredType_>(__vector, __vertical_max_wrapper<__generation, __register_policy, _DesiredType_>{});
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline auto __simd_arithmetic<arch::CpuFeature::AVX512BW, zmm512>::__reduce(_VectorType_ __vector) noexcept {
    if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>) {
        return _mm512_reduce_add_epi64(__intrin_bitcast<__m512i>(__vector));
    }
    else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>) {
        return _mm512_reduce_add_epi32(__intrin_bitcast<__m512i>(__vector));
    }
    else if constexpr (__is_ps_v<_DesiredType_>) {
        return _mm512_reduce_add_ps(__intrin_bitcast<__m512>(__vector));
    }
    else if constexpr (__is_pd_v<_DesiredType_>) {
        return _mm512_reduce_add_pd(__intrin_bitcast<__m512d>(__vector));
    }
    else if constexpr (__is_epi8_v<_DesiredType_> || __is_epu8_v<_DesiredType_>) {
        return _mm512_reduce_add_epi64(_mm512_sad_epu8(__intrin_bitcast<__m512i>(__vector), _mm512_setzero_si512()));
    }
    else if constexpr (__is_epi16_v<_DesiredType_> || __is_epu16_v<_DesiredType_>) {
        return __simd_reduce<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(__intrin_bitcast<__m256i>(__vector)) +
            __simd_reduce<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(_mm512_extractf64x4_pd(__intrin_bitcast<__m512d>(__vector), 1));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX512BW, zmm512>::__abs(_VectorType_ __vector) noexcept {
    if constexpr (__is_epi16_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_abs_epi16(__intrin_bitcast<__m512i>(__vector)));
    }
    else if constexpr (__is_epi8_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm512_abs_epi8(__intrin_bitcast<__m512i>(__vector)));
    }
    else {
        return __simd_abs<arch::CpuFeature::AVX512F, zmm512, _DesiredType_>(__vector);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX512VLF, ymm256>::__vertical_min(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (__is_epi64_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_min_epi64(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));
    }
    else if constexpr (__is_epu64_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_min_epu64(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));
    }
    else {
        return __simd_vertical_min<arch::CpuFeature::AVX2, __register_policy, _DesiredType_, _VectorType_>(__left, __right);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX512VLF, ymm256>::__vertical_max(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (__is_epi64_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_max_epi64(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));
    }
    else if constexpr (__is_epu64_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm256_max_epu64(
            __intrin_bitcast<__m256i>(__left), __intrin_bitcast<__m256i>(__right)));
    }
    else {
        return __simd_vertical_max<arch::CpuFeature::AVX2, __register_policy, _DesiredType_, _VectorType_>(__left, __right);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX512VLF, ymm256>::__abs(_VectorType_ __vector) noexcept {
    if constexpr (__is_epi64_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_abs_epi64(__intrin_bitcast<__m256i>(__vector)));
    else
        return __simd_abs<arch::CpuFeature::AVX2, __register_policy, _DesiredType_>(__vector);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX512VLF, xmm128>::__vertical_min(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (__is_epi64_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_min_epi64(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else if constexpr (__is_epu64_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_min_epu64(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else {
        return __simd_vertical_min<arch::CpuFeature::SSE42, __register_policy, _DesiredType_, _VectorType_>(__left, __right);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX512VLF, xmm128>::__vertical_max(
    _VectorType_ __left,
    _VectorType_ __right) noexcept
{
    if constexpr (__is_epi64_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_max_epi64(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else if constexpr (__is_epu64_v<_DesiredType_>) {
        return __intrin_bitcast<_VectorType_>(_mm_max_epu64(
            __intrin_bitcast<__m128i>(__left), __intrin_bitcast<__m128i>(__right)));
    }
    else {
        return __simd_vertical_max<arch::CpuFeature::SSE42, __register_policy, _DesiredType_, _VectorType_>(__left, __right);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_arithmetic<arch::CpuFeature::AVX512VLF, xmm128>::__abs(_VectorType_ __vector) noexcept {
    if constexpr (__is_epi64_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm_abs_epi64(__intrin_bitcast<__m128i>(__vector)));
    else
        return __simd_abs<arch::CpuFeature::SSE42, __register_policy, _DesiredType_>(__vector);
}


#pragma endregion

__SIMD_STL_NUMERIC_NAMESPACE_END
