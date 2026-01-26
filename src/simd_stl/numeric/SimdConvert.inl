#pragma once 


__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

#pragma region Sse2-Sse4.2 Simd convert

template <
    typename    _DesiredType_,
    typename    _VectorType_>
simd_stl_always_inline auto __simd_convert<arch::CpuFeature::SSE2, xmm128>::__to_mask(_VectorType_ __vector) noexcept
{
    using _SimdMaskType = simd_mask<__generation, _DesiredType_, __register_policy>;

    if      constexpr (sizeof(_DesiredType_) == 8)
        return static_cast<typename _SimdMaskType::mask_type>(_mm_movemask_pd(__intrin_bitcast<__m128d>(__vector)));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return static_cast<typename _SimdMaskType::mask_type>(_mm_movemask_ps(__intrin_bitcast<__m128>(__vector)));

    else if constexpr (sizeof(_DesiredType_) == 2)
        return static_cast<typename _SimdMaskType::mask_type>(_mm_movemask_epi8(_mm_packs_epi16(__intrin_bitcast<__m128i>(__vector), _mm_setzero_si128())));

    else if constexpr (sizeof(_DesiredType_) == 1)
        return static_cast<typename _SimdMaskType::mask_type>(_mm_movemask_epi8(__intrin_bitcast<__m128i>(__vector)));
}

template <
    typename    _DesiredType_,
    typename    _VectorType_>
simd_stl_always_inline auto __simd_convert<arch::CpuFeature::SSE2, xmm128>::__to_index_mask(_VectorType_ __vector) noexcept {
    return _mm_movemask_epi8(__intrin_bitcast<__m128i>(__vector));
}

template <
    typename    _VectorType_,
    typename    _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_convert<arch::CpuFeature::SSE2, xmm128>::__to_vector(__simd_mask_type<_DesiredType_> __mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        const auto __first = (__mask >> 1) & 1;
        const auto __second = __mask & 1;

        const auto __broadcasted = _mm_set_epi32(__first, __first, __second, __second);
        return __intrin_bitcast<_VectorType_>(_mm_cmpeq_epi32(__broadcasted, _mm_set1_epi32(1)));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        const auto __broadcasted = _mm_set_epi32((__mask >> 3) & 1, (__mask >> 2) & 1, (__mask >> 1) & 1, __mask & 1);
        return __intrin_bitcast<_VectorType_>(_mm_cmpeq_epi32(__broadcasted, _mm_set1_epi32(1)));
    }
    else if constexpr (sizeof(_DesiredType_) == 2) {
        const auto __broadcasted = _mm_set_epi16((__mask >> 7) & 1, (__mask >> 6) & 1, (__mask >> 5) & 1,
            (__mask >> 4) & 1, (__mask >> 3) & 1, (__mask >> 2) & 1, (__mask >> 1) & 1, __mask & 1);

        return __intrin_bitcast<_VectorType_>(_mm_cmpeq_epi16(__broadcasted, _mm_set1_epi16(1)));
    }
    else if constexpr (sizeof(_DesiredType_) == 1) {
        const auto __broadcasted = _mm_set_epi8((__mask >> 15) & 1, (__mask >> 14) & 1, (__mask >> 13) & 1, (__mask >> 12) & 1,
            (__mask >> 11) & 1, (__mask >> 10) & 1, (__mask >> 9) & 1, (__mask >> 8) & 1, (__mask >> 7) & 1,
            (__mask >> 6) & 1, (__mask >> 5) & 1, (__mask >> 4) & 1, (__mask >> 3) & 1, (__mask >> 2) & 1, (__mask >> 1) & 1, __mask & 1);

        return __intrin_bitcast<_VectorType_>(_mm_cmpeq_epi8(__broadcasted, _mm_set1_epi8(1)));
    }
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_convert<arch::CpuFeature::SSSE3, xmm128>::__to_vector(__simd_mask_type<_DesiredType_> __mask) noexcept 
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        const auto __first = (__mask >> 1) & 1;
        const auto __second = __mask & 1;

        const auto __broadcasted = _mm_set_epi32(__first, __first, __second, __second);
        return __intrin_bitcast<_VectorType_>(_mm_cmpeq_epi32(__broadcasted, _mm_set1_epi32(1)));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        const auto __broadcasted = _mm_set_epi32((__mask >> 3) & 1, (__mask >> 2) & 1, (__mask >> 1) & 1, __mask & 1);
        return __intrin_bitcast<_VectorType_>(_mm_cmpeq_epi32(__broadcasted, _mm_set1_epi32(1)));
    }
    else if constexpr (sizeof(_DesiredType_) == 2) {
        const auto __broadcasted = _mm_set_epi16((__mask >> 7) & 1, (__mask >> 6) & 1, (__mask >> 5) & 1, (__mask >> 4) & 1,
            (__mask >> 3) & 1, (__mask >> 2) & 1, (__mask >> 1) & 1, __mask & 1);

        return __intrin_bitcast<_VectorType_>(_mm_cmpeq_epi16(__broadcasted, _mm_set1_epi16(1)));
    }
    else if constexpr (sizeof(_DesiredType_) == 1) {
        const auto __select = _mm_set1_epi64x(0x8040201008040201ull);
        const auto __shuffled = _mm_shuffle_epi8(_mm_cvtsi32_si128(__mask), _mm_set_epi64x(0x0101010101010101ll, 0));

        return __intrin_bitcast<_VectorType_>(_mm_cmpeq_epi8(_mm_and_si128(__shuffled, __select), __select));
    }
}

#pragma endregion

#pragma region Avx-Avx2 Simd convert

template <
    typename    _DesiredType_,
    typename    _VectorType_>
simd_stl_always_inline auto __simd_convert<arch::CpuFeature::AVX2, ymm256>::__to_mask(_VectorType_ __vector) noexcept
{
    using _SimdMaskType = simd_mask<__generation, _DesiredType_, __register_policy>;

    if      constexpr (sizeof(_DesiredType_) == 8) {
        return static_cast<typename _SimdMaskType::mask_type>(_mm256_movemask_pd(__intrin_bitcast<__m256d>(__vector)));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        return static_cast<typename _SimdMaskType::mask_type>(_mm256_movemask_ps(__intrin_bitcast<__m256>(__vector)));
    }
    else if constexpr (sizeof(_DesiredType_) == 2) {
        const auto __pack = _mm256_packs_epi16(__intrin_bitcast<__m256i>(__vector), _mm256_setzero_si256());
        const auto __shuffled = _mm256_permute4x64_epi64(__pack, 0xD8);

        return static_cast<typename _SimdMaskType::mask_type>(_mm256_movemask_epi8(__shuffled));
    }
    else if constexpr (sizeof(_DesiredType_) == 1) {
        return static_cast<typename _SimdMaskType::mask_type>(_mm256_movemask_epi8(__intrin_bitcast<__m256i>(__vector)));
    }
}

template <
    typename    _DesiredType_,
    typename    _VectorType_>
simd_stl_always_inline auto __simd_convert<arch::CpuFeature::AVX2, ymm256>::__to_index_mask(_VectorType_ __vector) noexcept {
    return _mm256_movemask_epi8(__intrin_bitcast<__m256i>(__vector));
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_convert<arch::CpuFeature::AVX2, ymm256>::__to_vector(__simd_mask_type<_DesiredType_> __mask) noexcept 
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        const auto __broadcasted = _mm256_set_pd((__mask >> 3) & 1, (__mask >> 2) & 1, (__mask >> 1) & 1, __mask & 1);
        return __intrin_bitcast<_VectorType_>(_mm256_cmp_pd(__broadcasted, _mm256_set1_pd(1), 0));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        const auto __broadcasted = _mm256_set_ps((__mask >> 7) & 1, (__mask >> 6) & 1, (__mask >> 5) & 1, (__mask >> 4) & 1,
            (__mask >> 3) & 1, (__mask >> 2) & 1, (__mask >> 1) & 1, __mask & 1);
        return __intrin_bitcast<_VectorType_>(_mm256_cmp_ps(__broadcasted, _mm256_set1_ps(1), 0));
    }
    else if constexpr (sizeof(_DesiredType_) == 2) {
        const auto __broadcasted = _mm256_set_epi16((__mask >> 15) & 1, (__mask >> 14) & 1, (__mask >> 13) & 1, (__mask >> 12) & 1,
            (__mask >> 11) & 1, (__mask >> 10) & 1, (__mask >> 9) & 1, (__mask >> 8) & 1, (__mask >> 7) & 1, (__mask >> 6) & 1,
            (__mask >> 5) & 1, (__mask >> 4) & 1, (__mask >> 3) & 1, (__mask >> 2) & 1, (__mask >> 1) & 1, __mask & 1);

        return __intrin_bitcast<_VectorType_>(_mm256_cmpeq_epi16(__broadcasted, _mm256_set1_epi16(1)));
    }
    else if constexpr (sizeof(_DesiredType_) == 1) {
        const auto __vector_mask = _mm256_setr_epi32(__mask & 0xFFFF, 0, 0, 0, (__mask >> 16) & 0xFFFF, 0, 0, 0);

        const auto __select = _mm256_set1_epi64x(0x8040201008040201ull);
        const auto __shuffled = _mm256_shuffle_epi8(__vector_mask, _mm256_set_epi64x(0x0101010101010101ll, 0, 0x0101010101010101ll, 0));

        return __intrin_bitcast<_VectorType_>(_mm256_cmpeq_epi8(_mm256_and_si256(__shuffled, __select), __select));
    }
}
#pragma endregion

#pragma region Avx512 Simd convert


template <
    typename    _DesiredType_,
    typename    _VectorType_>
simd_stl_always_inline auto __simd_convert<arch::CpuFeature::AVX512F, zmm512>::__to_mask(_VectorType_ __vector) noexcept
{
    using _SimdMaskType = simd_mask<__generation, _DesiredType_, __register_policy>;

    if constexpr (sizeof(_DesiredType_) == 8) {
        return static_cast<typename _SimdMaskType::mask_type>(_mm512_cmp_epi64_mask(__intrin_bitcast<__m512i>(__vector), _mm512_setzero_si512(), _MM_CMPINT_LT));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        return static_cast<typename _SimdMaskType::mask_type>(_mm512_cmp_epi32_mask(__intrin_bitcast<__m512i>(__vector), _mm512_setzero_si512(), _MM_CMPINT_LT));
    }
    else {
        constexpr auto __ymm_bits = (sizeof(_VectorType_) / sizeof(_DesiredType_)) >> 1;

        const auto __low = __simd_to_mask<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(__intrin_bitcast<__m256d>(__vector));
        const auto __high = __simd_to_mask<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(_mm512_extractf64x4_pd(__intrin_bitcast<__m512d>(__vector), 1));

        return ((static_cast<typename _SimdMaskType::mask_type>(__high) << __ymm_bits) | static_cast<typename _SimdMaskType::mask_type>(__low));
    }
}


template <
    typename    _DesiredType_,
    typename    _VectorType_>
simd_stl_always_inline auto __simd_convert<arch::CpuFeature::AVX512F, zmm512>::__to_index_mask(_VectorType_ __vector) noexcept {
    if constexpr (sizeof(_DesiredType_) == 8) {
        return _mm512_cmp_epi64_mask(__intrin_bitcast<__m512i>(__vector), _mm512_setzero_si512(), _MM_CMPINT_LT);
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        return _mm512_cmp_epi32_mask(__intrin_bitcast<__m512i>(__vector), _mm512_setzero_si512(), _MM_CMPINT_LT);
    }
    else {
        constexpr auto __ymm_bits = (sizeof(_VectorType_) / sizeof(_DesiredType_)) >> 1;

        const auto __low = __simd_to_index_mask<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(__intrin_bitcast<__m256d>(__vector));
        const auto __high = __simd_to_index_mask<arch::CpuFeature::AVX2, ymm256, _DesiredType_>(_mm512_extractf64x4_pd(__intrin_bitcast<__m512d>(__vector), 1));

        return (static_cast<uint32>(__high) << __ymm_bits) | static_cast<uint32>(__low);
    }
}

template <
    typename _VectorType_,
    typename _DesiredType_>
 simd_stl_always_inline _VectorType_ __simd_convert<arch::CpuFeature::AVX512F, zmm512>
    ::__to_vector(__simd_mask_type<_DesiredType_> __mask) noexcept 
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        return __intrin_bitcast<_VectorType_>(_mm512_maskz_mov_epi64(__mask, _mm512_set1_epi64(-1)));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        return __intrin_bitcast<_VectorType_>(_mm512_maskz_mov_epi32(__mask, _mm512_set1_epi32(-1)));
    }
    else {
        using _HalfType = IntegerForSize<_Max<(sizeof(__simd_mask_type<_DesiredType_>) >> 1), 1>()>::Unsigned;

        constexpr auto __maximum    = math::__maximum_integral_limit<_HalfType>();
        constexpr auto __shift      = (sizeof(__simd_mask_type<_DesiredType_>) << 2);

        const auto __low = __simd_to_vector<arch::CpuFeature::AVX2, ymm256, __m256i, _DesiredType_>(__mask & __maximum);
        const auto __high = __simd_to_vector<arch::CpuFeature::AVX2, ymm256, __m256i, _DesiredType_>((__mask >> __shift));

        return __intrin_bitcast<_VectorType_>(_mm512_inserti64x4(__intrin_bitcast<__m512i>(__low), __high, 1));
    }
}

template <
    typename    _DesiredType_,
    typename    _VectorType_>
simd_stl_always_inline auto __simd_convert<arch::CpuFeature::AVX512BW, zmm512>::__to_mask(_VectorType_ __vector) noexcept
{
    using _SimdMaskType = simd_mask<__generation, _DesiredType_, __register_policy>;

    if constexpr (sizeof(_DesiredType_) == 2)
        return static_cast<typename _SimdMaskType::mask_type>(_mm512_movepi16_mask(__intrin_bitcast<__m512i>(__vector)));

    else if constexpr (sizeof(_DesiredType_) == 1)
        return static_cast<typename _SimdMaskType::mask_type>(_mm512_movepi8_mask(__intrin_bitcast<__m512i>(__vector)));

    else
        return __simd_to_mask<arch::CpuFeature::AVX512F, zmm512, _DesiredType_>(__vector);
}

template <
    typename    _DesiredType_,
    typename    _VectorType_>
simd_stl_always_inline auto __simd_convert<arch::CpuFeature::AVX512BW, zmm512>::__to_index_mask(_VectorType_ __vector) noexcept {
    return _mm512_movepi8_mask(__intrin_bitcast<__m512i>(__vector));
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_convert<arch::CpuFeature::AVX512BW, zmm512>::__to_vector(__simd_mask_type<_DesiredType_> __mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 1)
        return __intrin_bitcast<_VectorType_>(_mm512_movm_epi8(__mask));

    else if constexpr (sizeof(_DesiredType_) == 2)
        return __intrin_bitcast<_VectorType_>(_mm512_movm_epi16(__mask));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm512_maskz_mov_epi32(__mask, _mm512_set1_epi32(-1)));

    else if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm512_maskz_mov_epi64(__mask, _mm512_set1_epi64(-1)));
}

template <
     typename    _DesiredType_,
     typename    _VectorType_>
simd_stl_always_inline auto __simd_convert<arch::CpuFeature::AVX512BWDQ, zmm512>::__to_mask(_VectorType_ __vector) noexcept
{
    using _SimdMaskType = simd_mask<__generation, _DesiredType_, __register_policy>;

    if constexpr (sizeof(_DesiredType_) == 1)
        return static_cast<typename _SimdMaskType::mask_type>(_mm512_movepi8_mask(__intrin_bitcast<__m512i>(__vector)));

    else if constexpr (sizeof(_DesiredType_) == 2)
        return static_cast<typename _SimdMaskType::mask_type>(_mm512_movepi16_mask(__intrin_bitcast<__m512i>(__vector)));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return static_cast<typename _SimdMaskType::mask_type>(_mm512_movepi32_mask(__intrin_bitcast<__m512i>(__vector)));

    else if constexpr (sizeof(_DesiredType_) == 8)
        return static_cast<typename _SimdMaskType::mask_type>(_mm512_movepi64_mask(__intrin_bitcast<__m512i>(__vector)));
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_convert<arch::CpuFeature::AVX512BWDQ, zmm512>::__to_vector(__simd_mask_type<_DesiredType_> __mask) noexcept 
{
    if constexpr (sizeof(_DesiredType_) == 1)
        return __intrin_bitcast<_VectorType_>(_mm512_movm_epi8(__mask));

    else if constexpr (sizeof(_DesiredType_) == 2)
        return __intrin_bitcast<_VectorType_>(_mm512_movm_epi16(__mask));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm512_movm_epi32(__mask));

    else if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm512_movm_epi64(__mask));
}

template <
    typename    _DesiredType_,
    typename    _VectorType_>
simd_stl_always_inline auto __simd_convert<arch::CpuFeature::AVX512DQ, zmm512>::__to_mask(_VectorType_ __vector) noexcept
{
    using _SimdMaskType = simd_mask<__generation, _DesiredType_, __register_policy>;

    if constexpr (sizeof(_DesiredType_) == 4)
        return static_cast<typename _SimdMaskType::mask_type>(_mm512_movepi32_mask(__intrin_bitcast<__m512i>(__vector)));

    else if constexpr (sizeof(_DesiredType_) == 8)
        return static_cast<typename _SimdMaskType::mask_type>(_mm512_movepi64_mask(__intrin_bitcast<__m512i>(__vector)));

    else
        return __simd_to_mask<arch::CpuFeature::AVX512F, __register_policy, _DesiredType_>(__vector);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_convert<arch::CpuFeature::AVX512DQ, zmm512>::__to_vector(__simd_mask_type<_DesiredType_> __mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm512_movm_epi32(__mask));

    else if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm512_movm_epi64(__mask));

    else
        return __simd_to_vector<arch::CpuFeature::AVX512F, __register_policy, _VectorType_, _DesiredType_>(__mask);
}


template <
    typename    _DesiredType_,
    typename    _VectorType_>
simd_stl_always_inline auto __simd_convert<arch::CpuFeature::AVX512VLBW, ymm256>::__to_mask(_VectorType_ __vector) noexcept {
    using _SimdMaskType = simd_mask<__generation, _DesiredType_, __register_policy>;

    if constexpr (sizeof(_DesiredType_) == 1)
        return static_cast<typename _SimdMaskType::mask_type>(_mm256_movepi8_mask(__intrin_bitcast<__m256i>(__vector)));

    else if constexpr (sizeof(_DesiredType_) == 2)
        return static_cast<typename _SimdMaskType::mask_type>(_mm256_movepi16_mask(__intrin_bitcast<__m256i>(__vector)));

    else
        return __simd_to_mask<arch::CpuFeature::AVX2, __register_policy, _DesiredType_>(__vector);
}

template <
    typename    _DesiredType_,
    typename    _VectorType_>
simd_stl_always_inline auto __simd_convert<arch::CpuFeature::AVX512VLBW, ymm256>::__to_index_mask(_VectorType_ __vector) noexcept {
    return _mm256_movepi8_mask(__intrin_bitcast<__m256i>(__vector));
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_convert<arch::CpuFeature::AVX512VLBW, ymm256>::__to_vector(__simd_mask_type<_DesiredType_> __mask) noexcept 
{
    if constexpr (sizeof(_DesiredType_) == 1)
        return __intrin_bitcast<_VectorType_>(_mm256_movm_epi8(__mask));

    else if constexpr (sizeof(_DesiredType_) == 2)
        return __intrin_bitcast<_VectorType_>(_mm256_movm_epi16(__mask));

    else
        return __simd_to_vector<arch::CpuFeature::AVX2, __register_policy, _DesiredType_>(__mask);
}

template <
    typename    _DesiredType_,
    typename    _VectorType_>
simd_stl_always_inline auto __simd_convert<arch::CpuFeature::AVX512VLDQ, ymm256>::__to_mask(_VectorType_ __vector) noexcept
{
    using _SimdMaskType = simd_mask<__generation, _DesiredType_, __register_policy>;

    if constexpr (sizeof(_DesiredType_) == 4)
        return static_cast<typename _SimdMaskType::mask_type>(_mm256_movepi32_mask(__intrin_bitcast<__m256i>(__vector)));

    else if constexpr (sizeof(_DesiredType_) == 8)
        return static_cast<typename _SimdMaskType::mask_type>(_mm256_movepi64_mask(__intrin_bitcast<__m256i>(__vector)));

    else
        return __simd_to_mask<arch::CpuFeature::AVX2, __register_policy, _DesiredType_>(__vector);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_convert<arch::CpuFeature::AVX512VLDQ, ymm256>::__to_vector(__simd_mask_type<_DesiredType_> __mask) noexcept 
{
    if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm256_movm_epi32(__mask));

    else if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm256_movm_epi64(__mask));

    else
        return __simd_to_vector<arch::CpuFeature::AVX2, __register_policy, _DesiredType_>(__mask);
}

template <
    typename    _DesiredType_,
    typename    _VectorType_>
simd_stl_always_inline auto __simd_convert<arch::CpuFeature::AVX512VLBW, xmm128>::__to_mask(_VectorType_ __vector) noexcept
{
    using _SimdMaskType = simd_mask<__generation, _DesiredType_, __register_policy>;

    if constexpr (sizeof(_DesiredType_) == 1)
        return static_cast<typename _SimdMaskType::mask_type>(_mm_movepi8_mask(__intrin_bitcast<__m128i>(__vector)));

    else if constexpr (sizeof(_DesiredType_) == 8)
        return static_cast<typename _SimdMaskType::mask_type>(_mm_movepi16_mask(__intrin_bitcast<__m128i>(__vector)));

    else
        return __simd_to_mask<arch::CpuFeature::SSE42, __register_policy, _DesiredType_>(__vector);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_convert<arch::CpuFeature::AVX512VLBW, xmm128>::__to_vector(__simd_mask_type<_DesiredType_> __mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 1)
        return __intrin_bitcast<_VectorType_>(_mm_movm_epi8(__mask));

    else if constexpr (sizeof(_DesiredType_) == 2)
        return __intrin_bitcast<_VectorType_>(_mm_movm_epi16(__mask));

    else
        return __simd_to_vector<arch::CpuFeature::SSE42, __register_policy, _DesiredType_>(__mask);
}

template <
    typename    _DesiredType_,
    typename    _VectorType_>
simd_stl_always_inline auto __simd_convert<arch::CpuFeature::AVX512VLDQ, xmm128>::__to_mask(_VectorType_ __vector) noexcept
{
    using _SimdMaskType = simd_mask<__generation, _DesiredType_, __register_policy>;

    if constexpr (sizeof(_DesiredType_) == 4)
        return static_cast<typename _SimdMaskType::mask_type>(_mm_movepi32_mask(__intrin_bitcast<__m128i>(__vector)));

    else if constexpr (sizeof(_DesiredType_) == 8)
        return static_cast<typename _SimdMaskType::mask_type>(_mm_movepi64_mask(__intrin_bitcast<__m128i>(__vector)));

    else
        return __simd_to_mask<arch::CpuFeature::SSE42, __register_policy, _DesiredType_>(__vector);
}

template <
    typename    _DesiredType_,
    typename    _VectorType_>
simd_stl_always_inline auto __simd_convert<arch::CpuFeature::AVX512VLBW, xmm128>::__to_index_mask(_VectorType_ __vector) noexcept {
    return _mm_movepi8_mask(__intrin_bitcast<__m128i>(__vector));
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_convert<arch::CpuFeature::AVX512VLDQ, xmm128>::__to_vector(__simd_mask_type<_DesiredType_> __mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm_movm_epi32(__mask));

    else if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm_movm_epi64(__mask));

    else
        return __simd_to_vector<arch::CpuFeature::SSE42, __register_policy, _DesiredType_>(__mask);
}

template <
    typename    _DesiredType_,
    typename    _VectorType_>
simd_stl_always_inline auto __simd_convert<arch::CpuFeature::AVX512VLBWDQ, xmm128>::__to_mask(_VectorType_ __vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) <= 2)
        return __simd_to_mask<arch::CpuFeature::AVX512VLBW, __register_policy, _DesiredType_>(__vector);
    else
        return __simd_to_mask<arch::CpuFeature::AVX512VLDQ, __register_policy, _DesiredType_>(__vector);
}

template <
    typename    _DesiredType_,
    typename    _VectorType_>
simd_stl_always_inline auto __simd_convert<arch::CpuFeature::AVX512VLBWDQ, xmm128>::__to_index_mask(_VectorType_ __vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) <= 2)
        return __simd_to_index_mask<arch::CpuFeature::AVX512VLBW, __register_policy, _DesiredType_>(__vector);
    else
        return __simd_to_index_mask<arch::CpuFeature::AVX512VLDQ, __register_policy, _DesiredType_>(__vector);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_convert<arch::CpuFeature::AVX512VLBWDQ, xmm128>::__to_vector(__simd_mask_type<_DesiredType_> __mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) <= 2)
        return __simd_to_vector<arch::CpuFeature::AVX512VLBW, __register_policy, _DesiredType_>(__mask);
    else
        return __simd_to_vector<arch::CpuFeature::AVX512VLDQ, __register_policy, _DesiredType_>(__mask);
}

template <
    typename    _DesiredType_,
    typename    _VectorType_>
simd_stl_always_inline auto __simd_convert<arch::CpuFeature::AVX512VLBWDQ, ymm256>::__to_mask(_VectorType_ __vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) <= 2)
        return __simd_to_mask<arch::CpuFeature::AVX512VLBW, __register_policy, _DesiredType_>(__vector);
    else
        return __simd_to_mask<arch::CpuFeature::AVX512VLDQ, __register_policy, _DesiredType_>(__vector);
}

template <
    typename    _DesiredType_,
    typename    _VectorType_>
simd_stl_always_inline auto __simd_convert<arch::CpuFeature::AVX512VLBWDQ, ymm256>::__to_index_mask(_VectorType_ __vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) <= 2)
        return __simd_to_index_mask<arch::CpuFeature::AVX512VLBW, __register_policy, _DesiredType_>(__vector);
    else
        return __simd_to_index_mask<arch::CpuFeature::AVX512VLDQ, __register_policy, _DesiredType_>(__vector);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_convert<arch::CpuFeature::AVX512VLBWDQ, ymm256>::__to_vector(__simd_mask_type<_DesiredType_> __mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) <= 2)
        return __simd_to_vector<arch::CpuFeature::AVX512VLBW, __register_policy, _DesiredType_>(__mask);
    else
        return __simd_to_vector<arch::CpuFeature::AVX512VLDQ, __register_policy, _DesiredType_>(__mask);
}

#pragma endregion

__SIMD_STL_NUMERIC_NAMESPACE_END
