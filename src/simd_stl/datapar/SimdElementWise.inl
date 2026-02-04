#pragma once 

__SIMD_STL_DATAPAR_NAMESPACE_BEGIN

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
        const auto __mask_segment_lower = __mask & 0xFF;
        const auto __mask_segment_higher = (__mask >> 8) & 0xFF;

        constexpr auto __vector_element_capacity = sizeof(_VectorType_) / sizeof(_DesiredType_);
        alignas(sizeof(_VectorType_)) _DesiredType_ __temporary_stack_buffer[__vector_element_capacity];
        
        _DesiredType_* __destination_write_pointer = __temporary_stack_buffer;

        const auto __processed_byte_count_lower_segment = __tables_sse<sizeof(_DesiredType_)>.__size[__mask_segment_lower];
        const auto __processed_byte_count_higher_segment = __tables_sse<sizeof(_DesiredType_)>.__size[__mask_segment_higher];

        const auto __total_processed_byte_count_combined = __processed_byte_count_lower_segment + __processed_byte_count_higher_segment;
        const auto __unprocessed_tail_blending_mask = (__simd_mask_type<_DesiredType_>(1u << (sizeof(_VectorType_) - __total_processed_byte_count_combined)) - 1) << __total_processed_byte_count_combined;

        const auto __shuffle_control_mask_lower_segment = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(__tables_sse<sizeof(_DesiredType_)>.__shuffle[__mask_segment_lower]));
        const auto __shuffle_control_mask_higher_segment = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(__tables_sse<sizeof(_DesiredType_)>.__shuffle[__mask_segment_higher]));

        const auto __vector_upper_half_prepared_for_shuffling = __intrin_bitcast<__m128i>(_mm_movehl_ps(
            __intrin_bitcast<__m128>(_mm_slli_si128(__intrin_bitcast<__m128i>(__vector), 8)), 
            __intrin_bitcast<__m128>(__vector)));

        const auto __packed_data_lower_segment = _mm_shuffle_epi8(__intrin_bitcast<__m128i>(__vector), __shuffle_control_mask_lower_segment);
        const auto __packed_data_higher_segment = _mm_shuffle_epi8(__vector_upper_half_prepared_for_shuffling, __intrin_bitcast<__m128i>(__shuffle_control_mask_higher_segment));

        _mm_storel_epi64(reinterpret_cast<__m128i*>(__destination_write_pointer), __intrin_bitcast<__m128i>(__packed_data_lower_segment));
        algorithm::__advance_bytes(__destination_write_pointer, __processed_byte_count_lower_segment);

        _mm_storel_epi64(reinterpret_cast<__m128i*>(__destination_write_pointer), __intrin_bitcast<__m128i>(__packed_data_higher_segment));

        const auto __final_packed_vector            = _mm_load_si128(reinterpret_cast<const __m128i*>(__temporary_stack_buffer));
        const auto __final_blended_result_vector    = __blend<_DesiredType_>(__intrin_bitcast<__m128i>(__vector), __final_packed_vector, __unprocessed_tail_blending_mask);

        return { __total_processed_byte_count_combined, __intrin_bitcast<_VectorType_>(__final_blended_result_vector) };
    }
    else {
        const auto __shuffle_mask = _mm_load_si128(reinterpret_cast<const __m128i*>(__tables_sse<sizeof(_DesiredType_)>.__shuffle[__mask]));
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
        const auto __mask_segment_lower = __mask & 0xFF;
        const auto __mask_segment_higher = (__mask >> 8) & 0xFF;

        constexpr auto __vector_element_capacity = sizeof(_VectorType_) / sizeof(_DesiredType_);
        alignas(sizeof(_VectorType_)) _DesiredType_ __temporary_stack_buffer[__vector_element_capacity];

        _DesiredType_* __destination_write_pointer = __temporary_stack_buffer;

        const auto __processed_byte_count_lower_segment     = __tables_sse<sizeof(_DesiredType_)>.__size[__mask_segment_lower];
        const auto __processed_byte_count_higher_segment    = __tables_sse<sizeof(_DesiredType_)>.__size[__mask_segment_higher];

        const auto __total_processed_byte_count_combined    = __processed_byte_count_lower_segment + __processed_byte_count_higher_segment;
        const auto __unprocessed_tail_blending_mask         = (__simd_mask_type<_DesiredType_>(1u << (sizeof(_VectorType_) - __total_processed_byte_count_combined)) - 1) << __total_processed_byte_count_combined;

        const auto __shuffle_control_mask_lower_segment     = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(__tables_sse<sizeof(_DesiredType_)>.__shuffle[__mask_segment_lower]));
        const auto __shuffle_control_mask_higher_segment    = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(__tables_sse<sizeof(_DesiredType_)>.__shuffle[__mask_segment_higher]));

        const auto __vector_upper_half_prepared_for_shuffling = __intrin_bitcast<__m128i>(_mm_movehl_ps(
            __intrin_bitcast<__m128>(_mm_slli_si128(__intrin_bitcast<__m128i>(__vector), 8)),
            __intrin_bitcast<__m128>(__vector)));

        const auto __packed_data_lower_segment  = _mm_shuffle_epi8(__intrin_bitcast<__m128i>(__vector), __shuffle_control_mask_lower_segment);
        const auto __packed_data_higher_segment = _mm_shuffle_epi8(__vector_upper_half_prepared_for_shuffling, __intrin_bitcast<__m128i>(__shuffle_control_mask_higher_segment));

        _mm_storel_epi64(reinterpret_cast<__m128i*>(__destination_write_pointer), __intrin_bitcast<__m128i>(__packed_data_lower_segment));
        algorithm::__advance_bytes(__destination_write_pointer, __processed_byte_count_lower_segment);

        _mm_storel_epi64(reinterpret_cast<__m128i*>(__destination_write_pointer), __intrin_bitcast<__m128i>(__packed_data_higher_segment));
        
        const auto __final_packed_vector = _mm_load_si128(reinterpret_cast<const __m128i*>(__temporary_stack_buffer));
        const auto __final_blended_result_vector = __blend<_DesiredType_>(__intrin_bitcast<__m128i>(__vector), __final_packed_vector, __unprocessed_tail_blending_mask);

        return { __total_processed_byte_count_combined, __intrin_bitcast<_VectorType_>(__final_blended_result_vector) };
    }
    else {
        const auto __shuffle_mask = _mm_load_si128(reinterpret_cast<const __m128i*>(__tables_sse<sizeof(_DesiredType_)>.__shuffle[__mask]));
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
    return __compress<_DesiredType_>(__vector, __simd_to_mask<__generation, __register_policy, _DesiredType_>(__mask));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline std::pair<int32, _VectorType_> __simd_element_wise<arch::CpuFeature::AVX2, ymm256>::__compress(
    _VectorType_                    __vector,
    __simd_mask_type<_DesiredType_> __mask) noexcept
{
    constexpr auto __length = sizeof(_VectorType_) / sizeof(_DesiredType_);

    if constexpr (sizeof(_DesiredType_) >= 4) {
        const auto __processed_bytes = __tables_avx<sizeof(_DesiredType_)>.__size[__mask];
        const auto __shuffle = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(__tables_avx<sizeof(_DesiredType_)>.__shuffle[__mask]));

        const auto __converted_shuffle = _mm256_cvtepu8_epi32(__shuffle);
        __vector = __intrin_bitcast<_VectorType_>(_mm256_permutevar8x32_epi32(__intrin_bitcast<__m256i>(__vector), __converted_shuffle));

        return { __processed_bytes, __vector };
    }
    else if constexpr (sizeof(_DesiredType_) == 2) {
        alignas(sizeof(_VectorType_)) _DesiredType_ __temporary_stack_buffer[__length];
        _DesiredType_* __destination_write_pointer = __temporary_stack_buffer;

        const auto __lower_lane_vector = __intrin_bitcast<__m128i>(__vector);
        const auto __higher_lane_vector = _mm256_extracti128_si256(__intrin_bitcast<__m256i>(__vector), 1);

        const auto __mask_segment_lower = __mask & 0xFF;
        const auto __mask_segment_higher = (__mask >> 8) & 0xFF;

        const auto __processed_byte_count_lower_segment = __tables_sse<sizeof(_DesiredType_)>.__size[__mask_segment_lower];
        const auto __processed_byte_count_higher_segment = __tables_sse<sizeof(_DesiredType_)>.__size[__mask_segment_higher];

        const auto __total_processed_byte_count_combined = __processed_byte_count_lower_segment + __processed_byte_count_higher_segment;
        const auto __total_processed_element_count_combined = __total_processed_byte_count_combined / sizeof(_DesiredType_);

        const auto __unprocessed_tail_blending_mask = (__simd_mask_type<_DesiredType_>(1u << (__length - __total_processed_element_count_combined)) - 1) << __total_processed_element_count_combined;

        const auto __shuffle_control_mask_lower_segment = _mm_load_si128(reinterpret_cast<const __m128i*>(__tables_sse<sizeof(_DesiredType_)>.__shuffle[__mask_segment_lower]));
        const auto __shuffle_control_mask_higher_segment = _mm_load_si128(reinterpret_cast<const __m128i*>(__tables_sse<sizeof(_DesiredType_)>.__shuffle[__mask_segment_higher]));

        const auto __packed_data_lower_segment = _mm_shuffle_epi8(__lower_lane_vector, __shuffle_control_mask_lower_segment);
        const auto __packed_data_higher_segment = _mm_shuffle_epi8(__higher_lane_vector, __shuffle_control_mask_higher_segment);

        _mm_store_si128(reinterpret_cast<__m128i*>(__destination_write_pointer), __intrin_bitcast<__m128i>(__packed_data_lower_segment));
        algorithm::__advance_bytes(__destination_write_pointer, __processed_byte_count_lower_segment);

        _mm_store_si128(reinterpret_cast<__m128i*>(__destination_write_pointer), __intrin_bitcast<__m128i>(__packed_data_higher_segment));

        const auto __final_packed_vector = _mm256_load_si256(reinterpret_cast<const __m256i*>(__temporary_stack_buffer));
        const auto __final_blended_result_vector = __blend<_DesiredType_>(__intrin_bitcast<__m256i>(__vector), __final_packed_vector, __unprocessed_tail_blending_mask);

        return { __total_processed_byte_count_combined, __intrin_bitcast<_VectorType_>(__final_blended_result_vector) };
    }
    else if constexpr (sizeof(_DesiredType_) == 1) {
        alignas(sizeof(_VectorType_)) _DesiredType_ __temporary_stack_buffer[__length];
        _DesiredType_* __destination_write_pointer = __temporary_stack_buffer;

        const auto __lower_lane_vector  = __intrin_bitcast<__m128i>(__vector);
        const auto __higher_lane_vector = _mm256_extracti128_si256(__intrin_bitcast<__m256i>(__vector), 1);

        const auto __lower_lane_upper_half_vector   = __intrin_bitcast<__m128i>(_mm_movehl_ps(
            __intrin_bitcast<__m128>(_mm_slli_si128(__lower_lane_vector, 8)),
            __intrin_bitcast<__m128>(__lower_lane_vector)));

        const auto __higher_lane_upper_half_vector  = __intrin_bitcast<__m128i>(_mm_movehl_ps(
            __intrin_bitcast<__m128>(_mm_slli_si128(__higher_lane_vector, 8)),
            __intrin_bitcast<__m128>(__higher_lane_vector)));

        const auto __mask_segment_first     = __mask & 0xFF;
        const auto __mask_segment_second    = (__mask >> 8) & 0xFF;
        const auto __mask_segment_third     = (__mask >> 16) & 0xFF;
        const auto __mask_segment_fourth    = (__mask >> 24) & 0xFF;

        const auto __processed_byte_count_first_segment     = __tables_sse<sizeof(_DesiredType_)>.__size[__mask_segment_first];
        const auto __processed_byte_count_second_segment    = __tables_sse<sizeof(_DesiredType_)>.__size[__mask_segment_second];
        const auto __processed_byte_count_third_segment     = __tables_sse<sizeof(_DesiredType_)>.__size[__mask_segment_third];
        const auto __processed_byte_count_fourth_segment    = __tables_sse<sizeof(_DesiredType_)>.__size[__mask_segment_fourth];

        const auto __total_processed_byte_count_lower_lane = __processed_byte_count_first_segment + __processed_byte_count_second_segment;
        const auto __total_processed_byte_count_higher_lane = __processed_byte_count_third_segment + __processed_byte_count_fourth_segment;

        const auto __total_processed_byte_count_combined = __total_processed_byte_count_lower_lane + __total_processed_byte_count_higher_lane;
        const auto __unprocessed_tail_blending_mask = (__simd_mask_type<_DesiredType_>(1u << (sizeof(_VectorType_) - __total_processed_byte_count_combined)) - 1) << __total_processed_byte_count_combined;

        const auto __shuffle_control_mask_first_segment     = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(__tables_sse<sizeof(_DesiredType_)>.__shuffle[__mask_segment_first]));
        const auto __shuffle_control_mask_second_segment    = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(__tables_sse<sizeof(_DesiredType_)>.__shuffle[__mask_segment_second]));
        const auto __shuffle_control_mask_third_segment     = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(__tables_sse<sizeof(_DesiredType_)>.__shuffle[__mask_segment_third]));
        const auto __shuffle_control_mask_fourth_segment    = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(__tables_sse<sizeof(_DesiredType_)>.__shuffle[__mask_segment_fourth]));

        const auto __packed_data_first_segment  = _mm_shuffle_epi8(__lower_lane_vector, __shuffle_control_mask_first_segment);
        const auto __packed_data_second_segment = _mm_shuffle_epi8(__lower_lane_upper_half_vector, __shuffle_control_mask_second_segment);
        const auto __packed_data_third_segment  = _mm_shuffle_epi8(__higher_lane_vector, __shuffle_control_mask_third_segment);
        const auto __packed_data_fourth_segment = _mm_shuffle_epi8(__higher_lane_upper_half_vector, __shuffle_control_mask_fourth_segment);

        _mm_storel_epi64(reinterpret_cast<__m128i*>(__destination_write_pointer), __intrin_bitcast<__m128i>(__packed_data_first_segment));
        __destination_write_pointer += __processed_byte_count_first_segment;

        _mm_storel_epi64(reinterpret_cast<__m128i*>(__destination_write_pointer), __intrin_bitcast<__m128i>(__packed_data_second_segment));
        __destination_write_pointer += __processed_byte_count_second_segment;

        _mm_storel_epi64(reinterpret_cast<__m128i*>(__destination_write_pointer), __intrin_bitcast<__m128i>(__packed_data_third_segment));
        __destination_write_pointer += __processed_byte_count_third_segment;

        _mm_storel_epi64(reinterpret_cast<__m128i*>(__destination_write_pointer), __intrin_bitcast<__m128i>(__packed_data_fourth_segment));

        const auto __final_packed_vector = _mm256_load_si256(reinterpret_cast<const __m256i*>(__temporary_stack_buffer));
        const auto __final_blended_result_vector = __blend<_DesiredType_>(__intrin_bitcast<__m256i>(__vector), __final_packed_vector, __unprocessed_tail_blending_mask);

        return { __total_processed_byte_count_combined, __intrin_bitcast<_VectorType_>(__final_blended_result_vector) };
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
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline std::pair<int32, _VectorType_> __simd_element_wise<arch::CpuFeature::AVX512F, zmm512>::__compress(
    _VectorType_    __vector,
    _VectorType_    __mask) noexcept
{
    return __compress<_DesiredType_>(__vector, __simd_to_mask<__generation, __register_policy, _DesiredType_>(__mask));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline std::pair<int32, _VectorType_> __simd_element_wise<arch::CpuFeature::AVX512F, zmm512>::__compress(
    _VectorType_                    __vector,
    __simd_mask_type<_DesiredType_> __mask) noexcept
{
    constexpr auto __length = sizeof(_VectorType_) / sizeof(_DesiredType_);

    if constexpr (sizeof(_DesiredType_) == 8) {
        const auto __not = uint8(~__mask);
        const auto __processed_bytes = (math::population_count(__not) << 3);

        return { __processed_bytes, __intrin_bitcast<_VectorType_>(_mm512_mask_compress_epi64(
            __intrin_bitcast<__m512i>(__vector), __not, __intrin_bitcast<__m512i>(__vector))) };
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        const auto __not = uint16(~__mask);
        const auto __processed_bytes = (math::population_count(__not) << 2);

        return { __processed_bytes, __intrin_bitcast<_VectorType_>(_mm512_mask_compress_epi32(
            __intrin_bitcast<__m512i>(__vector), __not, __intrin_bitcast<__m512i>(__vector))) };
    }
    else if constexpr (sizeof(_DesiredType_) == 2) {
        alignas(sizeof(_VectorType_)) _DesiredType_ __temporary_stack_buffer[__length];
        _DesiredType_* __destination_write_pointer = __temporary_stack_buffer;

        const auto __xmm_lane_vector1 = __intrin_bitcast<__m128i>(__vector);
        const auto __xmm_lane_vector2 = _mm256_extracti128_si256(__intrin_bitcast<__m256i>(__vector), 1);
        const auto __xmm_lane_vector3 = __intrin_bitcast<__m128i>(_mm512_extractf32x4_ps(__intrin_bitcast<__m512>(__vector), 2));
        const auto __xmm_lane_vector4 = __intrin_bitcast<__m128i>(_mm512_extractf32x4_ps(__intrin_bitcast<__m512>(__vector), 3));

        const auto __mask_segment_first     = __mask & 0xFF;
        const auto __mask_segment_second    = (__mask >> 8) & 0xFF;
        const auto __mask_segment_third     = (__mask >> 16) & 0xFF;
        const auto __mask_segment_fourth    = (__mask >> 24) & 0xFF;

        const auto __processed_byte_count_first_segment     = __tables_sse<sizeof(_DesiredType_)>.__size[__mask_segment_first];
        const auto __processed_byte_count_second_segment   = __tables_sse<sizeof(_DesiredType_)>.__size[__mask_segment_second];
        const auto __processed_byte_count_third_segment     = __tables_sse<sizeof(_DesiredType_)>.__size[__mask_segment_third];
        const auto __processed_byte_count_fourth_segment    = __tables_sse<sizeof(_DesiredType_)>.__size[__mask_segment_fourth];

        const auto __total_processed_byte_count_combined = __processed_byte_count_first_segment + __processed_byte_count_second_segment
            + __processed_byte_count_third_segment + __processed_byte_count_fourth_segment;
        const auto __total_processed_element_count_combined = __total_processed_byte_count_combined / sizeof(_DesiredType_);

        const auto __unprocessed_tail_blending_mask = (__simd_mask_type<_DesiredType_>(1u << (__length - __total_processed_element_count_combined)) - 1) << __total_processed_element_count_combined;

        const auto __shuffle_control_mask_first_segment     = _mm_load_si128(reinterpret_cast<const __m128i*>(__tables_sse<sizeof(_DesiredType_)>.__shuffle[__mask_segment_first]));
        const auto __shuffle_control_mask_second_segment    = _mm_load_si128(reinterpret_cast<const __m128i*>(__tables_sse<sizeof(_DesiredType_)>.__shuffle[__mask_segment_second]));
        const auto __shuffle_control_mask_third_segment     = _mm_load_si128(reinterpret_cast<const __m128i*>(__tables_sse<sizeof(_DesiredType_)>.__shuffle[__mask_segment_third]));
        const auto __shuffle_control_mask_fourth_segment    = _mm_load_si128(reinterpret_cast<const __m128i*>(__tables_sse<sizeof(_DesiredType_)>.__shuffle[__mask_segment_fourth]));

        const auto __packed_data_first_segment  = _mm_shuffle_epi8(__xmm_lane_vector1, __shuffle_control_mask_first_segment);
        const auto __packed_data_second_segment = _mm_shuffle_epi8(__xmm_lane_vector2, __shuffle_control_mask_second_segment);
        const auto __packed_data_third_segment  = _mm_shuffle_epi8(__xmm_lane_vector3, __shuffle_control_mask_third_segment);
        const auto __packed_data_fourth_segment = _mm_shuffle_epi8(__xmm_lane_vector4, __shuffle_control_mask_fourth_segment);

        _mm_store_si128(reinterpret_cast<__m128i*>(__destination_write_pointer), __intrin_bitcast<__m128i>(__packed_data_first_segment));
        algorithm::__advance_bytes(__destination_write_pointer, __processed_byte_count_first_segment);

        _mm_store_si128(reinterpret_cast<__m128i*>(__destination_write_pointer), __intrin_bitcast<__m128i>(__packed_data_second_segment));
        algorithm::__advance_bytes(__destination_write_pointer, __processed_byte_count_second_segment);

        _mm_store_si128(reinterpret_cast<__m128i*>(__destination_write_pointer), __intrin_bitcast<__m128i>(__packed_data_third_segment));
        algorithm::__advance_bytes(__destination_write_pointer, __processed_byte_count_third_segment);

        _mm_store_si128(reinterpret_cast<__m128i*>(__destination_write_pointer), __intrin_bitcast<__m128i>(__packed_data_fourth_segment));

        const auto __final_packed_vector = _mm512_load_si512(__temporary_stack_buffer);
        const auto __final_blended_result_vector = __blend<_DesiredType_>(__intrin_bitcast<__m512i>(__vector), __final_packed_vector, __unprocessed_tail_blending_mask);

        return { __total_processed_byte_count_combined, __intrin_bitcast<_VectorType_>(__final_blended_result_vector) };
    }
    else if constexpr (sizeof(_DesiredType_) == 1) {
        alignas(sizeof(_VectorType_)) _DesiredType_ __temporary_stack_buffer[__length];
        _DesiredType_* __destination_write_pointer = __temporary_stack_buffer;

        const auto __ymm_lower_lane_vector  = __intrin_bitcast<__m256i>(__vector);;
        const auto __ymm_higher_lane_vector = _mm512_extractf64x4_pd(__intrin_bitcast<__m512d>(__vector), 1);

        const auto __xmm_lane_vector1   = __intrin_bitcast<__m128i>(__ymm_lower_lane_vector);
        const auto __xmm_lane_vector2   = __intrin_bitcast<__m128i>(_mm256_extractf128_pd(__intrin_bitcast<__m256d>(__ymm_lower_lane_vector), 1));

        const auto __xmm_lane_vector3   = __intrin_bitcast<__m128i>(__ymm_higher_lane_vector);
        const auto __xmm_lane_vector4   = __intrin_bitcast<__m128i>(_mm256_extractf128_pd(__intrin_bitcast<__m256d>(__ymm_higher_lane_vector), 1));

        const auto __xmm_lane_upper_half_vector1   = __intrin_bitcast<__m128i>(_mm_movehl_ps(
            __intrin_bitcast<__m128>(_mm_slli_si128(__xmm_lane_vector1, 8)),
            __intrin_bitcast<__m128>(__xmm_lane_vector1)));

        const auto __xmm_lane_upper_half_vector2 = __intrin_bitcast<__m128i>(_mm_movehl_ps(
            __intrin_bitcast<__m128>(_mm_slli_si128(__xmm_lane_vector2, 8)),
            __intrin_bitcast<__m128>(__xmm_lane_vector2)));

        const auto __xmm_lane_upper_half_vector3 = __intrin_bitcast<__m128i>(_mm_movehl_ps(
            __intrin_bitcast<__m128>(_mm_slli_si128(__xmm_lane_vector3, 8)),
            __intrin_bitcast<__m128>(__xmm_lane_vector3)));

        const auto __xmm_lane_upper_half_vector4 = __intrin_bitcast<__m128i>(_mm_movehl_ps(
            __intrin_bitcast<__m128>(_mm_slli_si128(__xmm_lane_vector4, 8)),
            __intrin_bitcast<__m128>(__xmm_lane_vector4)));

        const auto __mask_segment_first     = __mask & 0xFF;
        const auto __mask_segment_second    = (__mask >> 8) & 0xFF;
        const auto __mask_segment_third     = (__mask >> 16) & 0xFF;
        const auto __mask_segment_fourth    = (__mask >> 24) & 0xFF;
        const auto __mask_segment_fifth     = (__mask >> 32) & 0xFF;
        const auto __mask_segment_sixth     = (__mask >> 40) & 0xFF;
        const auto __mask_segment_seventh   = (__mask >> 48) & 0xFF;
        const auto __mask_segment_eighth    = (__mask >> 56) & 0xFF;

        const auto __processed_byte_count_first_segment     = __tables_sse<sizeof(_DesiredType_)>.__size[__mask_segment_first];
        const auto __processed_byte_count_second_segment    = __tables_sse<sizeof(_DesiredType_)>.__size[__mask_segment_second];
        const auto __processed_byte_count_third_segment     = __tables_sse<sizeof(_DesiredType_)>.__size[__mask_segment_third];
        const auto __processed_byte_count_fourth_segment    = __tables_sse<sizeof(_DesiredType_)>.__size[__mask_segment_fourth];
        const auto __processed_byte_count_fifth_segment     = __tables_sse<sizeof(_DesiredType_)>.__size[__mask_segment_fifth];
        const auto __processed_byte_count_sixth_segment     = __tables_sse<sizeof(_DesiredType_)>.__size[__mask_segment_sixth];
        const auto __processed_byte_count_seventh_segment   = __tables_sse<sizeof(_DesiredType_)>.__size[__mask_segment_seventh];
        const auto __processed_byte_count_eighth_segment    = __tables_sse<sizeof(_DesiredType_)>.__size[__mask_segment_eighth];

        const auto __total_processed_byte_count_xmm_lane1   = __processed_byte_count_first_segment + __processed_byte_count_second_segment;
        const auto __total_processed_byte_count_xmm_lane2   = __processed_byte_count_third_segment + __processed_byte_count_fourth_segment;
        const auto __total_processed_byte_count_xmm_lane3   = __processed_byte_count_fifth_segment + __processed_byte_count_sixth_segment;
        const auto __total_processed_byte_count_xmm_lane4   = __processed_byte_count_seventh_segment + __processed_byte_count_eighth_segment;

        const auto __total_processed_byte_count_combined = __total_processed_byte_count_xmm_lane1 + __total_processed_byte_count_xmm_lane2
            + __total_processed_byte_count_xmm_lane3 + __total_processed_byte_count_xmm_lane4;

        const auto __unprocessed_tail_blending_mask = (__simd_mask_type<_DesiredType_>(uint64(1) << (__length - __total_processed_byte_count_combined)) - 1) << __total_processed_byte_count_combined;

        const auto __shuffle_control_mask_first_segment     = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(__tables_sse<sizeof(_DesiredType_)>.__shuffle[__mask_segment_first]));
        const auto __shuffle_control_mask_second_segment    = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(__tables_sse<sizeof(_DesiredType_)>.__shuffle[__mask_segment_second]));
        const auto __shuffle_control_mask_third_segment     = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(__tables_sse<sizeof(_DesiredType_)>.__shuffle[__mask_segment_third]));
        const auto __shuffle_control_mask_fourth_segment    = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(__tables_sse<sizeof(_DesiredType_)>.__shuffle[__mask_segment_fourth]));
        const auto __shuffle_control_mask_fifth_segment     = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(__tables_sse<sizeof(_DesiredType_)>.__shuffle[__mask_segment_fifth]));
        const auto __shuffle_control_mask_sixth_segment     = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(__tables_sse<sizeof(_DesiredType_)>.__shuffle[__mask_segment_sixth]));
        const auto __shuffle_control_mask_seventh_segment   = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(__tables_sse<sizeof(_DesiredType_)>.__shuffle[__mask_segment_seventh]));
        const auto __shuffle_control_mask_eighth_segment    = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(__tables_sse<sizeof(_DesiredType_)>.__shuffle[__mask_segment_eighth]));

        const auto __packed_data_first_segment      = _mm_shuffle_epi8(__xmm_lane_vector1, __shuffle_control_mask_first_segment);
        const auto __packed_data_second_segment     = _mm_shuffle_epi8(__xmm_lane_upper_half_vector1, __shuffle_control_mask_second_segment);
        const auto __packed_data_third_segment      = _mm_shuffle_epi8(__xmm_lane_vector2, __shuffle_control_mask_third_segment);
        const auto __packed_data_fourth_segment     = _mm_shuffle_epi8(__xmm_lane_upper_half_vector2, __shuffle_control_mask_fourth_segment);
        const auto __packed_data_fifth_segment      = _mm_shuffle_epi8(__xmm_lane_vector3, __shuffle_control_mask_fifth_segment);
        const auto __packed_data_sixth_segment      = _mm_shuffle_epi8(__xmm_lane_upper_half_vector3, __shuffle_control_mask_sixth_segment);
        const auto __packed_data_seventh_segment    = _mm_shuffle_epi8(__xmm_lane_vector4, __shuffle_control_mask_seventh_segment);
        const auto __packed_data_eighth_segment     = _mm_shuffle_epi8(__xmm_lane_upper_half_vector4, __shuffle_control_mask_eighth_segment);

        _mm_storel_epi64(reinterpret_cast<__m128i*>(__destination_write_pointer), __intrin_bitcast<__m128i>(__packed_data_first_segment));
        algorithm::__advance_bytes(__destination_write_pointer, __processed_byte_count_first_segment);

        _mm_storel_epi64(reinterpret_cast<__m128i*>(__destination_write_pointer), __intrin_bitcast<__m128i>(__packed_data_second_segment));
        algorithm::__advance_bytes(__destination_write_pointer, __processed_byte_count_second_segment);

        _mm_storel_epi64(reinterpret_cast<__m128i*>(__destination_write_pointer), __intrin_bitcast<__m128i>(__packed_data_third_segment));
        algorithm::__advance_bytes(__destination_write_pointer, __processed_byte_count_third_segment);

        _mm_storel_epi64(reinterpret_cast<__m128i*>(__destination_write_pointer), __intrin_bitcast<__m128i>(__packed_data_fourth_segment));
        algorithm::__advance_bytes(__destination_write_pointer, __processed_byte_count_fourth_segment);
        
        _mm_storel_epi64(reinterpret_cast<__m128i*>(__destination_write_pointer), __intrin_bitcast<__m128i>(__packed_data_fifth_segment));
        algorithm::__advance_bytes(__destination_write_pointer, __processed_byte_count_fifth_segment);

        _mm_storel_epi64(reinterpret_cast<__m128i*>(__destination_write_pointer), __intrin_bitcast<__m128i>(__packed_data_sixth_segment));
        algorithm::__advance_bytes(__destination_write_pointer, __processed_byte_count_sixth_segment);

        _mm_storel_epi64(reinterpret_cast<__m128i*>(__destination_write_pointer), __intrin_bitcast<__m128i>(__packed_data_seventh_segment));
        algorithm::__advance_bytes(__destination_write_pointer, __processed_byte_count_seventh_segment);

        _mm_storel_epi64(reinterpret_cast<__m128i*>(__destination_write_pointer), __intrin_bitcast<__m128i>(__packed_data_eighth_segment));

        const auto __final_packed_vector = _mm512_load_si512(__temporary_stack_buffer);
        const auto __final_blended_result_vector = __blend<_DesiredType_>(__intrin_bitcast<__m512i>(__vector), __final_packed_vector, __unprocessed_tail_blending_mask);

        return { __total_processed_byte_count_combined, __intrin_bitcast<_VectorType_>(__final_blended_result_vector) };
    }
}

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

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline std::pair<int32, _VectorType_> __simd_element_wise<arch::CpuFeature::AVX512VLF, ymm256>::__compress(
    _VectorType_    __vector,
    _VectorType_    __mask) noexcept
{
    return __compress<_DesiredType_>(__vector, __simd_to_mask<__generation, __register_policy, _DesiredType_>(__mask));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline std::pair<int32, _VectorType_> __simd_element_wise<arch::CpuFeature::AVX512VLF, ymm256>::__compress(
    _VectorType_                    __vector,
    __simd_mask_type<_DesiredType_> __mask) noexcept
{
    constexpr auto __length = sizeof(_VectorType_) / sizeof(_DesiredType_);

    if constexpr (sizeof(_DesiredType_) == 8) {
        const auto __not = uint8(uint8(0xF) & uint8(~__mask));
        const auto __processed_bytes = (math::population_count(__not) << 3);

        return { __processed_bytes, __intrin_bitcast<_VectorType_>(_mm256_mask_compress_epi64(
            __intrin_bitcast<__m256i>(__vector), __not, __intrin_bitcast<__m256i>(__vector))) };
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        const auto __not = uint16(uint16(0xFF) & uint16(~__mask));
        const auto __processed_bytes = (math::population_count(__not) << 2);

        return { __processed_bytes, __intrin_bitcast<_VectorType_>(_mm256_mask_compress_epi32(
            __intrin_bitcast<__m256i>(__vector), __not, __intrin_bitcast<__m256i>(__vector))) };
    }
    else {
        return __simd_compress<arch::CpuFeature::AVX2, __register_policy, _DesiredType_>(__vector, __mask);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline std::pair<int32, _VectorType_> __simd_element_wise<arch::CpuFeature::AVX512VLF, xmm128>::__compress(
    _VectorType_    __vector,
    _VectorType_    __mask) noexcept
{
    return __compress<_DesiredType_>(__vector, __simd_to_mask<__generation, __register_policy, _DesiredType_>(__mask));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline std::pair<int32, _VectorType_> __simd_element_wise<arch::CpuFeature::AVX512VLF, xmm128>::__compress(
    _VectorType_                    __vector,
    __simd_mask_type<_DesiredType_> __mask) noexcept
{
    constexpr auto __length = sizeof(_VectorType_) / sizeof(_DesiredType_);

    if constexpr (sizeof(_DesiredType_) == 8) {
        const auto __not = uint8(uint8(0x03) & uint8(~__mask));
        const auto __processed_bytes = (math::population_count(__not) << 3);

        return { __processed_bytes, __intrin_bitcast<_VectorType_>(_mm_mask_compress_epi64(
            __intrin_bitcast<__m128i>(__vector), __not, __intrin_bitcast<__m128i>(__vector))) };
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        const auto __not = uint8(uint8(0xF) & uint8(~__mask));
        const auto __processed_bytes = (math::population_count(__not) << 2);

        return { __processed_bytes, __intrin_bitcast<_VectorType_>(_mm_mask_compress_epi32(
            __intrin_bitcast<__m128i>(__vector), __not, __intrin_bitcast<__m128i>(__vector))) };
    }
    else {
        return __simd_compress<arch::CpuFeature::SSE42, __register_policy, _DesiredType_>(__vector, __mask);
    }
}

#pragma endregion

__SIMD_STL_DATAPAR_NAMESPACE_END
