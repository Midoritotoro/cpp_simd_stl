#pragma once 

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN


#pragma region Sse2-Sse4.2 Simd element access


template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_element_access<arch::CpuFeature::SSE2, xmm128>::__insert(
    _VectorType_&       __vector,
    const uint8         __position,
    const _DesiredType_ __value) noexcept
{
    if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>) {
#if defined(simd_stl_processor_x86_64)
        auto __vector_value = _mm_cvtsi64_si128(memory::pointer_to_integral(__value));
#else
        union {
            __m128i __vector;
            int64   __number;
        } __convert;

        __convert.__number = __value;
        auto __vector_value = __simd_load_lower_half<__generation, __register_policy, __m128i>(&__convert.__vector);
#endif // defined(simd_stl_processor_x86_64)
        __vector = (__position == 0)
            ? __intrin_bitcast<_VectorType_>(_mm_unpackhi_epi64(
                _mm_unpacklo_epi64(__vector_value, __vector_value), __intrin_bitcast<__m128i>(__vector)))
            : __intrin_bitcast<_VectorType_>(_mm_unpacklo_epi64(
                __intrin_bitcast<__m128i>(__vector), __vector_value));
    }
    else if constexpr (__is_epi16_v<_DesiredType_> || __is_epu16_v<_DesiredType_>) {
        switch (_Position) {
        case 0:
            __vector = __intrin_bitcast<_VectorType_>(_mm_insert_epi16(__intrin_bitcast<__m128i>(__vector), __value, 0));
            break;
        case 1:
            __vector = __intrin_bitcast<_VectorType_>(_mm_insert_epi16(__intrin_bitcast<__m128i>(__vector), __value, 1));
            break;
        case 2:
            __vector = __intrin_bitcast<_VectorType_>(_mm_insert_epi16(__intrin_bitcast<__m128i>(__vector), __value, 2));
            break;
        case 3:
            __vector = __intrin_bitcast<_VectorType_>(_mm_insert_epi16(__intrin_bitcast<__m128i>(__vector), __value, 3));
            break;
        case 4:
            __vector = __intrin_bitcast<_VectorType_>(_mm_insert_epi16(__intrin_bitcast<__m128i>(__vector), __value, 4));
            break;
        case 5:
            __vector = __intrin_bitcast<_VectorType_>(_mm_insert_epi16(__intrin_bitcast<__m128i>(__vector), __value, 5));
            break;
        case 6:
            __vector = __intrin_bitcast<_VectorType_>(_mm_insert_epi16(__intrin_bitcast<__m128i>(__vector), __value, 6));
            break;
        default:
            __vector = __intrin_bitcast<_VectorType_>(_mm_insert_epi16(__intrin_bitcast<__m128i>(__vector), __value, 7));
            break;
        }
    }
    else if constexpr (__is_pd_v<_DesiredType_>) {
        const auto __broadcasted = _mm_set_sd(__value);

        __vector = (__position == 0)
            ? _mm_shuffle_pd(__broadcasted, __intrin_bitcast<__m128d>(__vector), 2)
            : _mm_shuffle_pd(__intrin_bitcast<__m128d>(__vector), __broadcasted, 0);
    }
    else {
        const auto __mask = __simd_make_insert_mask<_VectorType_, _DesiredType_>();

        const auto __broadcasted = __simd_broadcast<__generation, __register_policy, _VectorType_>(memory::pointer_to_integral(__value));
        const auto __insert_mask = __simd_load_unaligned<__generation, __register_policy, _VectorType_>(
            (__mask.__array + __mask.__offset - (__position & (__mask.__offset - 1))));

        __vector = __simd_blend<__generation, __register_policy, _DesiredType_>(__vector, __broadcasted, __insert_mask);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_ __simd_element_access<arch::CpuFeature::SSE2, xmm128>::__extract(
    _VectorType_    __vector,
    const uint8     __where) noexcept
{
    if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>) {
        if (__where == 0) {
#if defined(simd_stl_processor_x86_64)
            return static_cast<_DesiredType_>(_mm_cvtsi128_si64(__intrin_bitcast<__m128i>(__vector)));
#else
            const auto __high_dword = _mm_cvtsi128_si32(__intrin_bitcast<__m128i>(__vector));
            const auto __low_dword = _mm_cvtsi128_si32(_mm_shuffle_epi32(_IntrinBitcast<__m128i>(__vector), 0x55));

            return (static_cast<int64>(__high_dword) << 32) | static_cast<int64>(__low_dword);
#endif // defined(simd_stl_processor_x86_64)
        }

#if defined(simd_stl_processor_x86_64)
        const auto __shuffled = _mm_shuffle_epi32(__intrin_bitcast<__m128i>(__vector), 0xEE);
        return static_cast<_DesiredType_>(_mm_cvtsi128_si64(__shuffled));
#else
        const auto __high_dword = _mm_cvtsi128_si32(_mm_shuffle_epi32(__intrin_bitcast<__m128i>(__vector), 0xEE));
        const auto __low_dword = _mm_cvtsi128_si32(_mm_shuffle_epi32(__intrin_bitcast<__m128i>(__vector), 0xFF));

        return (static_cast<int64>(__high_dword) << 32) | static_cast<int64>(__low_dword);
#endif // defined(simd_stl_processor_x86_64)
    }
    else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>) {
        switch (__where) {
            case 0:
                return static_cast<_DesiredType_>(_mm_cvtsi128_si32(__intrin_bitcast<__m128i>(__vector)));

            case 1:
                return static_cast<_DesiredType_>(_mm_cvtsi128_si32(_mm_shuffle_epi32(__intrin_bitcast<__m128i>(__vector), 0x55)));

            case 2:
                return static_cast<_DesiredType_>(_mm_cvtsi128_si32(_mm_shuffle_epi32(__intrin_bitcast<__m128i>(__vector), 0xEE)));

            default:
                return static_cast<_DesiredType_>(_mm_cvtsi128_si32(_mm_shuffle_epi32(__intrin_bitcast<__m128i>(__vector), 0xFF)));
        }
    }
    else if constexpr (__is_epi16_v<_DesiredType_> || __is_epu16_v<_DesiredType_>) {
        switch (_Where) {
            case 0:
                return static_cast<_DesiredType_>(_mm_extract_epi16(__intrin_bitcast<__m128i>(__vector), 0));
            case 1:
                return static_cast<_DesiredType_>(_mm_extract_epi16(__intrin_bitcast<__m128i>(__vector), 1));
            case 2:
                return static_cast<_DesiredType_>(_mm_extract_epi16(__intrin_bitcast<__m128i>(__vector), 2));
            case 3:
                return static_cast<_DesiredType_>(_mm_extract_epi16(__intrin_bitcast<__m128i>(__vector), 3));
            case 4:
                return static_cast<_DesiredType_>(_mm_extract_epi16(__intrin_bitcast<__m128i>(__vector), 4));
            case 5:
                return static_cast<_DesiredType_>(_mm_extract_epi16(__intrin_bitcast<__m128i>(__vector), 5));
            case 6:
                return static_cast<_DesiredType_>(_mm_extract_epi16(__intrin_bitcast<__m128i>(__vector), 6));
            default:
                return static_cast<_DesiredType_>(_mm_extract_epi16(__intrin_bitcast<__m128i>(__vector), 7));
        }
    }
    else {
        constexpr auto __length = sizeof(_VectorType_) / sizeof(_DesiredType_);
        _DesiredType_ __array[__length];

        __simd_store_unaligned<__generation, __register_policy>(__array, __vector);
        return __array[__where & (__length - 1)];
    }
}


template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_element_access<arch::CpuFeature::SSE41, xmm128>::__insert(
    _VectorType_&       __vector,
    const uint8         __position,
    const _DesiredType_ __value) noexcept
{
    if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>) {
        auto __qword_value = memory::pointer_to_integral(__value);

        switch (_Position) {
            case 0:
                __vector = __intrin_bitcast<_VectorType_>(_mm_insert_epi64(__intrin_bitcast<__m128i>(__vector), __qword_value, 0));
                break;
            default:
                __vector = __intrin_bitcast<_VectorType_>(_mm_insert_epi64(__intrin_bitcast<__m128i>(__vector), __qword_value, 1));
                break;
        }
    }
    else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>) {
        auto __dword_value = memory::pointerToIntegral(__value);

        switch (_Position) {
            case 0:
                __vector = __intrin_bitcast<_VectorType_>(_mm_insert_epi32(__intrin_bitcast<__m128i>(__vector), __dword_value, 0));
                break;
            case 1:
                __vector = __intrin_bitcast<_VectorType_>(_mm_insert_epi32(__intrin_bitcast<__m128i>(__vector), __dword_value, 1));
                break;
            case 2:
                __vector = __intrin_bitcast<_VectorType_>(_mm_insert_epi32(__intrin_bitcast<__m128i>(__vector), __dword_value, 2));
                break;
            default:
                __vector = __intrin_bitcast<_VectorType_>(_mm_insert_epi32(__intrin_bitcast<__m128i>(__vector), __dword_value, 3));
                break;
        }
    }
    else if constexpr (__is_epi16_v<_DesiredType_> || __is_epu16_v<_DesiredType_>) {
        switch (__position) {
            case 0:
                __vector = __intrin_bitcast<_VectorType_>(_mm_insert_epi16(__intrin_bitcast<__m128i>(__vector), __value, 0));
                break;
            case 1:
                __vector = __intrin_bitcast<_VectorType_>(_mm_insert_epi16(__intrin_bitcast<__m128i>(__vector), __value, 1));
                break;
            case 2:
                __vector = __intrin_bitcast<_VectorType_>(_mm_insert_epi16(__intrin_bitcast<__m128i>(__vector), __value, 2));
                break;
            case 3:
                __vector = __intrin_bitcast<_VectorType_>(_mm_insert_epi16(__intrin_bitcast<__m128i>(__vector), __value, 3));
                break;
            case 4:
                __vector = __intrin_bitcast<_VectorType_>(_mm_insert_epi16(__intrin_bitcast<__m128i>(__vector), __value, 4));
                break;
            case 5:
                __vector = __intrin_bitcast<_VectorType_>(_mm_insert_epi16(__intrin_bitcast<__m128i>(__vector), __value, 5));
                break;
            case 6:
                __vector = __intrin_bitcast<_VectorType_>(_mm_insert_epi16(__intrin_bitcast<__m128i>(__vector), __value, 6));
                break;
            default:
                __vector = __intrin_bitcast<_VectorType_>(_mm_insert_epi16(__intrin_bitcast<__m128i>(__vector), __value, 7));
                break;
        }
    }
    else {
        return __simd_insert<arch::CpuFeature::SSE2, __register_policy>(__vector, __position, __value);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_ __simd_element_access<arch::CpuFeature::SSE41, xmm128>::__extract(
    _VectorType_    __vector,
    const uint8     __where) noexcept
{
    if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>) {
        switch (__where) {
            case 0:
                return static_cast<_DesiredType_>(_mm_extract_epi64(__intrin_bitcast<__m128i>(__vector), 0));
            default:
                return static_cast<_DesiredType_>(_mm_extract_epi64(__intrin_bitcast<__m128i>(__vector), 1));
        }
    }
    else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>) {
        switch (__where) {
            case 0:
                return static_cast<_DesiredType_>(_mm_extract_epi32(__intrin_bitcast<__m128i>(__vector), 0));
            case 1:
                return static_cast<_DesiredType_>(_mm_extract_epi32(__intrin_bitcast<__m128i>(__vector), 1));
            case 2:
                return static_cast<_DesiredType_>(_mm_extract_epi32(__intrin_bitcast<__m128i>(__vector), 2));
            default:
                return static_cast<_DesiredType_>(_mm_extract_epi32(__intrin_bitcast<__m128i>(__vector), 3));
        }
    }
    else if constexpr (__is_epi16_v<_DesiredType_> || __is_epu16_v<_DesiredType_>) {
        switch (__where) {
            case 0:
                return static_cast<_DesiredType_>(_mm_extract_epi16(__intrin_bitcast<__m128i>(__vector), 0));
            case 1:
                return static_cast<_DesiredType_>(_mm_extract_epi16(__intrin_bitcast<__m128i>(__vector), 1));
            case 2:
                return static_cast<_DesiredType_>(_mm_extract_epi16(__intrin_bitcast<__m128i>(__vector), 2));
            case 3:
                return static_cast<_DesiredType_>(_mm_extract_epi16(__intrin_bitcast<__m128i>(__vector), 3));
            case 4:
                return static_cast<_DesiredType_>(_mm_extract_epi16(__intrin_bitcast<__m128i>(__vector), 4));
            case 5:
                return static_cast<_DesiredType_>(_mm_extract_epi16(__intrin_bitcast<__m128i>(__vector), 5));
            case 6:
                return static_cast<_DesiredType_>(_mm_extract_epi16(__intrin_bitcast<__m128i>(__vector), 6));
            default:
                return static_cast<_DesiredType_>(_mm_extract_epi16(__intrin_bitcast<__m128i>(__vector), 7));
        }
    }
    else {
        return __simd_extract<arch::CpuFeature::SSE2, __register_policy, _DesiredType_>(__vector, __where);
    }
}

#pragma endregion

#pragma region Avx-Avx2 Simd element access 


template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_element_access<arch::CpuFeature::AVX, ymm256>::__insert(
    _VectorType_&       __vector,
    const uint8         __position,
    const _DesiredType_ __value) noexcept
{
    if constexpr (__is_pd_v<_DesiredType_>) {
        const auto __broadcasted = _mm256_broadcast_sd(&__value);

        switch (__position) {
            case 0:
                __vector = _mm256_blend_pd(__intrin_bitcast<__m256d>(__vector), __broadcasted, 1);
                break;
            case 1:
                __vector = _mm256_blend_pd(__intrin_bitcast<__m256d>(__vector), __broadcasted, 2);
                break;
            case 2:
                __vector = _mm256_blend_pd(__intrin_bitcast<__m256d>(__vector), __broadcasted, 4);
                break;
            default:
                __vector = _mm256_blend_pd(__intrin_bitcast<__m256d>(__vector), __broadcasted, 8);
                break;
        }
    }
    else if constexpr (__is_ps_v<_DesiredType_>) {
        const auto __broadcasted = _mm256_broadcast_ss(&__value);

        switch (__position) {
            case 0:
                __vector = _mm256_blend_ps(__intrin_bitcast<__m256>(__vector), _Broadcasted, 1);
                break;
            case 1:
                __vector = _mm256_blend_ps(__intrin_bitcast<__m256>(__vector), _Broadcasted, 2);
                break;
            case 2:
                __vector = _mm256_blend_ps(__intrin_bitcast<__m256>(__vector), _Broadcasted, 4);
                break;
            case 3:
                __vector = _mm256_blend_ps(__intrin_bitcast<__m256>(__vector), _Broadcasted, 8);
                break;
            case 4:
                __vector = _mm256_blend_ps(__intrin_bitcast<__m256>(__vector), _Broadcasted, 0x10);
                break;
            case 5:
                __vector = _mm256_blend_ps(__intrin_bitcast<__m256>(__vector), _Broadcasted, 0x20);
                break;
            case 6:
                __vector = _mm256_blend_ps(__intrin_bitcast<__m256>(__vector), _Broadcasted, 0x40);
                break;
            default:
                __vector = _mm256_blend_ps(__intrin_bitcast<__m256>(__vector), _Broadcasted, 0x80);
                break;
        }
    }
    else if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>) {
        auto __qword_value = memory::pointer_to_integral(__value);

        switch (__position) {
            case 0:
                __vector = __intrin_bitcast<_VectorType_>(_mm256_insert_epi64(__intrin_bitcast<__m256i>(__vector), __qword_value, 0));
                break;
            case 1:
                __vector = __intrin_bitcast<_VectorType_>(_mm256_insert_epi64(__intrin_bitcast<__m256i>(__vector), __qword_value, 1));
                break;
            case 2:
                __vector = __intrin_bitcast<_VectorType_>(_mm256_insert_epi64(__intrin_bitcast<__m256i>(__vector), __qword_value, 2));
                break;
            default:
                __vector = __intrin_bitcast<_VectorType_>(_mm256_insert_epi64(__intrin_bitcast<__m256i>(__vector), __qword_value, 3));
                break;
        }
    }
    else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>) {
        auto __dword_value = memory::pointer_to_integral(__value);

        switch (__position) {
            case 0:
                __vector = __intrin_bitcast<_VectorType_>(_mm256_insert_epi32(__intrin_bitcast<__m256i>(__vector), __dword_value, 0));
                break;
            case 1:
                __vector = __intrin_bitcast<_VectorType_>(_mm256_insert_epi32(__intrin_bitcast<__m256i>(__vector), __dword_value, 1));
                break;
            case 2:
                __vector = __intrin_bitcast<_VectorType_>(_mm256_insert_epi32(__intrin_bitcast<__m256i>(__vector), __dword_value, 2));
                break;
            case 3:
                __vector = __intrin_bitcast<_VectorType_>(_mm256_insert_epi32(__intrin_bitcast<__m256i>(__vector), __dword_value, 3));
                break;
            case 4:
                __vector = __intrin_bitcast<_VectorType_>(_mm256_insert_epi32(__intrin_bitcast<__m256i>(__vector), __dword_value, 4));
                break;
            case 5:
                __vector = __intrin_bitcast<_VectorType_>(_mm256_insert_epi32(__intrin_bitcast<__m256i>(__vector), __dword_value, 5));
                break;
            case 6:
                __vector = __intrin_bitcast<_VectorType_>(_mm256_insert_epi32(__intrin_bitcast<__m256i>(__vector), __dword_value, 6));
                break;
            default:
                __vector = __intrin_bitcast<_VectorType_>(_mm256_insert_epi32(__intrin_bitcast<__m256i>(__vector), __dword_value, 7));
                break;
        }
    }
    else {
        const auto __mask = __simd_make_insert_mask<_VectorType_, _DesiredType_>();

        const auto __broadcasted = __simd_broadcast<__generation, __register_policy, _VectorType_>(__value);
        const auto __insert_mask = __simd_load_unaligned<__generation, __register_policy, _VectorType_>(
            (__mask.__array + __mask.__offset - (__position & (__mask.__offset - 1))));

        __vector = __simd_blend<__generation, __register_policy, _DesiredType_>(__vector, __broadcasted, __insert_mask);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_ __simd_element_access<arch::CpuFeature::AVX, ymm256>::__extract(
    _VectorType_    __vector,
    const uint8     __where) noexcept
{
    if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>) {
        switch (__where) {
            case 0:
                return static_cast<_DesiredType_>(_mm256_extract_epi64(__intrin_bitcast<__m256i>(__vector), 0));
            case 1:
                return static_cast<_DesiredType_>(_mm256_extract_epi64(__intrin_bitcast<__m256i>(__vector), 1));
            case 2:
                return static_cast<_DesiredType_>(_mm256_extract_epi64(__intrin_bitcast<__m256i>(__vector), 2));
            case 3:
                return static_cast<_DesiredType_>(_mm256_extract_epi64(__intrin_bitcast<__m256i>(__vector), 3));
        }
    }
    else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>) {
        switch (__where) {
            case 0:
                return static_cast<_DesiredType_>(_mm256_extract_epi32(__intrin_bitcast<__m256i>(__vector), 0));
            case 1:
                return static_cast<_DesiredType_>(_mm256_extract_epi32(__intrin_bitcast<__m256i>(__vector), 1));
            case 2:
                return static_cast<_DesiredType_>(_mm256_extract_epi32(__intrin_bitcast<__m256i>(__vector), 2));
            case 3:
                return static_cast<_DesiredType_>(_mm256_extract_epi32(__intrin_bitcast<__m256i>(__vector), 3));
            case 4:
                return static_cast<_DesiredType_>(_mm256_extract_epi32(__intrin_bitcast<__m256i>(__vector), 4));
            case 5:
                return static_cast<_DesiredType_>(_mm256_extract_epi32(__intrin_bitcast<__m256i>(__vector), 5));
            case 6:
                return static_cast<_DesiredType_>(_mm256_extract_epi32(__intrin_bitcast<__m256i>(__vector), 6));
            case 7:
                return static_cast<_DesiredType_>(_mm256_extract_epi32(__intrin_bitcast<__m256i>(__vector), 7));
        }
    }
    else {
        constexpr auto __length = sizeof(_VectorType_) / sizeof(_DesiredType_);
        _DesiredType_ __array[__length];

        __simd_store_unaligned<__generation, __register_policy>(__array, __vector);
        return __array[__where & (__length - 1)];
    }
}

#pragma endregion

#pragma region Avx512 Simd element access


template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_element_access<arch::CpuFeature::AVX512F, zmm512>::__insert(
    _VectorType_&       __vector,
    const uint8         __position,
    const _DesiredType_ __value) noexcept
{
    if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>) {
        __vector = __intrin_bitcast<_VectorType_>(_mm512_mask_set1_epi64(
            __intrin_bitcast<__m512i>(__vector),
            static_cast<uint8>(1u << __position), 
            memory::pointerToIntegral(__value)));
    }
    else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>) {
        __vector = __intrin_bitcast<_VectorType_>(_mm512_mask_set1_epi32(
            __intrin_bitcast<__m512i>(__vector),
            static_cast<uint16>(1u << __position),
            memory::pointerToIntegral(__value)));
    }
    else if constexpr (__is_ps_v<_DesiredType_>) {
        __vector = __intrin_bitcast<_VectorType_>(_mm512_mask_broadcastss_ps(
            __intrin_bitcast<__m512>(__vector),
            static_cast<uint16>(1u << __position),
            _mm_set_ss(__value)));
    }
    else if constexpr (__is_pd_v<_DesiredType_>) {
        __vector = __intrin_bitcast<_VectorType_>(_mm512_mask_broadcastsd_pd(
            __intrin_bitcast<__m512d>(__vector),
            static_cast<uint8>(1u << __position), 
            _mm_set_sd(__value)));
    }
    else {
        const auto __mask = __simd_make_insert_mask<_VectorType_, _DesiredType_>();

        const auto __broadcasted = __simd_broadcast<__generation, __register_policy, _VectorType_>(__value);
        const auto __insert_mask = __simd_load_unaligned<__generation, __register_policy, _VectorType_>(
            (__mask.__array + __mask.__offset - (__position & (__mask.__offset - 1))));

        __vector = __simd_blend<__generation, __register_policy, _DesiredType_>(__vector, __broadcasted, __insert_mask);
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_ __simd_element_access<arch::CpuFeature::AVX512F, zmm512>::__extract(
    _VectorType_    __vector,
    const uint8     __where) noexcept
{
    if constexpr (__is_pd_v<_DesiredType_>) {
        return _mm512_cvtsd_f64(_mm512_maskz_compress_pd(
            static_cast<uint8>(1u << __where), __intrin_bitcast<__m512d>(__vector)));
    }
    else if constexpr (__is_ps_v<_DesiredType_>) {
        return _mm512_cvtss_f32(_mm512_maskz_compress_ps(
            static_cast<uint16>(1u << __where), __intrin_bitcast<__m512>(__vector)));
    }
    else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>) {
        return _mm_cvtsi128_si32(__intrin_bitcast<__m128i>(_mm512_maskz_compress_epi32(
            static_cast<uint16>(1u << __where), __intrin_bitcast<__m512i>(__vector))));
    }
    else if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>) {
        return _mm_cvtsi128_si64(__intrin_bitcast<__m128i>(_mm512_maskz_compress_epi64(
            static_cast<uint16>(1u << __where), __intrin_bitcast<__m512i>(__vector))));
    }
    else {
        constexpr auto __length = sizeof(_VectorType_) / sizeof(_DesiredType_);
        _DesiredType_ __array[__length];

        __simd_store_unaligned<__generation, __register_policy>(__array, __vector);
        return __array[__where & (__length - 1)];
    }
}

#pragma endregion 

__SIMD_STL_NUMERIC_NAMESPACE_END

