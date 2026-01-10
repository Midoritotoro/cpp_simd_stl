#pragma once 

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

#pragma region Sse2-Sse4.2 memory access 

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>
    ::__load_upper_half(const void* __address) noexcept 
{
    return __intrin_bitcast<_VectorType_>(_mm_loadh_pd(
        __simd_broadcast_zeros<__generation, __register_policy, __m128d>(),
        static_cast<const double*>(__address)));
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>
    ::__load_lower_half(const void* __address) noexcept 
{
    return __intrin_bitcast<_VectorType_>(_mm_loadl_pd(
        __simd_broadcast_zeros<__generation, __register_policy, __m128d>(),
        static_cast<const double*>(__address)));
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>
    ::__non_temporal_load(const void* __address) noexcept
{
    return __load_aligned<_VectorType_, void>(__address);
}

template <typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>
    ::__non_temporal_store(
        void* __address,
        _VectorType_    __vector) noexcept
{
    _mm_stream_si128(static_cast<__m128i*>(__address), __intrin_bitcast<__m128i>(__vector));
}

simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::__streaming_fence() noexcept {
    return _mm_sfence();
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>
    ::__load_unaligned(const void* __address) noexcept 
{
    if      constexpr (std::is_same_v<_VectorType_, __m128i>)
        return _mm_loadu_si128(reinterpret_cast<const __m128i*>(__address));

    else if constexpr (std::is_same_v<_VectorType_, __m128d>)
        return _mm_loadu_pd(reinterpret_cast<const double*>(__address));

    else if constexpr (std::is_same_v<_VectorType_, __m128>)
        return _mm_loadu_ps(reinterpret_cast<const float*>(__address));
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>
    ::__load_aligned(const void* __address) noexcept
{
    if      constexpr (std::is_same_v<_VectorType_, __m128i>)
        return _mm_load_si128(reinterpret_cast<const __m128i*>(__address));

    else if constexpr (std::is_same_v<_VectorType_, __m128d>)
        return _mm_load_pd(reinterpret_cast<const double*>(__address));

    else if constexpr (std::is_same_v<_VectorType_, __m128>)
        return _mm_load_ps(reinterpret_cast<const float*>(__address));
}

template <typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::__store_upper_half(
    void*           __address,
    _VectorType_    __vector) noexcept
{
    _mm_storeh_pd(reinterpret_cast<double*>(__address), __intrin_bitcast<__m128d>(__vector));
}

template <typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::__store_lower_half(
    void*           __address,
    _VectorType_    __vector) noexcept
{
    _mm_storel_epi64(reinterpret_cast<__m128i*>(__address), __intrin_bitcast<__m128i>(__vector));
}

template <typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::__store_unaligned(
    void*           __address,
    _VectorType_    __vector) noexcept
{
    if      constexpr (std::is_same_v<_VectorType_, __m128i>)
        return _mm_storeu_si128(reinterpret_cast<__m128i*>(__address), __vector);

    else if constexpr (std::is_same_v<_VectorType_, __m128d>)
        return _mm_storeu_pd(reinterpret_cast<double*>(__address), __vector);

    else if constexpr (std::is_same_v<_VectorType_, __m128>)
        return _mm_storeu_ps(reinterpret_cast<float*>(__address), __vector);
}

template <typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::__store_aligned(
    void*           __address,
    _VectorType_    __vector) noexcept
{
    if      constexpr (std::is_same_v<_VectorType_, __m128i>)
        return _mm_store_si128(reinterpret_cast<__m128i*>(__address), __vector);

    else if constexpr (std::is_same_v<_VectorType_, __m128d>)
        return _mm_store_pd(reinterpret_cast<double*>(__address), __vector);

    else if constexpr (std::is_same_v<_VectorType_, __m128>)
        return _mm_store_ps(reinterpret_cast<float*>(__address), __vector);
}


template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::__mask_store_unaligned(
    void*                                   __address,
    const __simd_mask_type<_DesiredType_>    __mask,
    const _VectorType_                      __vector) noexcept
{
    __store_unaligned(__address, __simd_blend<__generation, __register_policy, _DesiredType_>(
        __vector, __load_unaligned<_VectorType_>(__address), __mask));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::__mask_store_aligned(
    void*                                   __address,
    const __simd_mask_type<_DesiredType_>    __mask,
    const _VectorType_                      __vector) noexcept
{
    __store_aligned(__address, __simd_blend<__generation, __register_policy, _DesiredType_>(
        __vector, __load_aligned<_VectorType_>(__address), __mask));
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::__mask_store_unaligned(
    void* __address,
    const _MaskVectorType_  __mask,
    const _VectorType_      __vector) noexcept
{
    __store_unaligned(__address, __simd_blend<__generation, __register_policy, _DesiredType_>(
        __vector, __load_unaligned<_VectorType_>(__address), __intrin_bitcast<_VectorType_>(__mask)));
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::__mask_store_aligned(
    void* __address,
    const _MaskVectorType_  __mask,
    const _VectorType_      __vector) noexcept
{
    __store_aligned(__address, __simd_blend<__generation, __register_policy, _DesiredType_>(
        __vector, __load_aligned<_VectorType_>(__address), __intrin_bitcast<_VectorType_>(__mask)));
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::__mask_load_unaligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>    __mask) noexcept
{
    return __simd_blend<__generation, __register_policy, _DesiredType_>(
        __load_unaligned<_VectorType_>(__address),
        __simd_broadcast_zeros<__generation, __register_policy, _VectorType_>(), __mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::__mask_load_aligned(
    const void* __address,
    const __simd_mask_type<_DesiredType_>    __mask) noexcept
{
    return __simd_blend<__generation, __register_policy, _DesiredType_>(
        __load_aligned<_VectorType_>(__address),
        __simd_broadcast_zeros<__generation, __register_policy, _VectorType_>(), __mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::__mask_load_unaligned(
    const void* __address,
    const _MaskVectorType_  __mask) noexcept
{
    return __simd_blend<__generation, __register_policy, _DesiredType_>(
        __load_unaligned<_VectorType_>(__address),
        __simd_broadcast_zeros<__generation, __register_policy, _VectorType_>(),
        __intrin_bitcast<_VectorType_>(__mask));
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::__mask_load_aligned(
    const void* __address,
    const _MaskVectorType_  __mask) noexcept
{
    return __simd_blend<__generation, __register_policy, _DesiredType_>(
        __load_aligned<_VectorType_>(__address),
        __simd_broadcast_zeros<__generation, __register_policy, _VectorType_>(),
        __intrin_bitcast<_VectorType_>(__mask));
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::__mask_load_unaligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>   __mask,
    _VectorType_                            __additional_source) noexcept
{
    return __simd_blend<__generation, __register_policy, _DesiredType_>(
        __load_unaligned<_VectorType_>(__address), __additional_source, __mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::__mask_load_aligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>   __mask,
    _VectorType_                            __additional_source) noexcept
{
    return __simd_blend<__generation, __register_policy, _DesiredType_>(
        __load_aligned<_VectorType_>(__address), __additional_source, __mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::__mask_load_unaligned(
    const void*             __address,
    const _MaskVectorType_  __mask,
    _VectorType_            __additional_source) noexcept
{
    return __simd_blend<__generation, __register_policy, _DesiredType_>(
        __load_unaligned<_VectorType_>(__address), 
        __additional_source,
        __intrin_bitcast<_VectorType_>(__mask));
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::__mask_load_aligned(
    const void*             __address,
    const _MaskVectorType_  __mask,
    _VectorType_            __additional_source) noexcept
{
    return __simd_blend<__generation, __register_policy, _DesiredType_>(
        __load_aligned<_VectorType_>(__address),
        __additional_source,
        __intrin_bitcast<_VectorType_>(__mask));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::__compress_store_unaligned(
    _DesiredType_*                      __address,
    __simd_mask_type<_DesiredType_>     __mask,
    _VectorType_                        __vector) noexcept
{
    constexpr auto __length = sizeof(_VectorType_) / sizeof(_DesiredType_);
    _DesiredType_ __source[__length];

    __store_unaligned(__source, __vector);

    auto __start = __address;

    for (auto __index = 0; __index < __length; ++__index)
        if (!((__mask >> __index) & 1))
            *__address++ = __source[__index];

    const auto __size = (__address - __start);
    std::memcpy(__address, __source + __size, sizeof(_VectorType_) - (__size * sizeof(_DesiredType_)));

    return __address;
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::__compress_store_aligned(
    _DesiredType_*                      __address,
    __simd_mask_type<_DesiredType_>     __mask,
    _VectorType_                        __vector) noexcept
{
    constexpr auto __length = sizeof(_VectorType_) / sizeof(_DesiredType_);
    _DesiredType_ __source[__length];

    __store_aligned(__source, __vector);

    auto __start = __address;

    for (auto __index = 0; __index < __length; ++__index)
        if (!((__mask >> __index) & 1))
            *__address++ = __source[__index];

    const auto __size = (__address - __start);
    std::memcpy(__address, __source + __size, sizeof(_VectorType_) - (__size * sizeof(_DesiredType_)));

    return __address;
}

template <typename _Type_>
simd_stl_always_inline __m128i __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::__make_tail_mask(uint32 __bytes) noexcept {
    constexpr unsigned int __tail_mask[8] = { ~0u, ~0u, ~0u, ~0u, 0, 0, 0, 0 };
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(
        reinterpret_cast<const unsigned char*>(__tail_mask) + (16 - __bytes)));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::SSSE3, numeric::xmm128>::__compress_store_lower_half(
    _DesiredType_*                  __address,
    __simd_mask_type<_DesiredType_> __mask,
    _VectorType_                    __vector) noexcept
{
    static_assert(sizeof(_DesiredType_) != 8);

    const auto __shuffle = __load_lower_half<__m128i>(__tables_sse<sizeof(_DesiredType_)>.__shuffle[__mask]);
    __store_lower_half(__address, _mm_shuffle_epi8(__intrin_bitcast<__m128i>(__vector), __shuffle));

    algorithm::__advance_bytes(__address, __tables_sse<sizeof(_DesiredType_)>.__size[__mask]);

    return __address;
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::SSSE3, numeric::xmm128>::__compress_store_upper_half(
    _DesiredType_*                  __address,
    __simd_mask_type<_DesiredType_> __mask,
    _VectorType_                    __vector) noexcept
{
    static_assert(sizeof(_DesiredType_) != 8);

    const auto __shuffle = __load_upper_half<__m128i>(__tables_sse<sizeof(_DesiredType_)>.__shuffle[__mask]);
    __store_upper_half(__address, _mm_shuffle_epi8(__intrin_bitcast<__m128i>(__vector), __shuffle));

    algorithm::__advance_bytes(__address, __tables_sse<sizeof(_DesiredType_)>.__size[__mask]);

    return __address;
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::SSSE3, numeric::xmm128>::__compress_store_unaligned(
    _DesiredType_*                      __address,
    __simd_mask_type<_DesiredType_>     __mask,
    _VectorType_                        __vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 1) {
        auto __start = __address;

        __address = __compress_store_lower_half(__address, __mask & 0xFF, __vector);
        __address = __compress_store_lower_half(__address, (__mask >> 8) & 0xFF, _mm_movehl_ps(
            __intrin_bitcast<__m128>(_mm_slli_si128(__intrin_bitcast<__m128i>(__vector), 8)),
            __intrin_bitcast<__m128>(__vector)));

        __mask_store_unaligned<_DesiredType_>(__start, ~((1u << (__xmm_width - (__address - __start))) - 1u), __vector);
    }
    else {
        __store_unaligned(__address, _mm_shuffle_epi8(__intrin_bitcast<__m128i>(__vector),
            __load_unaligned<__m128i>(__tables_sse<sizeof(_DesiredType_)>.__shuffle[__mask])));

        algorithm::__advance_bytes(__address, __tables_sse<sizeof(_DesiredType_)>.__size[__mask]);
    }

    return __address;
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::SSSE3, numeric::xmm128>::__compress_store_aligned(
    _DesiredType_*                      __address,
    __simd_mask_type<_DesiredType_>     __mask,
    _VectorType_                        __vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 1) {
        auto __start = __address;

        __address = __compress_store_lower_half(__address, __mask & 0xFF, __vector);
        __address = __compress_store_lower_half(__address, (__mask >> 8) & 0xFF, _mm_movehl_ps(
            __intrin_bitcast<__m128>(_mm_slli_si128(__intrin_bitcast<__m128i>(__vector), 8)),
            __intrin_bitcast<__m128>(__vector)));

        __mask_store_aligned<_DesiredType_>(__start, ~((1u << (__xmm_width - (__address - __start))) - 1u), __vector);
    }
    else {
        __store_aligned(__address, _mm_shuffle_epi8(__intrin_bitcast<__m128i>(__vector),
            __load_unaligned<__m128i>(__tables_sse<sizeof(_DesiredType_)>.__shuffle[__mask])));
    
        algorithm::__advance_bytes(__address, __tables_sse<sizeof(_DesiredType_)>.__size[__mask]);
    }

    return __address;
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE41, numeric::xmm128>::__non_temporal_load(const void* __address) noexcept {
    return __intrin_bitcast<_VectorType_>(_mm_stream_load_si128(reinterpret_cast<const __m128i*>(__address)));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::SSE41, numeric::xmm128>::__mask_store_unaligned(
    void*                                   __address,
    const __simd_mask_type<_DesiredType_>   __mask,
    const _VectorType_                      __vector) noexcept
{
    __store_unaligned(__address, __simd_blend<__generation, __register_policy, _DesiredType_>(
        __vector, __load_unaligned<_VectorType_>(__address), __mask));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::SSE41, numeric::xmm128>::__mask_store_aligned(
    void*                                   __address,
    const __simd_mask_type<_DesiredType_>   __mask,
    const _VectorType_                      __vector) noexcept
{
    __store_aligned(__address, __simd_blend<__generation, __register_policy, _DesiredType_>(
        __vector, __load_aligned<_VectorType_>(__address), __mask));
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::SSE41, numeric::xmm128>::__mask_store_unaligned(
    void*                   __address,
    const _MaskVectorType_  __mask,
    const _VectorType_      __vector) noexcept
{
    __store_unaligned(__address, __simd_blend<__generation, __register_policy, _DesiredType_>(
        __vector, __load_unaligned<_VectorType_>(__address), __intrin_bitcast<_VectorType_>(__mask)));
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::SSE41, numeric::xmm128>::__mask_store_aligned(
    void*                   __address,
    const _MaskVectorType_  __mask,
    const _VectorType_      __vector) noexcept
{
    __store_aligned(__address, __simd_blend<__generation, __register_policy, _DesiredType_>(
        __vector, __load_aligned<_VectorType_>(__address), __intrin_bitcast<_VectorType_>(__mask)));
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE41, numeric::xmm128>::__mask_load_unaligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>   __mask) noexcept
{
    return __simd_blend<__generation, __register_policy, _DesiredType_>(
        __load_unaligned<_VectorType_>(__address),
        __simd_broadcast_zeros<__generation, __register_policy, _VectorType_>(), __mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE41, numeric::xmm128>::__mask_load_aligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>   __mask) noexcept
{
    return __simd_blend<__generation, __register_policy, _DesiredType_>(
        __load_aligned<_VectorType_>(__address),
        __simd_broadcast_zeros<__generation, __register_policy, _VectorType_>(), __mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE41, numeric::xmm128>::__mask_load_unaligned(
    const void*             __address,
    const _MaskVectorType_  __mask) noexcept
{
    return __simd_blend<__generation, __register_policy, _DesiredType_>(
        __load_unaligned<_VectorType_>(__address),
        __simd_broadcast_zeros<__generation, __register_policy, _VectorType_>(),
        __intrin_bitcast<_VectorType_>(__mask));
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE41, numeric::xmm128>::__mask_load_aligned(
    const void*             __address,
    const _MaskVectorType_  __mask) noexcept
{
    return __simd_blend<__generation, __register_policy, _DesiredType_>(
        __load_aligned<_VectorType_>(__address),
        __simd_broadcast_zeros<__generation, __register_policy, _VectorType_>(),
        __intrin_bitcast<_VectorType_>(__mask));
}


template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE41, numeric::xmm128>::__mask_load_unaligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>   __mask,
    _VectorType_                            __additional_source) noexcept
{
    return __simd_blend<__generation, __register_policy, _DesiredType_>(
        __load_unaligned<_VectorType_>(__address), __additional_source, __mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE41, numeric::xmm128>::__mask_load_aligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>    __mask,
    _VectorType_                            __additional_source) noexcept
{
    return __simd_blend<__generation, __register_policy, _DesiredType_>(
        __load_aligned<_VectorType_>(__address), __additional_source, __mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE41, numeric::xmm128>::__mask_load_unaligned(
    const void*             __address,
    const _MaskVectorType_  __mask,
    _VectorType_            __additional_source) noexcept
{
    return __simd_blend<__generation, __register_policy, _DesiredType_>(
        __load_unaligned<_VectorType_>(__address), 
        __additional_source,
        __intrin_bitcast<_VectorType_>(__mask));
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE41, numeric::xmm128>::__mask_load_aligned(
    const void*             __address,
    const _MaskVectorType_  __mask,
    _VectorType_            __additional_source) noexcept
{
    return __simd_blend<__generation, __register_policy, _DesiredType_>(
        __load_aligned<_VectorType_>(__address),
        __additional_source,
        __intrin_bitcast<_VectorType_>(__mask));
}


#pragma endregion

#pragma region Avx-Avx2 memory access


template <typename _Type_>
simd_stl_always_inline auto __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::__make_tail_mask(uint32 __bytes) noexcept {
    constexpr unsigned int __tail_mask[16] = {
        ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, 0, 0, 0, 0, 0, 0, 0, 0 };
    return _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(
        reinterpret_cast<const unsigned char*>(__tail_mask) + (32 - __bytes)));
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::__load_upper_half(const void* __address) noexcept {
    return __intrin_bitcast<_VectorType_>(_mm256_insertf128_si256(_mm256_setzero_si256(),
        _mm_lddqu_si128(reinterpret_cast<const __m128i*>(__address)), 1));
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::__load_lower_half(const void* __address) noexcept {
    return __intrin_bitcast<_VectorType_>(_mm_lddqu_si128(reinterpret_cast<const __m128i*>(__address)));
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::__non_temporal_load(const void* __address) noexcept {
    return __load_aligned<_VectorType_, void>(__address);
}

template <typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::__non_temporal_store(
    void*           __address,
    _VectorType_    __vector) noexcept
{
    _mm256_stream_si256(reinterpret_cast<__m256i*>(__address), __intrin_bitcast<__m256i>(__vector));
}

simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::__streaming_fence() noexcept {
    return _mm_sfence();
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::__load_unaligned(const void* __address) noexcept {
    if      constexpr (std::is_same_v<_VectorType_, __m256i>)
        return _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(__address));

    else if constexpr (std::is_same_v<_VectorType_, __m256d>)
        return _mm256_loadu_pd(reinterpret_cast<const double*>(__address));

    else if constexpr (std::is_same_v<_VectorType_, __m256>)
        return _mm256_loadu_ps(reinterpret_cast<const float*>(__address));
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::__load_aligned(const void* __address) noexcept {
    if      constexpr (std::is_same_v<_VectorType_, __m256i>)
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(__address));

    else if constexpr (std::is_same_v<_VectorType_, __m256d>)
        return _mm256_load_pd(reinterpret_cast<const double*>(__address));

    else if constexpr (std::is_same_v<_VectorType_, __m256>)
        return _mm256_load_ps(reinterpret_cast<const float*>(__address));
}

template <typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::__store_upper_half(
    void*           __address,
    _VectorType_    __vector) noexcept
{
    _mm_storeu_si128(reinterpret_cast<__m128i*>(__address), _mm256_extractf128_si256(__intrin_bitcast<__m256i>(__vector), 1));
}

template <typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::__store_lower_half(
    void*           __address,
    _VectorType_    __vector) noexcept
{
    _mm_storeu_si128(reinterpret_cast<__m128i*>(__address), __intrin_bitcast<__m128i>(__vector));
}

template <typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::__store_unaligned(
    void*           __address,
    _VectorType_    __vector) noexcept
{
    if      constexpr (std::is_same_v<_VectorType_, __m256i>)
        return _mm256_storeu_si256(reinterpret_cast<__m256i*>(__address), __vector);

    else if constexpr (std::is_same_v<_VectorType_, __m256d>)
        return _mm256_storeu_pd(reinterpret_cast<double*>(__address), __vector);

    else if constexpr (std::is_same_v<_VectorType_, __m256>)
        return _mm256_storeu_ps(reinterpret_cast<float*>(__address), __vector);
}

template <typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::__store_aligned(
    void*           __address,
    _VectorType_    __vector) noexcept
{
    if      constexpr (std::is_same_v<_VectorType_, __m256i>)
        return _mm256_store_si256(reinterpret_cast<__m256i*>(__address), __vector);

    else if constexpr (std::is_same_v<_VectorType_, __m256d>)
        return _mm256_store_pd(reinterpret_cast<double*>(__address), __vector);

    else if constexpr (std::is_same_v<_VectorType_, __m256>)
        return _mm256_store_ps(reinterpret_cast<float*>(__address), __vector);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::__mask_store_unaligned(
    void*                                   __address,
    const __simd_mask_type<_DesiredType_>   __mask,
    const _VectorType_                      __vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        _mm256_maskstore_pd(reinterpret_cast<double*>(__address),
            __simd_to_vector<__generation, __register_policy, __m256i, _DesiredType_>(__mask),
            __intrin_bitcast<__m256d>(__vector));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        _mm256_maskstore_ps(reinterpret_cast<float*>(__address),
            __simd_to_vector<__generation, __register_policy, __m256i, _DesiredType_>(__mask),
            __intrin_bitcast<__m256>(__vector));
    }
    else {
        __store_unaligned(__address, __simd_blend<__generation, __register_policy, _DesiredType_>(
            __vector, __load_unaligned<_VectorType_>(__address), __mask));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::__mask_store_aligned(
    void*                                   __address,
    const __simd_mask_type<_DesiredType_>   __mask,
    const _VectorType_                      __vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        _mm256_maskstore_pd(reinterpret_cast<double*>(__address),
            __simd_to_vector<__generation, __register_policy, __m256i, _DesiredType_>(__mask),
            __intrin_bitcast<__m256d>(__vector));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        _mm256_maskstore_ps(reinterpret_cast<float*>(__address),
            __simd_to_vector<__generation, __register_policy, __m256i, _DesiredType_>(__mask),
            __intrin_bitcast<__m256>(__vector));
    }
    else {
        __store_aligned(__address, __simd_blend<__generation, __register_policy, _DesiredType_>(
            __vector, __load_aligned<_VectorType_>(__address), __mask));
    }
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::__mask_store_unaligned(
    void*                   __address,
    const _MaskVectorType_  __mask,
    const _VectorType_      __vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        _mm256_maskstore_pd(reinterpret_cast<double*>(__address),
            __intrin_bitcast<__m256i>(__mask), __intrin_bitcast<__m256d>(__vector));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        _mm256_maskstore_ps(reinterpret_cast<float*>(__address),
            __intrin_bitcast<__m256i>(__mask), __intrin_bitcast<__m256>(__vector));
    }
    else {
        __store_unaligned(__address, __simd_blend<__generation, __register_policy, _DesiredType_>(
            __vector, __load_unaligned<_VectorType_>(__address), __intrin_bitcast<_VectorType_>(__mask)));
    }
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::__mask_store_aligned(
    void*                   __address,
    const _MaskVectorType_  __mask,
    const _VectorType_      __vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        _mm256_maskstore_pd(reinterpret_cast<double*>(__address),
            __intrin_bitcast<__m256i>(__mask), __intrin_bitcast<__m256d>(__vector));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        _mm256_maskstore_ps(reinterpret_cast<float*>(__address),
            __intrin_bitcast<__m256i>(__mask), __intrin_bitcast<__m256>(__vector));
    }
    else {
        __store_aligned(__address, __simd_blend<__generation, __register_policy, _DesiredType_>(
            __vector, __load_aligned<_VectorType_>(__address), __intrin_bitcast<_VectorType_>(__mask)));
    }
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::__mask_load_unaligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>   __mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        return __intrin_bitcast<_VectorType_>(_mm256_maskload_pd(
            reinterpret_cast<const double*>(__address),
            __simd_to_vector<__generation, __register_policy, __m256i, _DesiredType_>(__mask)));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        return __intrin_bitcast<_VectorType_>(_mm256_maskload_ps(
            reinterpret_cast<const float*>(__address),
            __simd_to_vector<__generation, __register_policy, __m256i, _DesiredType_>(__mask)));
    }
    else {
        return __simd_blend<__generation, __register_policy, _DesiredType_>(
            __load_unaligned<_VectorType_>(__address),
            __simd_broadcast_zeros<__generation, __register_policy, _VectorType_>(), __mask);
    }
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::__mask_load_aligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>   __mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        return __intrin_bitcast<_VectorType_>(_mm256_maskload_pd(
            reinterpret_cast<const double*>(__address),
            __simd_to_vector<__generation, __register_policy, __m256i, _DesiredType_>(__mask)));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        return __intrin_bitcast<_VectorType_>(_mm256_maskload_ps(
            reinterpret_cast<const float*>(__address),
            __simd_to_vector<__generation, __register_policy, __m256i, _DesiredType_>(__mask)));
    }
    else {
        return __simd_blend<__generation, __register_policy, _DesiredType_>(
            __load_aligned<_VectorType_>(__address),
            __simd_broadcast_zeros<__generation, __register_policy, _VectorType_>(), __mask);
    }
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::__mask_load_unaligned(
    const void*             __address,
    const _MaskVectorType_  __mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        return __intrin_bitcast<_VectorType_>(_mm256_maskload_pd(
            reinterpret_cast<const double*>(__address),
            __intrin_bitcast<__m256i>(__mask)));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        return __intrin_bitcast<_VectorType_>(_mm256_maskload_ps(
            reinterpret_cast<const float*>(__address),
            __intrin_bitcast<__m256i>(__mask)));
    }
    else {
        return __simd_blend<__generation, __register_policy, _DesiredType_>(
            __load_unaligned<_VectorType_>(__address),
            __simd_broadcast_zeros<__generation, __register_policy, _VectorType_>(),
            __intrin_bitcast<_VectorType_>(__mask));
    }
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __mask_load_aligned(
    const void*             __address,
    const _MaskVectorType_  __mask) noexcept
{
    return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address, __mask);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::__compress_store_lower_half(
    _DesiredType_*                  __address,
    __simd_mask_type<_DesiredType_> __mask,
    _VectorType_                    __vector) noexcept
{
    return __simd_compress_store_unaligned<arch::CpuFeature::SSE42, xmm128, _DesiredType_>(
        __address, __mask, __intrin_bitcast<__m128i>(__vector));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::__compress_store_upper_half(
    _DesiredType_* __address,
    __simd_mask_type<_DesiredType_>  __mask,
    _VectorType_                    __vector) noexcept
{
    return __simd_compress_store_unaligned<arch::CpuFeature::SSE42, xmm128, _DesiredType_>(__address, __mask,
        _mm256_extractf128_si256(__intrin_bitcast<__m256i>(__vector), 1));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::__compress_store_unaligned(
    _DesiredType_*                      __address,
    __simd_mask_type<_DesiredType_>     __mask,
    _VectorType_                        __vector) noexcept
{
    using _MaskType = __simd_mask_type<_DesiredType_>;
    using _HalfType = IntegerForSize<_Max<(sizeof(_DesiredType_) >> 1), 1>()>::Unsigned;

    constexpr auto __maximum    = math::__maximum_integral_limit<_HalfType>();
    constexpr auto __shift      = (sizeof(_MaskType) << 2);

    __address = __compress_store_lower_half<_DesiredType_>(__address, __mask & __maximum, __vector);
    __address = __compress_store_upper_half<_DesiredType_>(__address, (__mask >> __shift) & __maximum, __vector);

    return __address;
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::__compress_store_aligned(
    _DesiredType_*                      __address,
    __simd_mask_type<_DesiredType_>     __mask,
    _VectorType_                        __vector) noexcept
{
    return __compress_store_unaligned<_DesiredType_>(__address, __mask, __vector);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX2, numeric::ymm256>::__mask_load_unaligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>   __mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        return __intrin_bitcast<_VectorType_>(_mm256_maskload_epi64(
            reinterpret_cast<const long long*>(__address),
            __simd_to_vector<__generation, __register_policy, __m256i, _DesiredType_>(__mask)));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        return __intrin_bitcast<_VectorType_>(_mm256_maskload_epi32(
            reinterpret_cast<const int*>(__address),
            __simd_to_vector<__generation, __register_policy, __m256i, _DesiredType_>(__mask)));
    }
    else {
        return __simd_blend<__generation, __register_policy, _DesiredType_>(
            __load_unaligned<_VectorType_>(__address),
            __simd_broadcast_zeros<__generation, __register_policy, _VectorType_>(), __mask);
    }
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX2, numeric::ymm256>::__mask_load_aligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>   __mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) >= 4)
        return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address, __mask);
    else
        return __simd_blend<__generation, __register_policy, _DesiredType_>(
            __load_aligned<_VectorType_>(__address),
            __simd_broadcast_zeros<__generation, __register_policy, _VectorType_>(),
            __mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX2, numeric::ymm256>::__mask_load_unaligned(
    const void*             __address,
    const _MaskVectorType_  __mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        return __intrin_bitcast<_VectorType_>(_mm256_maskload_pd(
            reinterpret_cast<const double*>(__address),
            __intrin_bitcast<__m256i>(__mask)));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        return __intrin_bitcast<_VectorType_>(_mm256_maskload_ps(
            reinterpret_cast<const float*>(__address),
            __intrin_bitcast<__m256i>(__mask)));
    }
    else {
        return __simd_blend<__generation, __register_policy, _DesiredType_>(
            __load_unaligned<_VectorType_>(__address),
            __simd_broadcast_zeros<__generation, __register_policy, _VectorType_>(),
            __intrin_bitcast<_VectorType_>(__mask));
    }
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX2, numeric::ymm256>::__mask_load_aligned(
    const void*             __address,
    const _MaskVectorType_  __mask) noexcept
{
    return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address, __mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX2, numeric::ymm256>::__mask_load_unaligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>   __mask,
    _VectorType_                            __additional_source) noexcept
{
    return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address, 
        __simd_to_vector<__generation, __register_policy, _VectorType_, _DesiredType_>(__mask), __additional_source);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX2, numeric::ymm256>::__mask_load_aligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>   __mask,
    _VectorType_                            __additional_source) noexcept
{
    return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address, __mask, __additional_source);
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX2, numeric::ymm256>::__mask_load_unaligned(
    const void*             __address,
    const _MaskVectorType_  __mask,
    _VectorType_            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        const auto __loaded = __intrin_bitcast<_VectorType_>(_mm256_maskload_pd(
            reinterpret_cast<const double*>(__address),
            __intrin_bitcast<__m256i>(__mask)));

        return __simd_blend<__generation, __register_policy, _DesiredType_>(__loaded, __additional_source, __mask);
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        const auto __loaded = __intrin_bitcast<_VectorType_>(_mm256_maskload_ps(
            reinterpret_cast<const float*>(__address),
            __intrin_bitcast<__m256i>(__mask)));

        return __simd_blend<__generation, __register_policy, _DesiredType_>(__loaded, __additional_source, __mask);
    }
    else {
        return __simd_blend<__generation, __register_policy, _DesiredType_>(
            __load_unaligned<_VectorType_>(__address),
            __additional_source,
            __intrin_bitcast<_VectorType_>(__mask));
    }
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX2, numeric::ymm256>::__mask_load_aligned(
    const void*             __address,
    const _MaskVectorType_  __mask,
    _VectorType_            __additional_source) noexcept
{
    return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address, __mask, __additional_source);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX2, numeric::ymm256>::__mask_store_unaligned(
    void* __address,
    const __simd_mask_type<_DesiredType_>    __mask,
    const _VectorType_                      __vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        _mm256_maskstore_pd(reinterpret_cast<double*>(__address),
            __simd_to_vector<__generation, __register_policy, __m256i, _DesiredType_>(__mask),
            __intrin_bitcast<__m256d>(__vector));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        _mm256_maskstore_ps(reinterpret_cast<float*>(__address),
            __simd_to_vector<__generation, __register_policy, __m256i, _DesiredType_>(__mask),
            __intrin_bitcast<__m256>(__vector));
    }
    else {
        __store_unaligned(__address, __simd_blend<__generation, __register_policy, _DesiredType_>(
            __vector, __load_unaligned<_VectorType_>(__address), __mask));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX2, numeric::ymm256>::__mask_store_aligned(
    void*                                   __address,
    const __simd_mask_type<_DesiredType_>   __mask,
    const _VectorType_                      __vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        _mm256_maskstore_pd(reinterpret_cast<double*>(__address),
            __simd_to_vector<__generation, __register_policy, __m256i, _DesiredType_>(__mask),
            __intrin_bitcast<__m256d>(__vector));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        _mm256_maskstore_ps(reinterpret_cast<float*>(__address),
            __simd_to_vector<__generation, __register_policy, __m256i, _DesiredType_>(__mask),
            __intrin_bitcast<__m256>(__vector));
    }
    else {
        __store_aligned(__address, __simd_blend<__generation, __register_policy, _DesiredType_>(
            __vector, __load_aligned<_VectorType_>(__address), __mask));
    }
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX2, numeric::ymm256>::__mask_store_unaligned(
    void*                   __address,
    const _MaskVectorType_  __mask,
    const _VectorType_      __vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        _mm256_maskstore_pd(reinterpret_cast<double*>(__address),
            __intrin_bitcast<__m256i>(__mask), __intrin_bitcast<__m256d>(__vector));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        _mm256_maskstore_ps(reinterpret_cast<float*>(__address),
            __intrin_bitcast<__m256i>(__mask), __intrin_bitcast<__m256>(__vector));
    }
    else {
        __store_unaligned(__address, __simd_blend<__generation, __register_policy, _DesiredType_>(
            __vector, __load_unaligned<_VectorType_>(__address), __intrin_bitcast<_VectorType_>(__mask)));
    }
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX2, numeric::ymm256>::__mask_store_aligned(
    void*                   __address,
    const _MaskVectorType_  __mask,
    const _VectorType_      __vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        _mm256_maskstore_pd(reinterpret_cast<double*>(__address),
            __intrin_bitcast<__m256i>(__mask), __intrin_bitcast<__m256d>(__vector));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        _mm256_maskstore_ps(reinterpret_cast<float*>(__address),
            __intrin_bitcast<__m256i>(__mask), __intrin_bitcast<__m256>(__vector));
    }
    else {
        __store_aligned(__address, __simd_blend<__generation, __register_policy, _DesiredType_>(
            __vector, __load_aligned<_VectorType_>(__address), __intrin_bitcast<_VectorType_>(__mask)));
    }
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX2, numeric::ymm256>::__non_temporal_load(const void* __address) noexcept {
    return __intrin_bitcast<_VectorType_>(_mm256_stream_load_si256(reinterpret_cast<const __m256i*>(__address)));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::AVX2, numeric::ymm256>::__compress_store_unaligned(
    _DesiredType_*                      __address,
    __simd_mask_type<_DesiredType_>     __mask,
    _VectorType_                        __vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) >= 4) {
        const auto __shuffle = _mm256_cvtepu8_epi32(_mm_loadu_si64(__tables_avx<sizeof(_DesiredType_)>.__shuffle[__mask]));

        __store_unaligned(__address, _mm256_permutevar8x32_epi32(__intrin_bitcast<__m256i>(__vector), __shuffle));
        algorithm::__advance_bytes(__address, __tables_avx<sizeof(_DesiredType_)>.__size[__mask]);
    }
    else {
        using _MaskType = __simd_mask_type<_DesiredType_>;
        using _HalfType = IntegerForSize<_Max<(sizeof(_DesiredType_) >> 1), 1>()>::Unsigned;

        constexpr auto __maximum = math::__maximum_integral_limit<_HalfType>();
        constexpr auto __shift = (sizeof(_MaskType) << 2);

        const auto __low    = __intrin_bitcast<__m128i>(__vector);
        const auto __high   = _mm256_extracti128_si256(__intrin_bitcast<__m256i>(__vector), 1);

        const auto __start = __address;

        __address = __simd_compress_store_unaligned<arch::CpuFeature::SSE42, xmm128>(__address, (__mask & __maximum), __low);
        __address = __simd_compress_store_unaligned<arch::CpuFeature::SSE42, xmm128>(__address, ((__mask >> __shift) & __maximum), __high);

        const auto __length = (__address - __start);
        const auto __store_mask = (1u << (sizeof(_VectorType_) - __length * sizeof(_DesiredType_))) - 1;

        __mask_store_unaligned(__address, __mask, __vector);
    }

    return __address;
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::AVX2, numeric::ymm256>::__compress_store_aligned(
    _DesiredType_*                      __address,
    __simd_mask_type<_DesiredType_>     __mask,
    _VectorType_                        __vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) >= 4) {
        const auto __shuffle = _mm256_cvtepu8_epi32(_mm_loadu_si64(__tables_avx<sizeof(_DesiredType_)>.__shuffle[__mask]));

        __store_aligned(__address, _mm256_permutevar8x32_epi32(__intrin_bitcast<__m256i>(__vector), __shuffle));
        algorithm::__advance_bytes(__address, __tables_avx<sizeof(_DesiredType_)>.__size[__mask]);
    }
    else {
        return __compress_store_unaligned<_DesiredType_>(__address, __mask, __vector);
    }

    return __address;
}

#pragma endregion

#pragma region Avx512 memory access

    
template <typename _Type_>
simd_stl_always_inline auto __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__make_tail_mask(uint32 __bytes) noexcept {
    const auto __elements = __bytes / sizeof(_Type_);
    return (__elements == 0) ? 0 : (static_cast<__simd_mask_type<_Type_>>((1ull << __elements) - 1));
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__load_upper_half(const void* __address) noexcept {
    return __intrin_bitcast<_VectorType_>(_mm256_lddqu_si256(reinterpret_cast<const __m256i*>(__address)));
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__load_lower_half(const void* __address) noexcept {
    return __intrin_bitcast<_VectorType_>(_mm512_inserti64x4(
        _mm512_setzero_si512(), _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(__address)), 1));
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__non_temporal_load(const void* __address) noexcept {
    return __intrin_bitcast<_VectorType_>(_mm512_stream_load_si512(__address));
}

template <typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__non_temporal_store(
    void*           __address,
    _VectorType_    __vector) noexcept
{
    _mm512_stream_si512(__address, __intrin_bitcast<__m512i>(__vector));
}

simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__streaming_fence() noexcept {
    return _mm_sfence();
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__load_unaligned(const void* __address) noexcept {
    if      constexpr (std::is_same_v<_VectorType_, __m512i>)
        return _mm512_loadu_si512(__address);

    else if constexpr (std::is_same_v<_VectorType_, __m512d>)
        return _mm512_loadu_pd(__address);

    else if constexpr (std::is_same_v<_VectorType_, __m512>)
        return _mm512_loadu_ps(__address);
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__load_aligned(const void* __address) noexcept {
    if      constexpr (std::is_same_v<_VectorType_, __m512i>)
        return _mm512_load_si512(__address);

    else if constexpr (std::is_same_v<_VectorType_, __m512d>)
        return _mm512_load_pd(__address);

    else if constexpr (std::is_same_v<_VectorType_, __m512>)
        return _mm512_load_ps(__address);
}

template <typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__store_upper_half(
    void*           __address,
    _VectorType_    __vector) noexcept
{
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(__address), _mm512_extracti64x4_epi64(__intrin_bitcast<__m512i>(__vector), 1));
}

template <typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__store_lower_half(
    void*           __address,
    _VectorType_    __vector) noexcept
{
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(__address), __intrin_bitcast<__m256d>(__vector));
}

template <typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__store_unaligned(
    void*           __address,
    _VectorType_    __vector) noexcept
{
    if      constexpr (std::is_same_v<_VectorType_, __m512i>)
        return _mm512_storeu_si512(__address, __vector);

    else if constexpr (std::is_same_v<_VectorType_, __m512d>)
        return _mm512_storeu_pd(__address, __vector);

    else if constexpr (std::is_same_v<_VectorType_, __m512>)
        return _mm512_storeu_ps(__address, __vector);
}

template <typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__store_aligned(
    void*           __address,
    _VectorType_    __vector) noexcept
{
    if      constexpr (std::is_same_v<_VectorType_, __m512i>)
        return _mm512_store_si512(__address, __vector);

    else if constexpr (std::is_same_v<_VectorType_, __m512d>)
        return _mm512_store_pd(__address, __vector);

    else if constexpr (std::is_same_v<_VectorType_, __m512>)
        return _mm512_store_ps(__address, __vector);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__mask_store_unaligned(
    void*                                   __address,
    const __simd_mask_type<_DesiredType_>   __mask,
    const _VectorType_                      __vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        _mm512_mask_storeu_epi64(__address, __mask, __intrin_bitcast<__m512i>(__vector));

    else if constexpr (sizeof(_DesiredType_) == 4)
        _mm512_mask_storeu_epi32(__address, __mask, __intrin_bitcast<__m512i>(__vector));

    else
        __store_unaligned(__address, __simd_blend<__generation, __register_policy, _DesiredType_>(
            __vector, __load_unaligned<_VectorType_>(__address), __mask));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__mask_store_aligned(
    void*                                   __address,
    const __simd_mask_type<_DesiredType_>   __mask,
    const _VectorType_                      __vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        _mm512_mask_store_epi64(__address, __mask, __intrin_bitcast<__m512i>(__vector));

    else if constexpr (sizeof(_DesiredType_) == 4)
        _mm512_mask_store_epi32(__address, __mask, __intrin_bitcast<__m512i>(__vector));

    else
        __store_aligned(__address, __simd_blend<__generation, __register_policy, _DesiredType_>(
            __vector, __load_aligned<_VectorType_>(__address), __mask));
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__mask_store_unaligned(
    void*                   __address,
    const _MaskVectorType_  __mask,
    const _VectorType_      __vector) noexcept
{
    return __mask_store_unaligned<_DesiredType_>(__address, __simd_to_mask<
        __generation, __register_policy, _DesiredType_>(__mask), __vector);
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__mask_store_aligned(
    void*                   __address,
    const _MaskVectorType_  __mask,
    const _VectorType_      __vector) noexcept
{
    return __mask_store_unaligned<_DesiredType_>(__address, __simd_to_mask<
        __generation, __register_policy, _DesiredType_>(__mask), __vector);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__mask_load_unaligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>   __mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_loadu_epi64(_mm512_setzero_si512(), __mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_loadu_epi32(_mm512_setzero_si512(), __mask, __address));

    else
        return __simd_blend<__generation, __register_policy, _DesiredType_>(
            __load_unaligned<_VectorType_>(__address),
            __simd_broadcast_zeros<__generation, __register_policy, _VectorType_>(), __mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__mask_load_aligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>   __mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_load_epi64(_mm512_setzero_si512(), __mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_load_epi32(_mm512_setzero_si512(), __mask, __address));

    else
        return __simd_blend<__generation, __register_policy, _DesiredType_>(
            __load_aligned<_VectorType_>(__address),
            __simd_broadcast_zeros<__generation, __register_policy, _VectorType_>(), __mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__mask_load_unaligned(
    const void*             __address,
    const _MaskVectorType_  __mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8 || sizeof(_DesiredType_) == 4) {
        return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address,
            __simd_to_mask<__generation, __register_policy, _DesiredType_>(__mask));
    }
    else {
        return __simd_blend<__generation, __register_policy, _DesiredType_>(
            __load_aligned<_VectorType_>(__address),
            __simd_broadcast_zeros<__generation, __register_policy, _VectorType_>(),
            __intrin_bitcast<_VectorType_>(__mask));
    }
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__mask_load_aligned(
    const void*             __address,
    const _MaskVectorType_  __mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8 || sizeof(_DesiredType_) == 4) {
        return __mask_load_aligned<_VectorType_, _DesiredType_>(__address,
            __simd_to_mask<__generation, __register_policy, _DesiredType_>(__mask));
    }
    else {
        return __simd_blend<__generation, __register_policy, _DesiredType_>(
            __load_aligned<_VectorType_>(__address),
            __simd_broadcast_zeros<__generation, __register_policy, _VectorType_>(),
            __intrin_bitcast<_VectorType_>(__mask));
    }
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__mask_load_unaligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>   __mask,
    _VectorType_                            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_loadu_epi64(
            __intrin_bitcast<__m512i>(__additional_source), __mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_loadu_epi32(
            __intrin_bitcast<__m512i>(__additional_source), __mask, __address));

    else
        return __simd_blend<__generation, __register_policy, _DesiredType_>(
            __load_unaligned<_VectorType_>(__address), __additional_source, __mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__mask_load_aligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>   __mask,
    _VectorType_                            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_load_epi64(
            __intrin_bitcast<__m512i>(__additional_source), __mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_load_epi32(
            __intrin_bitcast<__m512i>(__additional_source), __mask, __address));

    else
        return __simd_blend<__generation, __register_policy, _DesiredType_>(
            __load_aligned<_VectorType_>(__address), __additional_source, __mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__mask_load_unaligned(
    const void*             __address,
    const _MaskVectorType_  __mask,
    _VectorType_            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) >= 4)
        return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address,
            __simd_to_mask<__generation, __register_policy, _DesiredType_>(__mask), __additional_source);
    else
        return __simd_blend<__generation, __register_policy, _DesiredType_>(
            __load_unaligned<_VectorType_>(__address),
            __additional_source,
            __intrin_bitcast<_VectorType_>(__mask));
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__mask_load_aligned(
    const void*             __address,
    const _MaskVectorType_  __mask,
    _VectorType_            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) >= 4)
        return __mask_load_aligned<_VectorType_, _DesiredType_>(__address,
            __simd_to_mask<__generation, __register_policy, _DesiredType_>(__mask), __additional_source);
    else
        return __simd_blend<__generation, __register_policy, _DesiredType_>(
            __load_aligned<_VectorType_>(__address),
            __additional_source,
            __intrin_bitcast<_VectorType_>(__mask));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__compress_store_unaligned(
    _DesiredType_*                      __address,
    __simd_mask_type<_DesiredType_>     __mask,
    _VectorType_                        __vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        const auto __not = uint8(~__mask);

        const auto __compressed = _mm512_mask_compress_epi64(
            __intrin_bitcast<__m512i>(__vector), __not, __intrin_bitcast<__m512i>(__vector));
        _mm512_storeu_si512(__address, __compressed);

        algorithm::__advance_bytes(__address, (math::population_count(__not) << 3));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        const auto __not = uint16(~__mask);

        const auto __compressed = _mm512_mask_compress_epi32(
            __intrin_bitcast<__m512i>(__vector), __not, __intrin_bitcast<__m512i>(__vector));
        _mm512_storeu_si512(__address, __compressed);

        algorithm::__advance_bytes(__address, (math::population_count(__not) << 2));
    }
    else {
       
    }

    return __address;
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__compress_store_aligned(
    _DesiredType_*                      __address,
    __simd_mask_type<_DesiredType_>     __mask,
    _VectorType_                        __vector) noexcept
{
    return __compress_store_unaligned<_DesiredType_>(__address, __mask, __vector);
}


template <typename _Type_>
simd_stl_always_inline auto __simd_memory_access<arch::CpuFeature::AVX512BW, numeric::zmm512>::__make_tail_mask(uint32 __bytes) noexcept {
    const auto __elements = __bytes / sizeof(_Type_);
    return (__elements == 0) ? 0 : (static_cast<__simd_mask_type<_Type_>>((1ull << __elements) - 1));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512BW, numeric::zmm512>::__mask_store_unaligned(
    void*                                   __address,
    const __simd_mask_type<_DesiredType_>   __mask,
    const _VectorType_                      __vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        _mm512_mask_storeu_epi64(__address, __mask, __intrin_bitcast<__m512i>(__vector));

    else if constexpr (sizeof(_DesiredType_) == 4)
        _mm512_mask_storeu_epi32(__address, __mask, __intrin_bitcast<__m512i>(__vector));

    else if constexpr (sizeof(_DesiredType_) == 2)
        _mm512_mask_storeu_epi16(__address, __mask, __intrin_bitcast<__m512i>(__vector));

    else if constexpr (sizeof(_DesiredType_) == 1)
        _mm512_mask_storeu_epi8(__address, __mask, __intrin_bitcast<__m512i>(__vector));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512BW, numeric::zmm512>::__mask_store_aligned(
    void*                                   __address,
    const __simd_mask_type<_DesiredType_>   __mask,
    const _VectorType_                      __vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        _mm512_mask_store_epi64(__address, __mask, __intrin_bitcast<__m512i>(__vector));

    else if constexpr (sizeof(_DesiredType_) == 4)
        _mm512_mask_store_epi32(__address, __mask, __intrin_bitcast<__m512i>(__vector));

    else
        return __mask_store_unaligned<_DesiredType_>(__address, __mask, __vector);
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512BW, numeric::zmm512>::__mask_store_unaligned(
    void*                   __address,
    const _MaskVectorType_  __mask,
    const _VectorType_      __vector) noexcept
{
    return __mask_store_unaligned<_DesiredType_>(__address, __simd_to_mask<
        __generation, __register_policy, _DesiredType_>(__mask), __vector);
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512BW, numeric::zmm512>::__mask_store_aligned(
    void*                   __address,
    const _MaskVectorType_  __mask,
    const _VectorType_      __vector) noexcept
{
    return __mask_store_unaligned<_DesiredType_>(__address, __simd_to_mask<
        __generation, __register_policy, _DesiredType_>(__mask), __vector);
}


template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512BW, numeric::zmm512>::__mask_load_unaligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>   __mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_loadu_epi64(_mm512_setzero_si512(), __mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_loadu_epi32(_mm512_setzero_si512(), __mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 2)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_loadu_epi16(_mm512_setzero_si512(), __mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 1)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_loadu_epi8(_mm512_setzero_si512(), __mask, __address));
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512BW, numeric::zmm512>::__mask_load_aligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>   __mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_load_epi64(_mm512_setzero_si512(), __mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_load_epi32(_mm512_setzero_si512(), __mask, __address));

    else
        return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address, __mask);
}


template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512BW, numeric::zmm512>::__mask_load_unaligned(
    const void*             __address,
    const _MaskVectorType_  __mask) noexcept
{
    return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address,
        __simd_to_mask<__generation, __register_policy, _DesiredType_>(__mask));
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512BW, numeric::zmm512>::__mask_load_aligned(
    const void*             __address,
    const _MaskVectorType_  __mask) noexcept
{
    return __mask_load_aligned<_VectorType_, _DesiredType_>(__address,
        __simd_to_mask<__generation, __register_policy, _DesiredType_>(__mask));
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512BW, numeric::zmm512>::__mask_load_unaligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>   __mask,
    _VectorType_                            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_loadu_epi64(__intrin_bitcast<__m512i>(__additional_source), __mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_loadu_epi32(__intrin_bitcast<__m512i>(__additional_source), __mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 2)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_loadu_epi16(__intrin_bitcast<__m512i>(__additional_source), __mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 1)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_loadu_epi8(__intrin_bitcast<__m512i>(__additional_source), __mask, __address));
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512BW, numeric::zmm512>::__mask_load_aligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>   __mask,
    _VectorType_                            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_load_epi64(__intrin_bitcast<__m512i>(__additional_source), __mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_load_epi32(__intrin_bitcast<__m512i>(__additional_source), __mask, __address));

    else
        return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address, __mask, __additional_source);
}


template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512BW, numeric::zmm512>::__mask_load_unaligned(
    const void*             __address,
    const _MaskVectorType_  __mask,
    _VectorType_            __additional_source) noexcept
{
    return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address,
        __simd_to_mask<__generation, __register_policy, _DesiredType_>(__mask), __additional_source);
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512BW, numeric::zmm512>::__mask_load_aligned(
    const void*             __address,
    const _MaskVectorType_  __mask,
    _VectorType_            __additional_source) noexcept
{
    return __mask_load_aligned<_VectorType_, _DesiredType_>(__address,
        __simd_to_mask<__generation, __register_policy, _DesiredType_>(__mask), __additional_source);
}

template <typename _Type_>
simd_stl_always_inline auto __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::ymm256>::__make_tail_mask(uint32 _Bytes) noexcept {
    const auto __elements = _Bytes / sizeof(_Type_);
    return (__elements == 0) ? 0 : (static_cast<__simd_mask_type<_Type_>>((1ull << __elements) - 1));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::ymm256>::__mask_store_unaligned(
    void*                                   __address,
    const __simd_mask_type<_DesiredType_>   __mask,
    const _VectorType_                      __vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        _mm256_mask_storeu_epi64(__address, __mask, __intrin_bitcast<__m256i>(__vector));

    else if constexpr (sizeof(_DesiredType_) == 4)
        _mm256_mask_storeu_epi32(__address, __mask, __intrin_bitcast<__m256i>(__vector));

    else
        __store_unaligned(__address, __simd_blend<__generation, __register_policy, _DesiredType_>(
            __vector, __load_unaligned<_VectorType_>(__address), __mask));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::ymm256>::__mask_store_aligned(
    void*                                   __address,
    const __simd_mask_type<_DesiredType_>   __mask,
    const _VectorType_                      __vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        _mm256_mask_store_epi64(__address, __mask, __intrin_bitcast<__m256i>(__vector));

    else if constexpr (sizeof(_DesiredType_) == 4)
        _mm256_mask_store_epi32(__address, __mask, __intrin_bitcast<__m256i>(__vector));

    else
        __store_aligned(__address, __simd_blend<__generation, __register_policy, _DesiredType_>(
            __vector, __load_aligned<_VectorType_>(__address), __mask));
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::ymm256>::__mask_store_unaligned(
    void*                   __address,
    const _MaskVectorType_  __mask,
    const _VectorType_      __vector) noexcept
{
    return __mask_store_unaligned<_DesiredType_>(__address, __simd_to_mask<
        __generation, __register_policy, _DesiredType_>(__mask), __vector);
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::ymm256>::__mask_store_aligned(
    void*                   __address,
    const _MaskVectorType_  __mask,
    const _VectorType_      __vector) noexcept
{
    return __mask_store_unaligned<_DesiredType_>(__address, __simd_to_mask<
        __generation, __register_policy, _DesiredType_>(__mask), __vector);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::ymm256>::__mask_load_unaligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>   __mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_loadu_epi64(_mm256_setzero_si256(), __mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_loadu_epi32(_mm256_setzero_si256(), __mask, __address));

    else
        return __simd_blend<__generation, __register_policy, _DesiredType_>(
            __load_unaligned<_VectorType_>(__address),
            __simd_broadcast_zeros<__generation, __register_policy, _VectorType_>(), __mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::ymm256>::__mask_load_aligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>   __mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_load_epi64(_mm256_setzero_si256(), __mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_load_epi32(_mm256_setzero_si256(), __mask, __address));

    else
        return __simd_blend<__generation, __register_policy, _DesiredType_>(
            __load_aligned<_VectorType_>(__address),
            __simd_broadcast_zeros<__generation, __register_policy, _VectorType_>(), __mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::ymm256>::__mask_load_unaligned(
    const void*             __address,
    const _MaskVectorType_  __mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8 || sizeof(_DesiredType_) == 4) {
        return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address,
            __simd_to_mask<__generation, __register_policy, _DesiredType_>(__mask));
    }
    else {
        return __simd_blend<__generation, __register_policy, _DesiredType_>(
            __load_aligned<_VectorType_>(__address),
            __simd_broadcast_zeros<__generation, __register_policy, _VectorType_>(),
            __intrin_bitcast<_VectorType_>(__mask));
    }
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::ymm256>::__mask_load_aligned(
    const void*             __address,
    const _MaskVectorType_  __mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8 || sizeof(_DesiredType_) == 4) {
        return __mask_load_aligned<_VectorType_, _DesiredType_>(__address,
            __simd_to_mask<__generation, __register_policy, _DesiredType_>(__mask));
    }
    else {
        return __simd_blend<__generation, __register_policy, _DesiredType_>(
            __load_aligned<_VectorType_>(__address),
            __simd_broadcast_zeros<__generation, __register_policy, _VectorType_>(),
            __intrin_bitcast<_VectorType_>(__mask));
    }
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::ymm256>::__mask_load_unaligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>   __mask,
    _VectorType_                            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_loadu_epi64(__intrin_bitcast<__m256i>(__additional_source), __mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_loadu_epi32(__intrin_bitcast<__m256i>(__additional_source), __mask, __address));

    else
        return __simd_blend<__generation, __register_policy, _DesiredType_>(
            __load_unaligned<_VectorType_>(__address), __additional_source, __mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::ymm256>::__mask_load_aligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>   __mask,
    _VectorType_                            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_load_epi64(__intrin_bitcast<__m256i>(__additional_source), __mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_load_epi32(__intrin_bitcast<__m256i>(__additional_source), __mask, __address));

    else
        return __simd_blend<__generation, __register_policy, _DesiredType_>(
            __load_aligned<_VectorType_>(__address), __additional_source, __mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::ymm256>::__mask_load_unaligned(
    const void*             __address,
    const _MaskVectorType_  __mask,
    _VectorType_            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) >= 4)
        return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address,
            __simd_to_mask<__generation, __register_policy, _DesiredType_>(__mask), __additional_source);
    else
        return __simd_blend<__generation, __register_policy, _DesiredType_>(
            __load_unaligned<_VectorType_>(__address),
            __additional_source,
            __intrin_bitcast<_VectorType_>(__mask));
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::ymm256>::__mask_load_aligned(
    const void*             __address,
    const _MaskVectorType_  __mask,
    _VectorType_            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) >= 4)
        return __mask_load_aligned<_VectorType_, _DesiredType_>(__address,
            __simd_to_mask<__generation, __register_policy, _DesiredType_>(__mask), __additional_source);
    else
        return __simd_blend<__generation, __register_policy, _DesiredType_>(
            __load_aligned<_VectorType_>(__address),
            __additional_source,
            __intrin_bitcast<_VectorType_>(__mask));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::ymm256>::__compress_store_unaligned(
    _DesiredType_*                      __address,
    __simd_mask_type<_DesiredType_>     __mask,
    _VectorType_                        __vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        const auto __not = uint8(uint8(0xF) & uint8(~__mask));

        const auto __compressed = _mm256_mask_compress_epi64(
            __intrin_bitcast<__m256i>(__vector), __not, __intrin_bitcast<__m256i>(__vector));
        __store_unaligned(__address, __compressed);

        algorithm::__advance_bytes(__address, (math::population_count(__not) << 3));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        const auto __not = uint16(uint16(0xFF) & uint16(~__mask));

        const auto __compressed = _mm256_mask_compress_epi32(
            __intrin_bitcast<__m256i>(__vector), __not, __intrin_bitcast<__m256i>(__vector));
        __store_unaligned(__address, __compressed);

        algorithm::__advance_bytes(__address, (math::population_count(__not) << 2));
    }
    else {
        constexpr auto __length = sizeof(_VectorType_) / sizeof(_DesiredType_);
        _DesiredType_ __source[__length];

        __store_unaligned(__source, __vector);
        auto __start = __address;

        for (auto __index = 0; __index < __length; ++__index)
            if (!((__mask >> __index) & 1))
                *__address++ = __source[__index];

        const auto __bytes = (__address - __start);
        std::memcpy(__address, __source + __bytes, sizeof(_VectorType_) - __bytes);
    }

    return __address;
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::ymm256>::__compress_store_aligned(
    _DesiredType_*                      __address,
    __simd_mask_type<_DesiredType_>     __mask,
    _VectorType_                        __vector) noexcept
{
    return __compress_store_unaligned<_DesiredType_>(__address, __mask, __vector);
}


template <typename _Type_>
simd_stl_always_inline auto __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::ymm256>::__make_tail_mask(uint32 __bytes) noexcept {
    const auto __elements = __bytes / sizeof(_Type_);
    return (__elements == 0) ? 0 : (static_cast<__simd_mask_type<_Type_>>((1ull << __elements) - 1));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::ymm256>::__mask_store_unaligned(
    void*                                   __address,
    const __simd_mask_type<_DesiredType_>   __mask,
    const _VectorType_                      __vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        _mm256_mask_storeu_epi64(__address, __mask, __intrin_bitcast<__m256i>(__vector));

    else if constexpr (sizeof(_DesiredType_) == 4)
        _mm256_mask_storeu_epi32(__address, __mask, __intrin_bitcast<__m256i>(__vector));

    else if constexpr (sizeof(_DesiredType_) == 2)
        _mm256_mask_storeu_epi16(__address, __mask, __intrin_bitcast<__m256i>(__vector));

    else if constexpr (sizeof(_DesiredType_) == 1)
        _mm256_mask_storeu_epi8(__address, __mask, __intrin_bitcast<__m256i>(__vector));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::ymm256>::__mask_store_aligned(
    void*                                   __address,
    const __simd_mask_type<_DesiredType_>   __mask,
    const _VectorType_                      __vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        _mm256_mask_store_epi64(__address, __mask, __intrin_bitcast<__m256i>(__vector));

    else if constexpr (sizeof(_DesiredType_) == 4)
        _mm256_mask_store_epi32(__address, __mask, __intrin_bitcast<__m256i>(__vector));

    else
        return __mask_store_unaligned<_DesiredType_>(__address, __mask, __vector);
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::ymm256>::__mask_store_unaligned(
    void*                   __address,
    const _MaskVectorType_  __mask,
    const _VectorType_      __vector) noexcept
{
    return __mask_store_unaligned<_DesiredType_>(__address, __simd_to_mask<
        __generation, __register_policy, _DesiredType_>(__mask), __vector);
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::ymm256>::__mask_store_aligned(
    void*                   __address,
    const _MaskVectorType_  __mask,
    const _VectorType_      __vector) noexcept
{
    return __mask_store_unaligned<_DesiredType_>(__address, __simd_to_mask<
        __generation, __register_policy, _DesiredType_>(__mask), __vector);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::ymm256>::__mask_load_unaligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>   __mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_loadu_epi64(_mm256_setzero_si256(), __mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_loadu_epi32(_mm256_setzero_si256(), __mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 2)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_loadu_epi16(_mm256_setzero_si256(), __mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 1)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_loadu_epi8(_mm256_setzero_si256(), __mask, __address));
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::ymm256>::__mask_load_aligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>   __mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_load_epi64(_mm256_setzero_si256(), __mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_load_epi32(_mm256_setzero_si256(), __mask, __address));

    else
        return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address, __mask);
}


template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::ymm256>::__mask_load_unaligned(
    const void*             __address,
    const _MaskVectorType_  __mask) noexcept
{
    return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address,
        __simd_to_mask<__generation, __register_policy, _DesiredType_>(__mask));
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::ymm256>::__mask_load_aligned(
    const void*             __address,
    const _MaskVectorType_  __mask) noexcept
{
    return __mask_load_aligned<_VectorType_, _DesiredType_>(__address,
        __simd_to_mask<__generation, __register_policy, _DesiredType_>(__mask));
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::ymm256>::__mask_load_unaligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>   __mask,
    _VectorType_                            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_loadu_epi64(
            __intrin_bitcast<__m256i>(__additional_source), __mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_loadu_epi32(
            __intrin_bitcast<__m256i>(__additional_source), __mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 2)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_loadu_epi16(
            __intrin_bitcast<__m256i>(__additional_source), __mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 1)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_loadu_epi8(
            __intrin_bitcast<__m256i>(__additional_source), __mask, __address));
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::ymm256>::__mask_load_aligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>   __mask,
    _VectorType_                            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_load_epi64(
            __intrin_bitcast<__m256i>(__additional_source), __mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_load_epi32(
            __intrin_bitcast<__m256i>(__additional_source), __mask, __address));

    else
        return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address, __mask, __additional_source);
}


template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::ymm256>::__mask_load_unaligned(
    const void*             __address,
    const _MaskVectorType_  __mask,
    _VectorType_            __additional_source) noexcept
{
    return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address,
        __simd_to_mask<__generation, __register_policy, _DesiredType_>(__mask), __additional_source);
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::ymm256>::__mask_load_aligned(
    const void*             __address,
    const _MaskVectorType_  __mask,
    _VectorType_            __additional_source) noexcept
{
    return __mask_load_aligned<_VectorType_, _DesiredType_>(__address,
        __simd_to_mask<__generation, __register_policy, _DesiredType_>(__mask), __additional_source);
}


template <typename _Type_>
simd_stl_always_inline auto __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::xmm128>::__make_tail_mask(uint32 __bytes) noexcept {
    const auto __elements = __bytes / sizeof(_Type_);
    return (__elements == 0) ? 0 : (static_cast<__simd_mask_type<_Type_>>((1ull << __elements) - 1));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::xmm128>::__mask_store_unaligned(
    void*                                   __address,
    const __simd_mask_type<_DesiredType_>   __mask,
    const _VectorType_                      __vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        _mm_mask_storeu_epi64(__address, __mask, __intrin_bitcast<__m128i>(__vector));

    else if constexpr (sizeof(_DesiredType_) == 4)
        _mm_mask_storeu_epi32(__address, __mask, __intrin_bitcast<__m128i>(__vector));

    else
        __store_unaligned(__address, __simd_blend<__generation, __register_policy, _DesiredType_>(
            __vector, __load_unaligned<_VectorType_>(__address), __mask));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::xmm128>::__mask_store_aligned(
    void*                                   __address,
    const __simd_mask_type<_DesiredType_>   __mask,
    const _VectorType_                      __vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        _mm_mask_store_epi64(__address, __mask, __intrin_bitcast<__m128i>(__vector));

    else if constexpr (sizeof(_DesiredType_) == 4)
        _mm_mask_store_epi32(__address, __mask, __intrin_bitcast<__m128i>(__vector));

    else
        __store_aligned(__address, __simd_blend<__generation, __register_policy, _DesiredType_>(
            __vector, __load_aligned<_VectorType_>(__address), __mask));
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::xmm128>::__mask_store_unaligned(
    void*                   __address,
    const _MaskVectorType_  __mask,
    const _VectorType_      __vector) noexcept
{
    return __mask_store_unaligned<_DesiredType_>(__address, __simd_to_mask<
        __generation, __register_policy, _DesiredType_>(__mask), __vector);
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::xmm128>::__mask_store_aligned(
    void*                   __address,
    const _MaskVectorType_  __mask,
    const _VectorType_      __vector) noexcept
{
    return __mask_store_unaligned<_DesiredType_>(__address, __simd_to_mask<
        __generation, __register_policy, _DesiredType_>(__mask), __vector);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::xmm128>::__mask_load_unaligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>   __mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm_mask_loadu_epi64(_mm_setzero_si128(), __mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm_mask_loadu_epi32(_mm_setzero_si128(), __mask, __address));

    else
        return __simd_blend<__generation, __register_policy, _DesiredType_>(
            __load_unaligned<_VectorType_>(__address),
            __simd_broadcast_zeros<__generation, __register_policy, _VectorType_>(), __mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::xmm128>::__mask_load_aligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>   __mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm_mask_load_epi64(_mm_setzero_si128(), __mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm_mask_load_epi32(_mm_setzero_si128(), __mask, __address));

    else
        return __simd_blend<__generation, __register_policy, _DesiredType_>(
            __load_aligned<_VectorType_>(__address),
            __simd_broadcast_zeros<__generation, __register_policy, _VectorType_>(), __mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::xmm128>::__mask_load_unaligned(
    const void*             __address,
    const _MaskVectorType_  __mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8 || sizeof(_DesiredType_) == 4) {
        return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address,
            __simd_to_mask<__generation, __register_policy, _DesiredType_>(__mask));
    }
    else {
        return __simd_blend<__generation, __register_policy, _DesiredType_>(
            __load_aligned<_VectorType_>(__address),
            __simd_broadcast_zeros<__generation, __register_policy, _VectorType_>(),
            __intrin_bitcast<_VectorType_>(__mask));
    }
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::xmm128>::__mask_load_aligned(
    const void*             __address,
    const _MaskVectorType_  __mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8 || sizeof(_DesiredType_) == 4) {
        return __mask_load_aligned<_VectorType_, _DesiredType_>(__address,
            __simd_to_mask<__generation, __register_policy, _DesiredType_>(__mask));
    }
    else {
        return __simd_blend<__generation, __register_policy, _DesiredType_>(
            __load_aligned<_VectorType_>(__address),
            __simd_broadcast_zeros<__generation, __register_policy, _VectorType_>(),
            __intrin_bitcast<_VectorType_>(__mask));
    }
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::xmm128>::__mask_load_unaligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>   __mask,
    _VectorType_                            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm_mask_loadu_epi64(__intrin_bitcast<__m128i>(__additional_source), __mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm_mask_loadu_epi32(__intrin_bitcast<__m128i>(__additional_source), __mask, __address));

    else
        return __simd_blend<__generation, __register_policy, _DesiredType_>(
            __load_unaligned<_VectorType_>(__address), __additional_source, __mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::xmm128>::__mask_load_aligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>   __mask,
    _VectorType_                            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm_mask_load_epi64(__intrin_bitcast<__m128i>(__additional_source), __mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm_mask_load_epi32(__intrin_bitcast<__m128i>(__additional_source), __mask, __address));

    else
        return __simd_blend<__generation, __register_policy, _DesiredType_>(
            __load_aligned<_VectorType_>(__address), __additional_source, __mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::xmm128>::__mask_load_unaligned(
    const void*             __address,
    const _MaskVectorType_  __mask,
    _VectorType_            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) >= 4)
        return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address,
            __simd_to_mask<__generation, __register_policy, _DesiredType_>(__mask), __additional_source);
    else
        return __simd_blend<__generation, __register_policy, _DesiredType_>(
            __load_unaligned<_VectorType_>(__address),
            __additional_source,
            __intrin_bitcast<_VectorType_>(__mask));
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::xmm128>::__mask_load_aligned(
    const void*             __address,
    const _MaskVectorType_  __mask,
    _VectorType_            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) >= 4)
        return __mask_load_aligned<_VectorType_, _DesiredType_>(__address,
            __simd_to_mask<__generation, __register_policy, _DesiredType_>(__mask), __additional_source);
    else
        return __simd_blend<__generation, __register_policy, _DesiredType_>(
            __load_aligned<_VectorType_>(__address),
            __additional_source,
            __intrin_bitcast<_VectorType_>(__mask));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::xmm128>::__compress_store_unaligned(
    _DesiredType_*                      __address,
    __simd_mask_type<_DesiredType_>     __mask,
    _VectorType_                        __vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        const auto __not = uint8(uint8(0x03) & uint8(~__mask));

        const auto __compressed = _mm_mask_compress_epi64(
            __intrin_bitcast<__m128i>(__vector), __not, __intrin_bitcast<__m128i>(__vector));
        __store_unaligned(__address, __compressed);

        algorithm::__advance_bytes(__address, (math::population_count(__not) << 3));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        const auto __not = uint8(uint8(0xF) & uint8(~__mask));

        const auto __compressed = _mm_mask_compress_epi32(
            __intrin_bitcast<__m128i>(__vector), __not, __intrin_bitcast<__m128i>(__vector));
        __store_unaligned(__address, __compressed);

        algorithm::__advance_bytes(__address, (math::population_count(__not) << 2));
    }
    else {
        constexpr auto __length = sizeof(_VectorType_) / sizeof(_DesiredType_);
        _DesiredType_ __source[__length];

        __store_unaligned(__source, __vector);
        auto __start = __address;

        for (auto __index = 0; __index < __length; ++__index)
            if (!((__mask >> __index) & 1))
                *__address++ = __source[__index];

        const auto __bytes = (__address - __start);
        std::memcpy(__address, __source + __bytes, sizeof(_VectorType_) - __bytes);
    }

    return __address;
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::xmm128>::__compress_store_aligned(
    _DesiredType_*                      __address,
    __simd_mask_type<_DesiredType_>     __mask,
    _VectorType_                        __vector) noexcept
{
    return __compress_store_unaligned<_DesiredType_>(__address, __mask, __vector);
}

template <typename _Type_>
simd_stl_always_inline auto __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::xmm128>::__make_tail_mask(uint32 __bytes) noexcept {
    const auto __elements = __bytes / sizeof(_Type_);
    return (__elements == 0) ? 0 : (static_cast<__simd_mask_type<_Type_>>((1ull << __elements) - 1));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::xmm128>::__mask_store_unaligned(
    void*                                   __address,
    const __simd_mask_type<_DesiredType_>   __mask,
    const _VectorType_                      __vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        _mm_mask_storeu_epi64(__address, __mask, __intrin_bitcast<__m128i>(__vector));

    else if constexpr (sizeof(_DesiredType_) == 4)
        _mm_mask_storeu_epi32(__address, __mask, __intrin_bitcast<__m128i>(__vector));

    else if constexpr (sizeof(_DesiredType_) == 2)
        _mm_mask_storeu_epi16(__address, __mask, __intrin_bitcast<__m128i>(__vector));

    else if constexpr (sizeof(_DesiredType_) == 1)
        _mm_mask_storeu_epi8(__address, __mask, __intrin_bitcast<__m128i>(__vector));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::xmm128>::__mask_store_aligned(
    void*                                   __address,
    const __simd_mask_type<_DesiredType_>   __mask,
    const _VectorType_                      __vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        _mm_mask_store_epi64(__address, __mask, __intrin_bitcast<__m128i>(__vector));

    else if constexpr (sizeof(_DesiredType_) == 4)
        _mm_mask_store_epi32(__address, __mask, __intrin_bitcast<__m128i>(__vector));

    else
        return __mask_store_unaligned<_DesiredType_>(__address, __mask, __vector);
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::xmm128>::__mask_store_unaligned(
    void*                   __address,
    const _MaskVectorType_  __mask,
    const _VectorType_      __vector) noexcept
{
    return __mask_store_unaligned<_DesiredType_>(__address, __simd_to_mask<
        __generation, __register_policy, _DesiredType_>(__mask), __vector);
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::xmm128>::__mask_store_aligned(
    void*                   __address,
    const _MaskVectorType_  __mask,
    const _VectorType_      __vector) noexcept
{
    return __mask_store_unaligned<_DesiredType_>(__address, __simd_to_mask<
        __generation, __register_policy, _DesiredType_>(__mask), __vector);
}


template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::xmm128>::__mask_load_unaligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>   __mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm_mask_loadu_epi64(_mm_setzero_si128(), __mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm_mask_loadu_epi32(_mm_setzero_si128(), __mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 2)
        return __intrin_bitcast<_VectorType_>(_mm_mask_loadu_epi16(_mm_setzero_si128(), __mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 1)
        return __intrin_bitcast<_VectorType_>(_mm_mask_loadu_epi8(_mm_setzero_si128(), __mask, __address));
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::xmm128>::__mask_load_aligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>   __mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm_mask_load_epi64(_mm_setzero_si128(), __mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm_mask_load_epi32(_mm_setzero_si128(), __mask, __address));

    else
        return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address, __mask);
}


template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::xmm128>::__mask_load_unaligned(
    const void*             __address,
    const _MaskVectorType_  __mask) noexcept
{
    return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address,
        __simd_to_mask<__generation, __register_policy, _DesiredType_>(__mask));
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::xmm128>::__mask_load_aligned(
    const void*             __address,
    const _MaskVectorType_  __mask) noexcept
{
    return __mask_load_aligned<_VectorType_, _DesiredType_>(__address,
        __simd_to_mask<__generation, __register_policy, _DesiredType_>(__mask));
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::xmm128>::__mask_load_unaligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>   __mask,
    _VectorType_                            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm_mask_loadu_epi64(
            __intrin_bitcast<__m128i>(__additional_source), __mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm_mask_loadu_epi32(
            __intrin_bitcast<__m128i>(__additional_source), __mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 2)
        return __intrin_bitcast<_VectorType_>(_mm_mask_loadu_epi16(
            __intrin_bitcast<__m128i>(__additional_source), __mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 1)
        return __intrin_bitcast<_VectorType_>(_mm_mask_loadu_epi8(
            __intrin_bitcast<__m128i>(__additional_source), __mask, __address));
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::xmm128>::__mask_load_aligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>   __mask,
    _VectorType_                            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm_mask_load_epi64(
            __intrin_bitcast<__m128i>(__additional_source), __mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm_mask_load_epi32(
            __intrin_bitcast<__m128i>(__additional_source), __mask, __address));

    else
        return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address, __mask, __additional_source);
}


template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::xmm128>::__mask_load_unaligned(
    const void*             __address,
    const _MaskVectorType_  __mask,
    _VectorType_            __additional_source) noexcept
{
    return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address,
        __simd_to_mask<__generation, __register_policy, _DesiredType_>(__mask), __additional_source);
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::xmm128>::__mask_load_aligned(
    const void*             __address,
    const _MaskVectorType_  __mask,
    _VectorType_            __additional_source) noexcept
{
    return __mask_load_aligned<_VectorType_, _DesiredType_>(__address,
        __simd_to_mask<__generation, __register_policy, _DesiredType_>(__mask), __additional_source);
}

#pragma endregion

__SIMD_STL_NUMERIC_NAMESPACE_END
