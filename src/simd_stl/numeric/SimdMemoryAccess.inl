#pragma once 

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

#pragma region Sse2-Sse4.2 memory access 

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>
    ::_LoadUpperHalf(const void* __address) noexcept 
{
    return __intrin_bitcast<_VectorType_>(_mm_loadh_pd(
        _SimdBroadcastZeros<__generation, __register_policy, __m128d>(),
        static_cast<const double*>(__address)));
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>
    ::_LoadLowerHalf(const void* __address) noexcept 
{
    return __intrin_bitcast<_VectorType_>(_mm_loadl_pd(
        _SimdBroadcastZeros<__generation, __register_policy, __m128d>(),
        static_cast<const double*>(__address)));
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>
    ::_NonTemporalLoad(const void* __address) noexcept
{
    return _LoadAligned<_VectorType_, void>(__address);
}

template <typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>
    ::_NonTemporalStore(
        void* __address,
        _VectorType_    _Vector) noexcept
{
    _mm_stream_si128(static_cast<__m128i*>(__address), __intrin_bitcast<__m128i>(_Vector));
}

simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::_StreamingFence() noexcept {
    return _mm_sfence();
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>
    ::_LoadUnaligned(const void* __address) noexcept 
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
    ::_LoadAligned(const void* __address) noexcept
{
    if      constexpr (std::is_same_v<_VectorType_, __m128i>)
        return _mm_load_si128(reinterpret_cast<const __m128i*>(__address));

    else if constexpr (std::is_same_v<_VectorType_, __m128d>)
        return _mm_load_pd(reinterpret_cast<const double*>(__address));

    else if constexpr (std::is_same_v<_VectorType_, __m128>)
        return _mm_load_ps(reinterpret_cast<const float*>(__address));
}

template <typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::_StoreUpperHalf(
    void*           __address,
    _VectorType_    _Vector) noexcept
{
    _mm_storeh_pd(reinterpret_cast<double*>(__address), __intrin_bitcast<__m128d>(_Vector));
}

template <typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::_StoreLowerHalf(
    void*           __address,
    _VectorType_    _Vector) noexcept
{
    _mm_storel_epi64(reinterpret_cast<__m128i*>(__address), __intrin_bitcast<__m128i>(_Vector));
}

template <typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::_StoreUnaligned(
    void*           __address,
    _VectorType_    _Vector) noexcept
{
    if      constexpr (std::is_same_v<_VectorType_, __m128i>)
        return _mm_storeu_si128(reinterpret_cast<__m128i*>(__address), _Vector);

    else if constexpr (std::is_same_v<_VectorType_, __m128d>)
        return _mm_storeu_pd(reinterpret_cast<double*>(__address), _Vector);

    else if constexpr (std::is_same_v<_VectorType_, __m128>)
        return _mm_storeu_ps(reinterpret_cast<float*>(__address), _Vector);
}

template <typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::_StoreAligned(
    void*           __address,
    _VectorType_    _Vector) noexcept
{
    if      constexpr (std::is_same_v<_VectorType_, __m128i>)
        return _mm_store_si128(reinterpret_cast<__m128i*>(__address), _Vector);

    else if constexpr (std::is_same_v<_VectorType_, __m128d>)
        return _mm_store_pd(reinterpret_cast<double*>(__address), _Vector);

    else if constexpr (std::is_same_v<_VectorType_, __m128>)
        return _mm_store_ps(reinterpret_cast<float*>(__address), _Vector);
}


template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::__mask_store_unaligned(
    void*                                   __address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    const _VectorType_                      _Vector) noexcept
{
    _StoreUnaligned(__address, _SimdBlend<__generation, __register_policy, _DesiredType_>(
        _Vector, _LoadUnaligned<_VectorType_>(__address), _Mask));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::__mask_store_aligned(
    void*                                   __address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    const _VectorType_                      _Vector) noexcept
{
    _StoreAligned(__address, _SimdBlend<__generation, __register_policy, _DesiredType_>(
        _Vector, _LoadAligned<_VectorType_>(__address), _Mask));
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::__mask_store_unaligned(
    void* __address,
    const _MaskVectorType_  _Mask,
    const _VectorType_      _Vector) noexcept
{
    _StoreUnaligned(__address, _SimdBlend<__generation, __register_policy, _DesiredType_>(
        _Vector, _LoadUnaligned<_VectorType_>(__address), __intrin_bitcast<_VectorType_>(_Mask)));
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::__mask_store_aligned(
    void* __address,
    const _MaskVectorType_  _Mask,
    const _VectorType_      _Vector) noexcept
{
    _StoreAligned(__address, _SimdBlend<__generation, __register_policy, _DesiredType_>(
        _Vector, _LoadAligned<_VectorType_>(__address), __intrin_bitcast<_VectorType_>(_Mask)));
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::__mask_load_unaligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>    _Mask) noexcept
{
    return _SimdBlend<__generation, __register_policy, _DesiredType_>(
        _LoadUnaligned<_VectorType_>(__address),
        _SimdBroadcastZeros<__generation, __register_policy, _VectorType_>(), _Mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::__mask_load_aligned(
    const void* __address,
    const __simd_mask_type<_DesiredType_>    _Mask) noexcept
{
    return _SimdBlend<__generation, __register_policy, _DesiredType_>(
        _LoadAligned<_VectorType_>(__address),
        _SimdBroadcastZeros<__generation, __register_policy, _VectorType_>(), _Mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::__mask_load_unaligned(
    const void* __address,
    const _MaskVectorType_  _Mask) noexcept
{
    return _SimdBlend<__generation, __register_policy, _DesiredType_>(
        _LoadUnaligned<_VectorType_>(__address),
        _SimdBroadcastZeros<__generation, __register_policy, _VectorType_>(),
        __intrin_bitcast<_VectorType_>(_Mask));
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::__mask_load_aligned(
    const void* __address,
    const _MaskVectorType_  _Mask) noexcept
{
    return _SimdBlend<__generation, __register_policy, _DesiredType_>(
        _LoadAligned<_VectorType_>(__address),
        _SimdBroadcastZeros<__generation, __register_policy, _VectorType_>(),
        __intrin_bitcast<_VectorType_>(_Mask));
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::__mask_load_unaligned(
    const void*                             _Address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    _VectorType_                            __additional_source) noexcept
{
    return _SimdBlend<__generation, __register_policy, _DesiredType_>(
        _LoadUnaligned<_VectorType_>(_Address), __additional_source, _Mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::__mask_load_aligned(
    const void*                             _Address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    _VectorType_                            __additional_source) noexcept
{
    return _SimdBlend<__generation, __register_policy, _DesiredType_>(
        _LoadAligned<_VectorType_>(_Address), __additional_source, _Mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::__mask_load_unaligned(
    const void*             _Address,
    const _MaskVectorType_  _Mask,
    _VectorType_            __additional_source) noexcept
{
    return _SimdBlend<__generation, __register_policy, _DesiredType_>(
        _LoadUnaligned<_VectorType_>(_Address), 
        __additional_source,
        __intrin_bitcast<_VectorType_>(_Mask));
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::__mask_load_aligned(
    const void*             _Address,
    const _MaskVectorType_  _Mask,
    _VectorType_            __additional_source) noexcept
{
    return _SimdBlend<__generation, __register_policy, _DesiredType_>(
        _LoadAligned<_VectorType_>(_Address),
        __additional_source,
        __intrin_bitcast<_VectorType_>(_Mask));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::_CompressStoreLowerHalf(
    _DesiredType_* __address,
    __simd_mask_type<_DesiredType_>  _Mask,
    _VectorType_                    _Vector) noexcept
{
    static_assert(sizeof(_DesiredType_) != 8);

    __m128i _Shuffle;

    if constexpr (sizeof(_DesiredType_) == 4)
        _Shuffle = _LoadLowerHalf<__m128i>(_Tables32BitSse._Shuffle[_Mask]);

    else if constexpr (sizeof(_DesiredType_) == 2)
        _Shuffle = _LoadLowerHalf<__m128i>(_Tables16BitSse._Shuffle[_Mask]);

    else if constexpr (sizeof(_DesiredType_) == 1)
        _Shuffle = _LoadLowerHalf<__m128i>(_Tables8BitSse._Shuffle[_Mask]);

    const auto _Destination = _mm_shuffle_epi8(__intrin_bitcast<__m128i>(_Vector), _Shuffle);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(__address), _Destination);

    if constexpr (sizeof(_DesiredType_) == 4)
        algorithm::__advance_bytes(__address, _Tables32BitSse._Size[_Mask]);

    else if constexpr (sizeof(_DesiredType_) == 2)
        algorithm::__advance_bytes(__address, _Tables16BitSse._Size[_Mask]);

    else if constexpr (sizeof(_DesiredType_) == 1)
        algorithm::__advance_bytes(__address, _Tables8BitSse._Size[_Mask]);

    return __address;
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::_CompressStoreUpperHalf(
    _DesiredType_*                  __address,
    __simd_mask_type<_DesiredType_>  _Mask,
    _VectorType_                    _Vector) noexcept
{
    static_assert(sizeof(_DesiredType_) != 8);
    __m128i _Shuffle;

    if constexpr (sizeof(_DesiredType_) == 4)
        _Shuffle = _LoadUpperHalf<__m128i>(_Tables32BitSse._Shuffle[_Mask]);
    else if constexpr (sizeof(_DesiredType_) == 2)
        _Shuffle = _LoadUpperHalf<__m128i>(_Tables16BitSse._Shuffle[_Mask]);
    else if constexpr (sizeof(_DesiredType_) == 1)
        _Shuffle = _LoadUpperHalf<__m128i>(_Tables8BitSse._Shuffle[_Mask]);


    _mm_storeh_pd(reinterpret_cast<double*>(__address), __intrin_bitcast<__m128d>(
        _mm_shuffle_epi8(__intrin_bitcast<__m128i>(_Vector), _Shuffle)));

    if constexpr (sizeof(_DesiredType_) == 4)
        algorithm::__advance_bytes(__address, _Tables32BitSse._Size[_Mask]);
    else if constexpr (sizeof(_DesiredType_) == 2)
        algorithm::__advance_bytes(__address, _Tables16BitSse._Size[_Mask]);
    else if constexpr (sizeof(_DesiredType_) == 1)
        algorithm::__advance_bytes(__address, _Tables8BitSse._Size[_Mask]);

    return __address;
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::__compress_store_unaligned(
    _DesiredType_* __address,
    __simd_mask_type<_DesiredType_>      _Mask,
    _VectorType_                        _Vector) noexcept
{
    if      constexpr (sizeof(_DesiredType_) == 8) {
        _StoreUnaligned(__address, _mm_shuffle_epi8(__intrin_bitcast<__m128i>(_Vector),
            _LoadUnaligned<__m128i>(_Tables64BitSse._Shuffle[_Mask])));

        algorithm::__advance_bytes(__address, _Tables64BitSse._Size[_Mask]);
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        _StoreUnaligned(__address, _mm_shuffle_epi8(__intrin_bitcast<__m128i>(_Vector),
            _LoadUnaligned<__m128i>(_Tables32BitSse._Shuffle[_Mask])));

        algorithm::__advance_bytes(__address, _Tables32BitSse._Size[_Mask]);
    }
    else if constexpr (sizeof(_DesiredType_) == 2) {
        _StoreUnaligned(__address, _mm_shuffle_epi8(__intrin_bitcast<__m128i>(_Vector),
            _LoadUnaligned<__m128i>(_Tables16BitSse._Shuffle[_Mask])));

        algorithm::__advance_bytes(__address, _Tables16BitSse._Size[_Mask]);
    }
    else if constexpr (sizeof(_DesiredType_) == 1) {
        auto _Start = __address;

        __address = _CompressStoreLowerHalf(__address, _Mask & 0xFF, _Vector);
        __address = _CompressStoreLowerHalf(__address, (_Mask >> 8) & 0xFF, _mm_movehl_ps(
            __intrin_bitcast<__m128>(_mm_slli_si128(__intrin_bitcast<__m128i>(_Vector), 8)),
            __intrin_bitcast<__m128>(_Vector)));

        __mask_store_unaligned<_DesiredType_>(_Start, ~((1u << (__xmm_width - (__address - _Start))) - 1u), _Vector);
    }

    return __address;
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::__compress_store_aligned(
    _DesiredType_* __address,
    __simd_mask_type<_DesiredType_>      _Mask,
    _VectorType_                        _Vector) noexcept
{
    if      constexpr (sizeof(_DesiredType_) == 8) {
        _StoreAligned(__address, _mm_shuffle_epi8(__intrin_bitcast<__m128i>(_Vector),
            _LoadUnaligned<__m128i>(_Tables64BitSse._Shuffle[_Mask])));

        algorithm::__advance_bytes(__address, _Tables64BitSse._Size[_Mask]);
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        _StoreAligned(__address, _mm_shuffle_epi8(__intrin_bitcast<__m128i>(_Vector),
            _LoadUnaligned<__m128i>(_Tables32BitSse._Shuffle[_Mask])));

        algorithm::__advance_bytes(__address, _Tables32BitSse._Size[_Mask]);
    }
    else if constexpr (sizeof(_DesiredType_) == 2) {
        _StoreAligned(__address, _mm_shuffle_epi8(__intrin_bitcast<__m128i>(_Vector),
            _LoadUnaligned<__m128i>(_Tables16BitSse._Shuffle[_Mask])));

        algorithm::__advance_bytes(__address, _Tables16BitSse._Size[_Mask]);
    }
    else if constexpr (sizeof(_DesiredType_) == 1) {
        auto _Start = __address;

        __address = _CompressStoreLowerHalf(__address, _Mask & 0xFF, _Vector);
        __address = _CompressStoreLowerHalf(__address, (_Mask >> 8) & 0xFF, _mm_movehl_ps(
            __intrin_bitcast<__m128>(_mm_slli_si128(__intrin_bitcast<__m128i>(_Vector), 8)),
            __intrin_bitcast<__m128>(_Vector)));

        __mask_store_unaligned<_DesiredType_>(_Start, ~((1u << (__xmm_width - (__address - _Start))) - 1u), _Vector);
    }

    return __address;
}

template <typename _Type_>
simd_stl_always_inline __m128i __simd_memory_access<arch::CpuFeature::SSE2, numeric::xmm128>::_MakeTailMask(uint32 bytes) noexcept {
    constexpr unsigned int _TailMask[8] = { ~0u, ~0u, ~0u, ~0u, 0, 0, 0, 0 };
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(
        reinterpret_cast<const unsigned char*>(_TailMask) + (16 - bytes)));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::SSSE3, numeric::xmm128>::_CompressStoreLowerHalf(
    _DesiredType_* __address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    const _VectorType_                      _Vector) noexcept
{
    static_assert(sizeof(_DesiredType_) != 8);

    __m128i _Shuffle;

    if constexpr (sizeof(_DesiredType_) == 4)
        _Shuffle = _LoadLowerHalf<__m128i>(_Tables32BitSse._Shuffle[_Mask]);

    else if constexpr (sizeof(_DesiredType_) == 2)
        _Shuffle = _LoadLowerHalf<__m128i>(_Tables16BitSse._Shuffle[_Mask]);

    else if constexpr (sizeof(_DesiredType_) == 1)
        _Shuffle = _LoadLowerHalf<__m128i>(_Tables8BitSse._Shuffle[_Mask]);

    const auto _Destination = _mm_shuffle_epi8(__intrin_bitcast<__m128i>(_Vector), _Shuffle);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(__address), _Destination);

    if constexpr (sizeof(_DesiredType_) == 4)
        algorithm::__advance_bytes(__address, _Tables32BitSse._Size[_Mask]);

    else if constexpr (sizeof(_DesiredType_) == 2)
        algorithm::__advance_bytes(__address, _Tables16BitSse._Size[_Mask]);

    else if constexpr (sizeof(_DesiredType_) == 1)
        algorithm::__advance_bytes(__address, _Tables8BitSse._Size[_Mask]);

    return __address;
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::SSSE3, numeric::xmm128>::_CompressStoreUpperHalf(
    _DesiredType_* __address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    const _VectorType_                      _Vector) noexcept
{
    static_assert(sizeof(_DesiredType_) != 8);
    __m128i _Shuffle;

    if constexpr (sizeof(_DesiredType_) == 4)
        _Shuffle = _LoadUpperHalf<__m128i>(_Tables32BitSse._Shuffle[_Mask]);
    else if constexpr (sizeof(_DesiredType_) == 2)
        _Shuffle = _LoadUpperHalf<__m128i>(_Tables16BitSse._Shuffle[_Mask]);
    else if constexpr (sizeof(_DesiredType_) == 1)
        _Shuffle = _LoadUpperHalf<__m128i>(_Tables8BitSse._Shuffle[_Mask]);


    _mm_storeh_pd(reinterpret_cast<double*>(__address), __intrin_bitcast<__m128d>(
        _mm_shuffle_epi8(__intrin_bitcast<__m128i>(_Vector), _Shuffle)));

    if constexpr (sizeof(_DesiredType_) == 4)
        algorithm::__advance_bytes(__address, _Tables32BitSse._Size[_Mask]);
    else if constexpr (sizeof(_DesiredType_) == 2)
        algorithm::__advance_bytes(__address, _Tables16BitSse._Size[_Mask]);
    else if constexpr (sizeof(_DesiredType_) == 1)
        algorithm::__advance_bytes(__address, _Tables8BitSse._Size[_Mask]);

    return __address;
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::SSSE3, numeric::xmm128>::__compress_store_unaligned(
    _DesiredType_* __address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    const _VectorType_                      _Vector) noexcept
{
    if      constexpr (sizeof(_DesiredType_) == 8) {
        _StoreUnaligned(__address, _mm_shuffle_epi8(__intrin_bitcast<__m128i>(_Vector),
            _LoadUnaligned<__m128i>(_Tables64BitSse._Shuffle[_Mask])));

        algorithm::__advance_bytes(__address, _Tables64BitSse._Size[_Mask]);
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        _StoreUnaligned(__address, _mm_shuffle_epi8(__intrin_bitcast<__m128i>(_Vector),
            _LoadUnaligned<__m128i>(_Tables32BitSse._Shuffle[_Mask])));

        algorithm::__advance_bytes(__address, _Tables32BitSse._Size[_Mask]);
    }
    else if constexpr (sizeof(_DesiredType_) == 2) {
        _StoreUnaligned(__address, _mm_shuffle_epi8(__intrin_bitcast<__m128i>(_Vector),
            _LoadUnaligned<__m128i>(_Tables16BitSse._Shuffle[_Mask])));

        algorithm::__advance_bytes(__address, _Tables16BitSse._Size[_Mask]);
    }
    else if constexpr (sizeof(_DesiredType_) == 1) {
        auto _Start = __address;

        __address = _CompressStoreLowerHalf(__address, _Mask & 0xFF, _Vector);
        __address = _CompressStoreLowerHalf(__address, (_Mask >> 8) & 0xFF, _mm_movehl_ps(
            __intrin_bitcast<__m128>(_mm_slli_si128(__intrin_bitcast<__m128i>(_Vector), 8)),
            __intrin_bitcast<__m128>(_Vector)));

        __mask_store_unaligned<_DesiredType_>(_Start, ~((1u << (__xmm_width - (__address - _Start))) - 1u), _Vector);
    }

    return __address;
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::SSSE3, numeric::xmm128>::__compress_store_aligned(
    _DesiredType_* __address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    const _VectorType_                      _Vector) noexcept
{
    if      constexpr (sizeof(_DesiredType_) == 8) {
        _StoreAligned(__address, _mm_shuffle_epi8(__intrin_bitcast<__m128i>(_Vector),
            _LoadUnaligned<__m128i>(_Tables64BitSse._Shuffle[_Mask])));

        algorithm::__advance_bytes(__address, _Tables64BitSse._Size[_Mask]);
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        _StoreAligned(__address, _mm_shuffle_epi8(__intrin_bitcast<__m128i>(_Vector),
            _LoadUnaligned<__m128i>(_Tables32BitSse._Shuffle[_Mask])));

        algorithm::__advance_bytes(__address, _Tables32BitSse._Size[_Mask]);
    }
    else if constexpr (sizeof(_DesiredType_) == 2) {
        _StoreAligned(__address, _mm_shuffle_epi8(__intrin_bitcast<__m128i>(_Vector),
            _LoadUnaligned<__m128i>(_Tables16BitSse._Shuffle[_Mask])));

        algorithm::__advance_bytes(__address, _Tables16BitSse._Size[_Mask]);
    }
    else if constexpr (sizeof(_DesiredType_) == 1) {
        auto _Start = __address;

        __address = _CompressStoreLowerHalf(__address, _Mask & 0xFF, _Vector);
        __address = _CompressStoreLowerHalf(__address, (_Mask >> 8) & 0xFF, _mm_movehl_ps(
            __intrin_bitcast<__m128>(_mm_slli_si128(__intrin_bitcast<__m128i>(_Vector), 8)),
            __intrin_bitcast<__m128>(_Vector)));

        __mask_store_unaligned<_DesiredType_>(_Start, ~((1u << (__xmm_width - (__address - _Start))) - 1u), _Vector);
    }

    return __address;
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE41, numeric::xmm128>::_NonTemporalLoad(const void* where) noexcept {
    return __intrin_bitcast<_VectorType_>(_mm_stream_load_si128(reinterpret_cast<const __m128i*>(where)));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::SSE41, numeric::xmm128>::__mask_store_unaligned(
    void* __address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    const _VectorType_                      _Vector) noexcept
{
    _StoreUnaligned(__address, _SimdBlend<__generation, __register_policy, _DesiredType_>(
        _Vector, _LoadUnaligned<_VectorType_>(__address), _Mask));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::SSE41, numeric::xmm128>::__mask_store_aligned(
    void* __address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    const _VectorType_                      _Vector) noexcept
{
    _StoreAligned(__address, _SimdBlend<__generation, __register_policy, _DesiredType_>(
        _Vector, _LoadAligned<_VectorType_>(__address), _Mask));
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::SSE41, numeric::xmm128>::__mask_store_unaligned(
    void* __address,
    const _MaskVectorType_  _Mask,
    const _VectorType_      _Vector) noexcept
{
    _StoreUnaligned(__address, _SimdBlend<__generation, __register_policy, _DesiredType_>(
        _Vector, _LoadUnaligned<_VectorType_>(__address), __intrin_bitcast<_VectorType_>(_Mask)));
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::SSE41, numeric::xmm128>::__mask_store_aligned(
    void* __address,
    const _MaskVectorType_  _Mask,
    const _VectorType_      _Vector) noexcept
{
    _StoreAligned(__address, _SimdBlend<__generation, __register_policy, _DesiredType_>(
        _Vector, _LoadAligned<_VectorType_>(__address), __intrin_bitcast<_VectorType_>(_Mask)));
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE41, numeric::xmm128>::__mask_load_unaligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>    _Mask) noexcept
{
    return _SimdBlend<__generation, __register_policy, _DesiredType_>(
        _LoadUnaligned<_VectorType_>(__address),
        _SimdBroadcastZeros<__generation, __register_policy, _VectorType_>(), _Mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE41, numeric::xmm128>::__mask_load_aligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>    _Mask) noexcept
{
    return _SimdBlend<__generation, __register_policy, _DesiredType_>(
        _LoadAligned<_VectorType_>(__address),
        _SimdBroadcastZeros<__generation, __register_policy, _VectorType_>(), _Mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE41, numeric::xmm128>::__mask_load_unaligned(
    const void* __address,
    const _MaskVectorType_  _Mask) noexcept
{
    return _SimdBlend<__generation, __register_policy, _DesiredType_>(
        _LoadUnaligned<_VectorType_>(__address),
        _SimdBroadcastZeros<__generation, __register_policy, _VectorType_>(),
        __intrin_bitcast<_VectorType_>(_Mask));
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE41, numeric::xmm128>::__mask_load_aligned(
    const void* __address,
    const _MaskVectorType_  _Mask) noexcept
{
    return _SimdBlend<__generation, __register_policy, _DesiredType_>(
        _LoadAligned<_VectorType_>(__address),
        _SimdBroadcastZeros<__generation, __register_policy, _VectorType_>(),
        __intrin_bitcast<_VectorType_>(_Mask));
}


template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE41, numeric::xmm128>::__mask_load_unaligned(
    const void*                             _Address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    _VectorType_                            __additional_source) noexcept
{
    return _SimdBlend<__generation, __register_policy, _DesiredType_>(
        _LoadUnaligned<_VectorType_>(_Address), __additional_source, _Mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE41, numeric::xmm128>::__mask_load_aligned(
    const void*                             _Address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    _VectorType_                            __additional_source) noexcept
{
    return _SimdBlend<__generation, __register_policy, _DesiredType_>(
        _LoadAligned<_VectorType_>(_Address), __additional_source, _Mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE41, numeric::xmm128>::__mask_load_unaligned(
    const void*             _Address,
    const _MaskVectorType_  _Mask,
    _VectorType_            __additional_source) noexcept
{
    return _SimdBlend<__generation, __register_policy, _DesiredType_>(
        _LoadUnaligned<_VectorType_>(_Address), 
        __additional_source,
        __intrin_bitcast<_VectorType_>(_Mask));
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE41, numeric::xmm128>::__mask_load_aligned(
    const void*             _Address,
    const _MaskVectorType_  _Mask,
    _VectorType_            __additional_source) noexcept
{
    return _SimdBlend<__generation, __register_policy, _DesiredType_>(
        _LoadAligned<_VectorType_>(_Address),
        __additional_source,
        __intrin_bitcast<_VectorType_>(_Mask));
}


#pragma endregion

#pragma region Avx-Avx2 memory access


template <typename _Type_>
simd_stl_always_inline auto __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::_MakeTailMask(uint32 _Bytes) noexcept {
    constexpr unsigned int _TailMask[16] = {
        ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, 0, 0, 0, 0, 0, 0, 0, 0 };
    return _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(
        reinterpret_cast<const unsigned char*>(_TailMask) + (32 - _Bytes)));
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::_LoadUpperHalf(const void* __address) noexcept {
    return __intrin_bitcast<_VectorType_>(_mm256_insertf128_si256(_mm256_setzero_si256(),
        _mm_lddqu_si128(reinterpret_cast<const __m128i*>(__address)), 1));
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::_LoadLowerHalf(const void* __address) noexcept {
    return __intrin_bitcast<_VectorType_>(_mm_lddqu_si128(reinterpret_cast<const __m128i*>(__address)));
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::_NonTemporalLoad(const void* __address) noexcept {
    return _LoadAligned<_VectorType_, void>(__address);
}

template <typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::_NonTemporalStore(
    void* __address,
    _VectorType_    _Vector) noexcept
{
    _mm256_stream_si256(reinterpret_cast<__m256i*>(__address), __intrin_bitcast<__m256i>(_Vector));
}

simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::_StreamingFence() noexcept {
    return _mm_sfence();
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::_LoadUnaligned(const void* __address) noexcept {
    if      constexpr (std::is_same_v<_VectorType_, __m256i>)
        return _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(__address));

    else if constexpr (std::is_same_v<_VectorType_, __m256d>)
        return _mm256_loadu_pd(reinterpret_cast<const double*>(__address));

    else if constexpr (std::is_same_v<_VectorType_, __m256>)
        return _mm256_loadu_ps(reinterpret_cast<const float*>(__address));
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::_LoadAligned(const void* __address) noexcept {
    if      constexpr (std::is_same_v<_VectorType_, __m256i>)
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(__address));

    else if constexpr (std::is_same_v<_VectorType_, __m256d>)
        return _mm256_load_pd(reinterpret_cast<const double*>(__address));

    else if constexpr (std::is_same_v<_VectorType_, __m256>)
        return _mm256_load_ps(reinterpret_cast<const float*>(__address));
}

template <typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::_StoreUpperHalf(
    void* __address,
    _VectorType_    _Vector) noexcept
{
    _mm_storeu_si128(reinterpret_cast<__m128i*>(__address), _mm256_extractf128_si256(__intrin_bitcast<__m256i>(_Vector), 1));
}

template <typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::_StoreLowerHalf(
    void* __address,
    _VectorType_    _Vector) noexcept
{
    _mm_storeu_si128(reinterpret_cast<__m128i*>(__address), __intrin_bitcast<__m128i>(_Vector));
}

template <typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::_StoreUnaligned(
    void* __address,
    _VectorType_    _Vector) noexcept
{
    if      constexpr (std::is_same_v<_VectorType_, __m256i>)
        return _mm256_storeu_si256(reinterpret_cast<__m256i*>(__address), _Vector);

    else if constexpr (std::is_same_v<_VectorType_, __m256d>)
        return _mm256_storeu_pd(reinterpret_cast<double*>(__address), _Vector);

    else if constexpr (std::is_same_v<_VectorType_, __m256>)
        return _mm256_storeu_ps(reinterpret_cast<float*>(__address), _Vector);
}

template <typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::_StoreAligned(
    void* __address,
    _VectorType_    _Vector) noexcept
{
    if      constexpr (std::is_same_v<_VectorType_, __m256i>)
        return _mm256_store_si256(reinterpret_cast<__m256i*>(__address), _Vector);

    else if constexpr (std::is_same_v<_VectorType_, __m256d>)
        return _mm256_store_pd(reinterpret_cast<double*>(__address), _Vector);

    else if constexpr (std::is_same_v<_VectorType_, __m256>)
        return _mm256_store_ps(reinterpret_cast<float*>(__address), _Vector);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::__mask_store_unaligned(
    void* __address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    const _VectorType_                      _Vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        _mm256_maskstore_pd(reinterpret_cast<double*>(__address),
            __simd_to_vector<__generation, __register_policy, __m256i, _DesiredType_>(_Mask),
            __intrin_bitcast<__m256d>(_Vector));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        _mm256_maskstore_ps(reinterpret_cast<float*>(__address),
            __simd_to_vector<__generation, __register_policy, __m256i, _DesiredType_>(_Mask),
            __intrin_bitcast<__m256>(_Vector));
    }
    else {
        _StoreUnaligned(__address, _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _Vector, _LoadUnaligned<_VectorType_>(__address), _Mask));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::__mask_store_aligned(
    void* __address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    const _VectorType_                      _Vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        _mm256_maskstore_pd(reinterpret_cast<double*>(__address),
            __simd_to_vector<__generation, __register_policy, __m256i, _DesiredType_>(_Mask),
            __intrin_bitcast<__m256d>(_Vector));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        _mm256_maskstore_ps(reinterpret_cast<float*>(__address),
            __simd_to_vector<__generation, __register_policy, __m256i, _DesiredType_>(_Mask),
            __intrin_bitcast<__m256>(_Vector));
    }
    else {
        _StoreAligned(__address, _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _Vector, _LoadAligned<_VectorType_>(__address), _Mask));
    }
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::__mask_store_unaligned(
    void* __address,
    const _MaskVectorType_  _Mask,
    const _VectorType_      _Vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        _mm256_maskstore_pd(reinterpret_cast<double*>(__address),
            __intrin_bitcast<__m256i>(_Mask), __intrin_bitcast<__m256d>(_Vector));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        _mm256_maskstore_ps(reinterpret_cast<float*>(__address),
            __intrin_bitcast<__m256i>(_Mask), __intrin_bitcast<__m256>(_Vector));
    }
    else {
        _StoreUnaligned(__address, _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _Vector, _LoadUnaligned<_VectorType_>(__address), __intrin_bitcast<_VectorType_>(_Mask)));
    }
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::__mask_store_aligned(
    void* __address,
    const _MaskVectorType_  _Mask,
    const _VectorType_      _Vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        _mm256_maskstore_pd(reinterpret_cast<double*>(__address),
            __intrin_bitcast<__m256i>(_Mask), __intrin_bitcast<__m256d>(_Vector));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        _mm256_maskstore_ps(reinterpret_cast<float*>(__address),
            __intrin_bitcast<__m256i>(_Mask), __intrin_bitcast<__m256>(_Vector));
    }
    else {
        _StoreAligned(__address, _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _Vector, _LoadAligned<_VectorType_>(__address), __intrin_bitcast<_VectorType_>(_Mask)));
    }
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::__mask_load_unaligned(
    const void* __address,
    const __simd_mask_type<_DesiredType_>    _Mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        return __intrin_bitcast<_VectorType_>(_mm256_maskload_pd(
            reinterpret_cast<const double*>(__address),
            __simd_to_vector<__generation, __register_policy, __m256i, _DesiredType_>(_Mask)));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        return __intrin_bitcast<_VectorType_>(_mm256_maskload_ps(
            reinterpret_cast<const float*>(__address),
            __simd_to_vector<__generation, __register_policy, __m256i, _DesiredType_>(_Mask)));
    }
    else {
        return _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _LoadUnaligned<_VectorType_>(__address),
            _SimdBroadcastZeros<__generation, __register_policy, _VectorType_>(), _Mask);
    }
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::__mask_load_aligned(
    const void* __address,
    const __simd_mask_type<_DesiredType_>    _Mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        return __intrin_bitcast<_VectorType_>(_mm256_maskload_pd(
            reinterpret_cast<const double*>(__address),
            __simd_to_vector<__generation, __register_policy, __m256i, _DesiredType_>(_Mask)));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        return __intrin_bitcast<_VectorType_>(_mm256_maskload_ps(
            reinterpret_cast<const float*>(__address),
            __simd_to_vector<__generation, __register_policy, __m256i, _DesiredType_>(_Mask)));
    }
    else {
        return _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _LoadAligned<_VectorType_>(__address),
            _SimdBroadcastZeros<__generation, __register_policy, _VectorType_>(), _Mask);
    }
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::__mask_load_unaligned(
    const void* __address,
    const _MaskVectorType_  _Mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        return __intrin_bitcast<_VectorType_>(_mm256_maskload_pd(
            reinterpret_cast<const double*>(__address),
            __intrin_bitcast<__m256i>(_Mask)));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        return __intrin_bitcast<_VectorType_>(_mm256_maskload_ps(
            reinterpret_cast<const float*>(__address),
            __intrin_bitcast<__m256i>(_Mask)));
    }
    else {
        return _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _LoadUnaligned<_VectorType_>(__address),
            _SimdBroadcastZeros<__generation, __register_policy, _VectorType_>(),
            __intrin_bitcast<_VectorType_>(_Mask));
    }
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __mask_load_aligned(
    const void* __address,
    const _MaskVectorType_  _Mask) noexcept
{
    return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address, _Mask);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::_CompressStoreLowerHalf(
    _DesiredType_* __address,
    __simd_mask_type<_DesiredType_>  _Mask,
    _VectorType_                    _Vector) noexcept
{
    return _SimdCompressStoreUnaligned<arch::CpuFeature::SSE42, xmm128, _DesiredType_>(
        __address, _Mask, __intrin_bitcast<__m128i>(_Vector));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::_CompressStoreUpperHalf(
    _DesiredType_* __address,
    __simd_mask_type<_DesiredType_>  _Mask,
    _VectorType_                    _Vector) noexcept
{
    return _SimdCompressStoreUnaligned<arch::CpuFeature::SSE42, xmm128, _DesiredType_>(__address, _Mask,
        _mm256_extractf128_si256(__intrin_bitcast<__m256i>(_Vector), 1));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::__compress_store_unaligned(
    _DesiredType_* __address,
    __simd_mask_type<_DesiredType_>      _Mask,
    _VectorType_                        _Vector) noexcept
{
    using _MaskType = __simd_mask_type<_DesiredType_>;
    using _HalfType = IntegerForSize<_Max<(sizeof(_DesiredType_) >> 1), 1>()>::Unsigned;

    constexpr auto _Maximum = math::__maximum_integral_limit<_HalfType>();
    constexpr auto _Shift = (sizeof(_MaskType) << 2);

    __address = _CompressStoreLowerHalf<_DesiredType_>(__address, _Mask & _Maximum, _Vector);
    __address = _CompressStoreUpperHalf<_DesiredType_>(__address, (_Mask >> _Shift) & _Maximum, _Vector);

    return __address;
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::AVX, numeric::ymm256>::__compress_store_aligned(
    _DesiredType_* __address,
    __simd_mask_type<_DesiredType_>      _Mask,
    _VectorType_                        _Vector) noexcept
{
    return __compress_store_unaligned<_DesiredType_>(__address, _Mask, _Vector);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX2, numeric::ymm256>::__mask_load_unaligned(
    const void* __address,
    const __simd_mask_type<_DesiredType_>    _Mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        return __intrin_bitcast<_VectorType_>(_mm256_maskload_epi64(
            reinterpret_cast<const long long*>(__address),
            __simd_to_vector<__generation, __register_policy, __m256i, _DesiredType_>(_Mask)));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        return __intrin_bitcast<_VectorType_>(_mm256_maskload_epi32(
            reinterpret_cast<const int*>(__address),
            __simd_to_vector<__generation, __register_policy, __m256i, _DesiredType_>(_Mask)));
    }
    else {
        return _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _LoadUnaligned<_VectorType_>(__address),
            _SimdBroadcastZeros<__generation, __register_policy, _VectorType_>(), _Mask);
    }
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX2, numeric::ymm256>::__mask_load_aligned(
    const void*                             _Address,
    const __simd_mask_type<_DesiredType_>    _Mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) >= 4)
        return __mask_load_unaligned<_VectorType_, _DesiredType_>(_Address, _Mask);
    else
        return _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _LoadAligned<_VectorType_>(_Address),
            _SimdBroadcastZeros<__generation, __register_policy, _VectorType_>(),
            _Mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX2, numeric::ymm256>::__mask_load_unaligned(
    const void*             _Address,
    const _MaskVectorType_  _Mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        return __intrin_bitcast<_VectorType_>(_mm256_maskload_pd(
            reinterpret_cast<const double*>(_Address),
            __intrin_bitcast<__m256i>(_Mask)));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        return __intrin_bitcast<_VectorType_>(_mm256_maskload_ps(
            reinterpret_cast<const float*>(_Address),
            __intrin_bitcast<__m256i>(_Mask)));
    }
    else {
        return _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _LoadUnaligned<_VectorType_>(_Address),
            _SimdBroadcastZeros<__generation, __register_policy, _VectorType_>(),
            __intrin_bitcast<_VectorType_>(_Mask));
    }
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX2, numeric::ymm256>::__mask_load_aligned(
    const void*             _Address,
    const _MaskVectorType_  _Mask) noexcept
{
    return __mask_load_unaligned<_VectorType_, _DesiredType_>(_Address, _Mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX2, numeric::ymm256>::__mask_load_unaligned(
    const void*                             _Address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    _VectorType_                            __additional_source) noexcept
{
    return __mask_load_unaligned<_VectorType_, _DesiredType_>(_Address, 
        __simd_to_vector<__generation, __register_policy, _VectorType_, _DesiredType_>(_Mask), __additional_source);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX2, numeric::ymm256>::__mask_load_aligned(
    const void*                             _Address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    _VectorType_                            __additional_source) noexcept
{
    return __mask_load_unaligned<_VectorType_, _DesiredType_>(_Address, _Mask, __additional_source);
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX2, numeric::ymm256>::__mask_load_unaligned(
    const void*             _Address,
    const _MaskVectorType_  _Mask,
    _VectorType_            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        const auto _Loaded = __intrin_bitcast<_VectorType_>(_mm256_maskload_pd(
            reinterpret_cast<const double*>(_Address),
            __intrin_bitcast<__m256i>(_Mask)));

        return _SimdBlend<__generation, __register_policy, _DesiredType_>(_Loaded, __additional_source, _Mask);
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        const auto _Loaded = __intrin_bitcast<_VectorType_>(_mm256_maskload_ps(
            reinterpret_cast<const float*>(_Address),
            __intrin_bitcast<__m256i>(_Mask)));

        return _SimdBlend<__generation, __register_policy, _DesiredType_>(_Loaded, __additional_source, _Mask);
    }
    else {
        return _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _LoadUnaligned<_VectorType_>(_Address),
            __additional_source,
            __intrin_bitcast<_VectorType_>(_Mask));
    }
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX2, numeric::ymm256>::__mask_load_aligned(
    const void*             _Address,
    const _MaskVectorType_  _Mask,
    _VectorType_            __additional_source) noexcept
{
    return __mask_load_unaligned<_VectorType_, _DesiredType_>(_Address, _Mask, __additional_source);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX2, numeric::ymm256>::__mask_store_unaligned(
    void* __address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    const _VectorType_                      _Vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        _mm256_maskstore_pd(reinterpret_cast<double*>(__address),
            __simd_to_vector<__generation, __register_policy, __m256i, _DesiredType_>(_Mask),
            __intrin_bitcast<__m256d>(_Vector));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        _mm256_maskstore_ps(reinterpret_cast<float*>(__address),
            __simd_to_vector<__generation, __register_policy, __m256i, _DesiredType_>(_Mask),
            __intrin_bitcast<__m256>(_Vector));
    }
    else {
        _StoreUnaligned(__address, _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _Vector, _LoadUnaligned<_VectorType_>(__address), _Mask));
    }
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX2, numeric::ymm256>::__mask_store_aligned(
    void* __address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    const _VectorType_                      _Vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        _mm256_maskstore_pd(reinterpret_cast<double*>(__address),
            __simd_to_vector<__generation, __register_policy, __m256i, _DesiredType_>(_Mask),
            __intrin_bitcast<__m256d>(_Vector));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        _mm256_maskstore_ps(reinterpret_cast<float*>(__address),
            __simd_to_vector<__generation, __register_policy, __m256i, _DesiredType_>(_Mask),
            __intrin_bitcast<__m256>(_Vector));
    }
    else {
        _StoreAligned(__address, _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _Vector, _LoadAligned<_VectorType_>(__address), _Mask));
    }
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX2, numeric::ymm256>::__mask_store_unaligned(
    void* __address,
    const _MaskVectorType_  _Mask,
    const _VectorType_      _Vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        _mm256_maskstore_pd(reinterpret_cast<double*>(__address),
            __intrin_bitcast<__m256i>(_Mask), __intrin_bitcast<__m256d>(_Vector));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        _mm256_maskstore_ps(reinterpret_cast<float*>(__address),
            __intrin_bitcast<__m256i>(_Mask), __intrin_bitcast<__m256>(_Vector));
    }
    else {
        _StoreUnaligned(__address, _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _Vector, _LoadUnaligned<_VectorType_>(__address), __intrin_bitcast<_VectorType_>(_Mask)));
    }
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX2, numeric::ymm256>::__mask_store_aligned(
    void* __address,
    const _MaskVectorType_  _Mask,
    const _VectorType_      _Vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        _mm256_maskstore_pd(reinterpret_cast<double*>(__address),
            __intrin_bitcast<__m256i>(_Mask), __intrin_bitcast<__m256d>(_Vector));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        _mm256_maskstore_ps(reinterpret_cast<float*>(__address),
            __intrin_bitcast<__m256i>(_Mask), __intrin_bitcast<__m256>(_Vector));
    }
    else {
        _StoreAligned(__address, _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _Vector, _LoadAligned<_VectorType_>(__address), __intrin_bitcast<_VectorType_>(_Mask)));
    }
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX2, numeric::ymm256>::_NonTemporalLoad(const void* __address) noexcept {
    return __intrin_bitcast<_VectorType_>(_mm256_stream_load_si256(reinterpret_cast<const __m256i*>(__address)));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::AVX2, numeric::ymm256>::__compress_store_unaligned(
    _DesiredType_* __address,
    __simd_mask_type<_DesiredType_>      _Mask,
    _VectorType_                        _Vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        const auto _Shuffle = _mm256_cvtepu8_epi32(_mm_loadu_si64(_Tables64BitAvx._Shuffle[_Mask]));
        const auto _Destination = _mm256_permutevar8x32_epi32(__intrin_bitcast<__m256i>(_Vector), _Shuffle);

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(__address), _Destination);
        algorithm::__advance_bytes(__address, _Tables64BitAvx._Size[_Mask]);
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        const auto _Shuffle = _mm256_cvtepu8_epi32(_mm_loadu_si64(_Tables32BitAvx._Shuffle[_Mask]));
        const auto _Destination = _mm256_permutevar8x32_epi32(__intrin_bitcast<__m256i>(_Vector), _Shuffle);

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(__address), _Destination);
        algorithm::__advance_bytes(__address, _Tables32BitAvx._Size[_Mask]);
    }
    else {
        constexpr auto _Length = sizeof(_VectorType_) / sizeof(_DesiredType_);
        _DesiredType_ _Source[_Length];

        _SimdStoreUnaligned<__generation, __register_policy>(_Source, _Vector);

        auto _Start = __address;

        for (auto _Index = 0; _Index < _Length; ++_Index)
            if (!((_Mask >> _Index) & 1))
                *__address++ = _Source[_Index];

        const auto _Bytes = (__address - _Start);
        std::memcpy(__address, _Source + _Bytes, sizeof(_VectorType_) - _Bytes);
    }

    return __address;
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::AVX2, numeric::ymm256>::__compress_store_aligned(
    _DesiredType_* __address,
    __simd_mask_type<_DesiredType_>      _Mask,
    _VectorType_                        _Vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        const auto _Shuffle = _mm256_cvtepu8_epi32(_mm_loadu_si64(_Tables64BitAvx._Shuffle[_Mask]));
        const auto _Destination = _mm256_permutevar8x32_epi32(__intrin_bitcast<__m256i>(_Vector), _Shuffle);

        _mm256_store_si256(reinterpret_cast<__m256i*>(__address), _Destination);
        algorithm::__advance_bytes(__address, _Tables64BitAvx._Size[_Mask]);
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        const auto _Shuffle = _mm256_cvtepu8_epi32(_mm_loadu_si64(_Tables32BitAvx._Shuffle[_Mask]));
        const auto _Destination = _mm256_permutevar8x32_epi32(__intrin_bitcast<__m256i>(_Vector), _Shuffle);

        _mm256_store_si256(reinterpret_cast<__m256i*>(__address), _Destination);
        algorithm::__advance_bytes(__address, _Tables32BitAvx._Size[_Mask]);
    }
    else {
        constexpr auto _Length = sizeof(_VectorType_) / sizeof(_DesiredType_);

        _DesiredType_ _Source[_Length];

        _SimdStoreUnaligned<__generation, __register_policy>(_Source, _Vector);

        auto _Start = __address;

        for (auto _Index = 0; _Index < _Length; ++_Index)
            if (!((_Mask >> _Index) & 1))
                *__address++ = _Source[_Index];

        const auto _Bytes = (__address - _Start);
        std::memcpy(__address, _Source + _Bytes, sizeof(_VectorType_) - _Bytes);
    }

    return __address;
}

#pragma endregion

#pragma region Avx512 memory access

    
template <typename _Type_>
simd_stl_always_inline auto __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::_MakeTailMask(uint32 _Bytes) noexcept {
    const auto _Elements = _Bytes / sizeof(_Type_);
    return (_Elements == 0) ? 0 : (static_cast<__simd_mask_type<_Type_>>((1ull << _Elements) - 1));
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::_LoadUpperHalf(const void* __address) noexcept {
    return __intrin_bitcast<_VectorType_>(_mm256_lddqu_si256(reinterpret_cast<const __m256i*>(__address)));
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::_LoadLowerHalf(const void* __address) noexcept {
    return __intrin_bitcast<_VectorType_>(_mm512_inserti64x4(
        _mm512_setzero_si512(), _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(__address)), 1));
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::_NonTemporalLoad(const void* __address) noexcept {
    return __intrin_bitcast<_VectorType_>(_mm512_stream_load_si512(__address));
}

template <typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::_NonTemporalStore(
    void* __address,
    _VectorType_    _Vector) noexcept
{
    _mm512_stream_si512(__address, __intrin_bitcast<__m512i>(_Vector));
}

simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::_StreamingFence() noexcept {
    return _mm_sfence();
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::_LoadUnaligned(const void* __address) noexcept {
    if      constexpr (std::is_same_v<_VectorType_, __m512i>)
        return _mm512_loadu_si512(__address);

    else if constexpr (std::is_same_v<_VectorType_, __m512d>)
        return _mm512_loadu_pd(__address);

    else if constexpr (std::is_same_v<_VectorType_, __m512>)
        return _mm512_loadu_ps(__address);
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::_LoadAligned(const void* __address) noexcept {
    if      constexpr (std::is_same_v<_VectorType_, __m512i>)
        return _mm512_load_si512(__address);

    else if constexpr (std::is_same_v<_VectorType_, __m512d>)
        return _mm512_load_pd(__address);

    else if constexpr (std::is_same_v<_VectorType_, __m512>)
        return _mm512_load_ps(__address);
}

template <typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::_StoreUpperHalf(
    void* __address,
    _VectorType_    _Vector) noexcept
{
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(__address), _mm512_extracti64x4_epi64(__intrin_bitcast<__m512i>(_Vector), 1));
}

template <typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::_StoreLowerHalf(
    void* __address,
    _VectorType_    _Vector) noexcept
{
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(__address), __intrin_bitcast<__m256d>(_Vector));
}

template <typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::_StoreUnaligned(
    void* __address,
    _VectorType_    _Vector) noexcept
{
    if      constexpr (std::is_same_v<_VectorType_, __m512i>)
        return _mm512_storeu_si512(__address, _Vector);

    else if constexpr (std::is_same_v<_VectorType_, __m512d>)
        return _mm512_storeu_pd(__address, _Vector);

    else if constexpr (std::is_same_v<_VectorType_, __m512>)
        return _mm512_storeu_ps(__address, _Vector);
}

template <typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::_StoreAligned(
    void* __address,
    _VectorType_    _Vector) noexcept
{
    if      constexpr (std::is_same_v<_VectorType_, __m512i>)
        return _mm512_store_si512(__address, _Vector);

    else if constexpr (std::is_same_v<_VectorType_, __m512d>)
        return _mm512_store_pd(__address, _Vector);

    else if constexpr (std::is_same_v<_VectorType_, __m512>)
        return _mm512_store_ps(__address, _Vector);
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__mask_store_unaligned(
    void* __address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    const _VectorType_                      _Vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        _mm512_mask_storeu_epi64(__address, _Mask, __intrin_bitcast<__m512i>(_Vector));

    else if constexpr (sizeof(_DesiredType_) == 4)
        _mm512_mask_storeu_epi32(__address, _Mask, __intrin_bitcast<__m512i>(_Vector));

    else
        _StoreUnaligned(__address, _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _Vector, _LoadUnaligned<_VectorType_>(__address), _Mask));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__mask_store_aligned(
    void* __address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    const _VectorType_                      _Vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        _mm512_mask_store_epi64(__address, _Mask, __intrin_bitcast<__m512i>(_Vector));

    else if constexpr (sizeof(_DesiredType_) == 4)
        _mm512_mask_store_epi32(__address, _Mask, __intrin_bitcast<__m512i>(_Vector));

    else
        _StoreAligned(__address, _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _Vector, _LoadAligned<_VectorType_>(__address), _Mask));
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__mask_store_unaligned(
    void* __address,
    const _MaskVectorType_  _Mask,
    const _VectorType_      _Vector) noexcept
{
    return __mask_store_unaligned<_DesiredType_>(__address, __simd_to_mask<
        __generation, __register_policy, _DesiredType_>(_Mask), _Vector);
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__mask_store_aligned(
    void* __address,
    const _MaskVectorType_  _Mask,
    const _VectorType_      _Vector) noexcept
{
    return __mask_store_unaligned<_DesiredType_>(__address, __simd_to_mask<
        __generation, __register_policy, _DesiredType_>(_Mask), _Vector);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__mask_load_unaligned(
    const void* __address,
    const __simd_mask_type<_DesiredType_>    _Mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_loadu_epi64(_mm512_setzero_si512(), _Mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_loadu_epi32(_mm512_setzero_si512(), _Mask, __address));

    else
        return _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _LoadUnaligned<_VectorType_>(__address),
            _SimdBroadcastZeros<__generation, __register_policy, _VectorType_>(), _Mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__mask_load_aligned(
    const void* __address,
    const __simd_mask_type<_DesiredType_>    _Mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_load_epi64(_mm512_setzero_si512(), _Mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_load_epi32(_mm512_setzero_si512(), _Mask, __address));

    else
        return _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _LoadAligned<_VectorType_>(__address),
            _SimdBroadcastZeros<__generation, __register_policy, _VectorType_>(), _Mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__mask_load_unaligned(
    const void* __address,
    const _MaskVectorType_  _Mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8 || sizeof(_DesiredType_) == 4) {
        return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address,
            __simd_to_mask<__generation, __register_policy, _DesiredType_>(_Mask));
    }
    else {
        return _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _LoadAligned<_VectorType_>(__address),
            _SimdBroadcastZeros<__generation, __register_policy, _VectorType_>(),
            __intrin_bitcast<_VectorType_>(_Mask));
    }
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__mask_load_aligned(
    const void* __address,
    const _MaskVectorType_  _Mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8 || sizeof(_DesiredType_) == 4) {
        return __mask_load_aligned<_VectorType_, _DesiredType_>(__address,
            __simd_to_mask<__generation, __register_policy, _DesiredType_>(_Mask));
    }
    else {
        return _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _LoadAligned<_VectorType_>(__address),
            _SimdBroadcastZeros<__generation, __register_policy, _VectorType_>(),
            __intrin_bitcast<_VectorType_>(_Mask));
    }
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__mask_load_unaligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    _VectorType_                            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_loadu_epi64(__intrin_bitcast<__m512i>(__additional_source), _Mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_loadu_epi32(__intrin_bitcast<__m512i>(__additional_source), _Mask, __address));

    else
        return _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _LoadUnaligned<_VectorType_>(__address), __additional_source, _Mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__mask_load_aligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    _VectorType_                            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_load_epi64(__intrin_bitcast<__m512i>(__additional_source), _Mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_load_epi32(__intrin_bitcast<__m512i>(__additional_source), _Mask, __address));

    else
        return _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _LoadAligned<_VectorType_>(__address), __additional_source, _Mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__mask_load_unaligned(
    const void*             __address,
    const _MaskVectorType_  _Mask,
    _VectorType_            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) >= 4)
        return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address,
            __simd_to_mask<__generation, __register_policy, _DesiredType_>(_Mask), __additional_source);
    else
        return _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _LoadUnaligned<_VectorType_>(__address),
            __additional_source,
            __intrin_bitcast<_VectorType_>(_Mask));
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__mask_load_aligned(
    const void*             __address,
    const _MaskVectorType_  _Mask,
    _VectorType_            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) >= 4)
        return __mask_load_aligned<_VectorType_, _DesiredType_>(__address,
            __simd_to_mask<__generation, __register_policy, _DesiredType_>(_Mask), __additional_source);
    else
        return _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _LoadAligned<_VectorType_>(__address),
            __additional_source,
            __intrin_bitcast<_VectorType_>(_Mask));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__compress_store_unaligned(
    _DesiredType_*                      __address,
    __simd_mask_type<_DesiredType_>      _Mask,
    _VectorType_                        _Vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        const auto _Not = uint8(~_Mask);

        const auto _Compressed = _mm512_mask_compress_epi64(__intrin_bitcast<__m512i>(_Vector), _Not, __intrin_bitcast<__m512i>(_Vector));
        _mm512_storeu_si512(__address, _Compressed);

        algorithm::__advance_bytes(__address, (math::PopulationCount(_Not) << 3));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        const auto _Not = uint16(~_Mask);

        const auto _Compressed = _mm512_mask_compress_epi32(__intrin_bitcast<__m512i>(_Vector), _Not, __intrin_bitcast<__m512i>(_Vector));
        _mm512_storeu_si512(__address, _Compressed);

        algorithm::__advance_bytes(__address, (math::PopulationCount(_Not) << 2));
    }
    else {
        constexpr auto _Length = sizeof(_VectorType_) / sizeof(_DesiredType_);
        _DesiredType_ _Source[_Length];

        _SimdStoreUnaligned<__generation, __register_policy>(_Source, _Vector);
        auto _Start = __address;

        for (auto _Index = 0; _Index < _Length; ++_Index)
            if (!((_Mask >> _Index) & 1))
                *__address++ = _Source[_Index];

        const auto _Bytes = (__address - _Start);
        std::memcpy(__address, _Source + _Bytes, sizeof(_VectorType_) - _Bytes);
    }

    return __address;
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::AVX512F, numeric::zmm512>::__compress_store_aligned(
    _DesiredType_* __address,
    __simd_mask_type<_DesiredType_>      _Mask,
    _VectorType_                        _Vector) noexcept
{
    return __compress_store_unaligned<_DesiredType_>(__address, _Mask, _Vector);
}


template <typename _Type_>
simd_stl_always_inline auto __simd_memory_access<arch::CpuFeature::AVX512BW, numeric::zmm512>::_MakeTailMask(uint32 _Bytes) noexcept {
    const auto _Elements = _Bytes / sizeof(_Type_);
    return (_Elements == 0) ? 0 : (static_cast<__simd_mask_type<_Type_>>((1ull << _Elements) - 1));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512BW, numeric::zmm512>::__mask_store_unaligned(
    void* __address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    const _VectorType_                      _Vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        _mm512_mask_storeu_epi64(__address, _Mask, __intrin_bitcast<__m512i>(_Vector));

    else if constexpr (sizeof(_DesiredType_) == 4)
        _mm512_mask_storeu_epi32(__address, _Mask, __intrin_bitcast<__m512i>(_Vector));

    else if constexpr (sizeof(_DesiredType_) == 2)
        _mm512_mask_storeu_epi16(__address, _Mask, __intrin_bitcast<__m512i>(_Vector));

    else if constexpr (sizeof(_DesiredType_) == 1)
        _mm512_mask_storeu_epi8(__address, _Mask, __intrin_bitcast<__m512i>(_Vector));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512BW, numeric::zmm512>::__mask_store_aligned(
    void* __address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    const _VectorType_                      _Vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        _mm512_mask_store_epi64(__address, _Mask, __intrin_bitcast<__m512i>(_Vector));

    else if constexpr (sizeof(_DesiredType_) == 4)
        _mm512_mask_store_epi32(__address, _Mask, __intrin_bitcast<__m512i>(_Vector));

    else
        return __mask_store_unaligned<_DesiredType_>(__address, _Mask, _Vector);
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512BW, numeric::zmm512>::__mask_store_unaligned(
    void* __address,
    const _MaskVectorType_  _Mask,
    const _VectorType_      _Vector) noexcept
{
    return __mask_store_unaligned<_DesiredType_>(__address, __simd_to_mask<
        __generation, __register_policy, _DesiredType_>(_Mask), _Vector);
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512BW, numeric::zmm512>::__mask_store_aligned(
    void* __address,
    const _MaskVectorType_  _Mask,
    const _VectorType_      _Vector) noexcept
{
    return __mask_store_unaligned<_DesiredType_>(__address, __simd_to_mask<
        __generation, __register_policy, _DesiredType_>(_Mask), _Vector);
}


template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512BW, numeric::zmm512>::__mask_load_unaligned(
    const void* __address,
    const __simd_mask_type<_DesiredType_>    _Mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_loadu_epi64(_mm512_setzero_si512(), _Mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_loadu_epi32(_mm512_setzero_si512(), _Mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 2)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_loadu_epi16(_mm512_setzero_si512(), _Mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 1)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_loadu_epi8(_mm512_setzero_si512(), _Mask, __address));
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512BW, numeric::zmm512>::__mask_load_aligned(
    const void* __address,
    const __simd_mask_type<_DesiredType_>    _Mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_load_epi64(_mm512_setzero_si512(), _Mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_load_epi32(_mm512_setzero_si512(), _Mask, __address));

    else
        return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address, _Mask);
}


template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512BW, numeric::zmm512>::__mask_load_unaligned(
    const void* __address,
    const _MaskVectorType_  _Mask) noexcept
{
    return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address,
        __simd_to_mask<__generation, __register_policy, _DesiredType_>(_Mask));
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512BW, numeric::zmm512>::__mask_load_aligned(
    const void* __address,
    const _MaskVectorType_  _Mask) noexcept
{
    return __mask_load_aligned<_VectorType_, _DesiredType_>(__address,
        __simd_to_mask<__generation, __register_policy, _DesiredType_>(_Mask));
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512BW, numeric::zmm512>::__mask_load_unaligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    _VectorType_                            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_loadu_epi64(__intrin_bitcast<__m512i>(__additional_source), _Mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_loadu_epi32(__intrin_bitcast<__m512i>(__additional_source), _Mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 2)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_loadu_epi16(__intrin_bitcast<__m512i>(__additional_source), _Mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 1)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_loadu_epi8(__intrin_bitcast<__m512i>(__additional_source), _Mask, __address));
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512BW, numeric::zmm512>::__mask_load_aligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    _VectorType_                            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_load_epi64(__intrin_bitcast<__m512i>(__additional_source), _Mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_load_epi32(__intrin_bitcast<__m512i>(__additional_source), _Mask, __address));

    else
        return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address, _Mask, __additional_source);
}


template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512BW, numeric::zmm512>::__mask_load_unaligned(
    const void*             _Address,
    const _MaskVectorType_  _Mask,
    _VectorType_            __additional_source) noexcept
{
    return __mask_load_unaligned<_VectorType_, _DesiredType_>(_Address,
        __simd_to_mask<__generation, __register_policy, _DesiredType_>(_Mask), __additional_source);
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512BW, numeric::zmm512>::__mask_load_aligned(
    const void*             _Address,
    const _MaskVectorType_  _Mask,
    _VectorType_            __additional_source) noexcept
{
    return __mask_load_aligned<_VectorType_, _DesiredType_>(_Address,
        __simd_to_mask<__generation, __register_policy, _DesiredType_>(_Mask), __additional_source);
}

template <typename _Type_>
simd_stl_always_inline auto __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::ymm256>::_MakeTailMask(uint32 _Bytes) noexcept {
    const auto _Elements = _Bytes / sizeof(_Type_);
    return (_Elements == 0) ? 0 : (static_cast<__simd_mask_type<_Type_>>((1ull << _Elements) - 1));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::ymm256>::__mask_store_unaligned(
    void* __address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    const _VectorType_                      _Vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        _mm256_mask_storeu_epi64(__address, _Mask, __intrin_bitcast<__m256i>(_Vector));

    else if constexpr (sizeof(_DesiredType_) == 4)
        _mm256_mask_storeu_epi32(__address, _Mask, __intrin_bitcast<__m256i>(_Vector));

    else
        _StoreUnaligned(__address, _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _Vector, _LoadUnaligned<_VectorType_>(__address), _Mask));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::ymm256>::__mask_store_aligned(
    void* __address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    const _VectorType_                      _Vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        _mm256_mask_store_epi64(__address, _Mask, __intrin_bitcast<__m256i>(_Vector));

    else if constexpr (sizeof(_DesiredType_) == 4)
        _mm256_mask_store_epi32(__address, _Mask, __intrin_bitcast<__m256i>(_Vector));

    else
        _StoreAligned(__address, _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _Vector, _LoadAligned<_VectorType_>(__address), _Mask));
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::ymm256>::__mask_store_unaligned(
    void* __address,
    const _MaskVectorType_  _Mask,
    const _VectorType_      _Vector) noexcept
{
    return __mask_store_unaligned<_DesiredType_>(__address, __simd_to_mask<
        __generation, __register_policy, _DesiredType_>(_Mask), _Vector);
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::ymm256>::__mask_store_aligned(
    void* __address,
    const _MaskVectorType_  _Mask,
    const _VectorType_      _Vector) noexcept
{
    return __mask_store_unaligned<_DesiredType_>(__address, __simd_to_mask<
        __generation, __register_policy, _DesiredType_>(_Mask), _Vector);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::ymm256>::__mask_load_unaligned(
    const void* __address,
    const __simd_mask_type<_DesiredType_>    _Mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_loadu_epi64(_mm256_setzero_si256(), _Mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_loadu_epi32(_mm256_setzero_si256(), _Mask, __address));

    else
        return _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _LoadUnaligned<_VectorType_>(__address),
            _SimdBroadcastZeros<__generation, __register_policy, _VectorType_>(), _Mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::ymm256>::__mask_load_aligned(
    const void* __address,
    const __simd_mask_type<_DesiredType_>    _Mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_load_epi64(_mm256_setzero_si256(), _Mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_load_epi32(_mm256_setzero_si256(), _Mask, __address));

    else
        return _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _LoadAligned<_VectorType_>(__address),
            _SimdBroadcastZeros<__generation, __register_policy, _VectorType_>(), _Mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::ymm256>::__mask_load_unaligned(
    const void* __address,
    const _MaskVectorType_  _Mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8 || sizeof(_DesiredType_) == 4) {
        return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address,
            __simd_to_mask<__generation, __register_policy, _DesiredType_>(_Mask));
    }
    else {
        return _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _LoadAligned<_VectorType_>(__address),
            _SimdBroadcastZeros<__generation, __register_policy, _VectorType_>(),
            __intrin_bitcast<_VectorType_>(_Mask));
    }
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::ymm256>::__mask_load_aligned(
    const void* __address,
    const _MaskVectorType_  _Mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8 || sizeof(_DesiredType_) == 4) {
        return __mask_load_aligned<_VectorType_, _DesiredType_>(__address,
            __simd_to_mask<__generation, __register_policy, _DesiredType_>(_Mask));
    }
    else {
        return _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _LoadAligned<_VectorType_>(__address),
            _SimdBroadcastZeros<__generation, __register_policy, _VectorType_>(),
            __intrin_bitcast<_VectorType_>(_Mask));
    }
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::ymm256>::__mask_load_unaligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    _VectorType_                            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_loadu_epi64(__intrin_bitcast<__m256i>(__additional_source), _Mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_loadu_epi32(__intrin_bitcast<__m256i>(__additional_source), _Mask, __address));

    else
        return _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _LoadUnaligned<_VectorType_>(__address), __additional_source, _Mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::ymm256>::__mask_load_aligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    _VectorType_                            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_load_epi64(__intrin_bitcast<__m256i>(__additional_source), _Mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_load_epi32(__intrin_bitcast<__m256i>(__additional_source), _Mask, __address));

    else
        return _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _LoadAligned<_VectorType_>(__address), __additional_source, _Mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::ymm256>::__mask_load_unaligned(
    const void*             __address,
    const _MaskVectorType_  _Mask,
    _VectorType_            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) >= 4)
        return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address,
            __simd_to_mask<__generation, __register_policy, _DesiredType_>(_Mask), __additional_source);
    else
        return _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _LoadUnaligned<_VectorType_>(__address),
            __additional_source,
            __intrin_bitcast<_VectorType_>(_Mask));
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::ymm256>::__mask_load_aligned(
    const void*             __address,
    const _MaskVectorType_  _Mask,
    _VectorType_            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) >= 4)
        return __mask_load_aligned<_VectorType_, _DesiredType_>(__address,
            __simd_to_mask<__generation, __register_policy, _DesiredType_>(_Mask), __additional_source);
    else
        return _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _LoadAligned<_VectorType_>(__address),
            __additional_source,
            __intrin_bitcast<_VectorType_>(_Mask));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::ymm256>::__compress_store_unaligned(
    _DesiredType_*                      __address,
    __simd_mask_type<_DesiredType_>      _Mask,
    _VectorType_                        _Vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        const auto _Not = uint8(~_Mask);

        const auto _Compressed = _mm256_mask_compress_epi64(__intrin_bitcast<__m256i>(_Vector), _Not, __intrin_bitcast<__m256i>(_Vector));
        _SimdStoreUnaligned<__generation, __register_policy>(__address, _Compressed);

        algorithm::__advance_bytes(__address, (math::PopulationCount(_Not) << 3));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        const auto _Not = uint16(~_Mask);

        const auto _Compressed = _mm256_mask_compress_epi32(__intrin_bitcast<__m256i>(_Vector), _Not, __intrin_bitcast<__m256i>(_Vector));
        _SimdStoreUnaligned<__generation, __register_policy>(__address, _Compressed);

        algorithm::__advance_bytes(__address, (math::PopulationCount(_Not) << 2));
    }
    else {
        constexpr auto _Length = sizeof(_VectorType_) / sizeof(_DesiredType_);
        _DesiredType_ _Source[_Length];

        _SimdStoreUnaligned<__generation, __register_policy>(_Source, _Vector);
        auto _Start = __address;

        for (auto _Index = 0; _Index < _Length; ++_Index)
            if (!((_Mask >> _Index) & 1))
                *__address++ = _Source[_Index];

        const auto _Bytes = (__address - _Start);
        std::memcpy(__address, _Source + _Bytes, sizeof(_VectorType_) - _Bytes);
    }

    return __address;
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::ymm256>::__compress_store_aligned(
    _DesiredType_* __address,
    __simd_mask_type<_DesiredType_>      _Mask,
    _VectorType_                        _Vector) noexcept
{
    return __compress_store_unaligned<_DesiredType_>(__address, _Mask, _Vector);
}


template <typename _Type_>
simd_stl_always_inline auto __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::ymm256>::_MakeTailMask(uint32 _Bytes) noexcept {
    const auto _Elements = _Bytes / sizeof(_Type_);
    return (_Elements == 0) ? 0 : (static_cast<__simd_mask_type<_Type_>>((1ull << _Elements) - 1));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::ymm256>::__mask_store_unaligned(
    void* __address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    const _VectorType_                      _Vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        _mm256_mask_storeu_epi64(__address, _Mask, __intrin_bitcast<__m256i>(_Vector));

    else if constexpr (sizeof(_DesiredType_) == 4)
        _mm256_mask_storeu_epi32(__address, _Mask, __intrin_bitcast<__m256i>(_Vector));

    else if constexpr (sizeof(_DesiredType_) == 2)
        _mm256_mask_storeu_epi16(__address, _Mask, __intrin_bitcast<__m256i>(_Vector));

    else if constexpr (sizeof(_DesiredType_) == 1)
        _mm256_mask_storeu_epi8(__address, _Mask, __intrin_bitcast<__m256i>(_Vector));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::ymm256>::__mask_store_aligned(
    void* __address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    const _VectorType_                      _Vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        _mm256_mask_store_epi64(__address, _Mask, __intrin_bitcast<__m256i>(_Vector));

    else if constexpr (sizeof(_DesiredType_) == 4)
        _mm256_mask_store_epi32(__address, _Mask, __intrin_bitcast<__m256i>(_Vector));

    else
        return __mask_store_unaligned<_DesiredType_>(__address, _Mask, _Vector);
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::ymm256>::__mask_store_unaligned(
    void* __address,
    const _MaskVectorType_  _Mask,
    const _VectorType_      _Vector) noexcept
{
    return __mask_store_unaligned<_DesiredType_>(__address, __simd_to_mask<
        __generation, __register_policy, _DesiredType_>(_Mask), _Vector);
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::ymm256>::__mask_store_aligned(
    void* __address,
    const _MaskVectorType_  _Mask,
    const _VectorType_      _Vector) noexcept
{
    return __mask_store_unaligned<_DesiredType_>(__address, __simd_to_mask<
        __generation, __register_policy, _DesiredType_>(_Mask), _Vector);
}


template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::ymm256>::__mask_load_unaligned(
    const void* __address,
    const __simd_mask_type<_DesiredType_>    _Mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_loadu_epi64(_mm256_setzero_si256(), _Mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_loadu_epi32(_mm256_setzero_si256(), _Mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 2)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_loadu_epi16(_mm256_setzero_si256(), _Mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 1)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_loadu_epi8(_mm256_setzero_si256(), _Mask, __address));
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::ymm256>::__mask_load_aligned(
    const void* __address,
    const __simd_mask_type<_DesiredType_>    _Mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_load_epi64(_mm256_setzero_si256(), _Mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_load_epi32(_mm256_setzero_si256(), _Mask, __address));

    else
        return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address, _Mask);
}


template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::ymm256>::__mask_load_unaligned(
    const void* __address,
    const _MaskVectorType_  _Mask) noexcept
{
    return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address,
        __simd_to_mask<__generation, __register_policy, _DesiredType_>(_Mask));
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::ymm256>::__mask_load_aligned(
    const void* __address,
    const _MaskVectorType_  _Mask) noexcept
{
    return __mask_load_aligned<_VectorType_, _DesiredType_>(__address,
        __simd_to_mask<__generation, __register_policy, _DesiredType_>(_Mask));
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::ymm256>::__mask_load_unaligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    _VectorType_                            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_loadu_epi64(__intrin_bitcast<__m256i>(__additional_source), _Mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_loadu_epi32(__intrin_bitcast<__m256i>(__additional_source), _Mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 2)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_loadu_epi16(__intrin_bitcast<__m256i>(__additional_source), _Mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 1)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_loadu_epi8(__intrin_bitcast<__m256i>(__additional_source), _Mask, __address));
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::ymm256>::__mask_load_aligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    _VectorType_                            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_load_epi64(__intrin_bitcast<__m256i>(__additional_source), _Mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_load_epi32(__intrin_bitcast<__m256i>(__additional_source), _Mask, __address));

    else
        return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address, _Mask, __additional_source);
}


template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::ymm256>::__mask_load_unaligned(
    const void*             _Address,
    const _MaskVectorType_  _Mask,
    _VectorType_            __additional_source) noexcept
{
    return __mask_load_unaligned<_VectorType_, _DesiredType_>(_Address,
        __simd_to_mask<__generation, __register_policy, _DesiredType_>(_Mask), __additional_source);
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::ymm256>::__mask_load_aligned(
    const void*             _Address,
    const _MaskVectorType_  _Mask,
    _VectorType_            __additional_source) noexcept
{
    return __mask_load_aligned<_VectorType_, _DesiredType_>(_Address,
        __simd_to_mask<__generation, __register_policy, _DesiredType_>(_Mask), __additional_source);
}


template <typename _Type_>
simd_stl_always_inline auto __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::xmm128>::_MakeTailMask(uint32 _Bytes) noexcept {
    const auto _Elements = _Bytes / sizeof(_Type_);
    return (_Elements == 0) ? 0 : (static_cast<__simd_mask_type<_Type_>>((1ull << _Elements) - 1));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::xmm128>::__mask_store_unaligned(
    void* __address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    const _VectorType_                      _Vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        _mm_mask_storeu_epi64(__address, _Mask, __intrin_bitcast<__m128i>(_Vector));

    else if constexpr (sizeof(_DesiredType_) == 4)
        _mm_mask_storeu_epi32(__address, _Mask, __intrin_bitcast<__m128i>(_Vector));

    else
        _StoreUnaligned(__address, _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _Vector, _LoadUnaligned<_VectorType_>(__address), _Mask));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::xmm128>::__mask_store_aligned(
    void* __address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    const _VectorType_                      _Vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        _mm_mask_store_epi64(__address, _Mask, __intrin_bitcast<__m128i>(_Vector));

    else if constexpr (sizeof(_DesiredType_) == 4)
        _mm_mask_store_epi32(__address, _Mask, __intrin_bitcast<__m128i>(_Vector));

    else
        _StoreAligned(__address, _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _Vector, _LoadAligned<_VectorType_>(__address), _Mask));
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::xmm128>::__mask_store_unaligned(
    void* __address,
    const _MaskVectorType_  _Mask,
    const _VectorType_      _Vector) noexcept
{
    return __mask_store_unaligned<_DesiredType_>(__address, __simd_to_mask<
        __generation, __register_policy, _DesiredType_>(_Mask), _Vector);
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::xmm128>::__mask_store_aligned(
    void* __address,
    const _MaskVectorType_  _Mask,
    const _VectorType_      _Vector) noexcept
{
    return __mask_store_unaligned<_DesiredType_>(__address, __simd_to_mask<
        __generation, __register_policy, _DesiredType_>(_Mask), _Vector);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::xmm128>::__mask_load_unaligned(
    const void* __address,
    const __simd_mask_type<_DesiredType_>    _Mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm_mask_loadu_epi64(_mm_setzero_si128(), _Mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm_mask_loadu_epi32(_mm_setzero_si128(), _Mask, __address));

    else
        return _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _LoadUnaligned<_VectorType_>(__address),
            _SimdBroadcastZeros<__generation, __register_policy, _VectorType_>(), _Mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::xmm128>::__mask_load_aligned(
    const void* __address,
    const __simd_mask_type<_DesiredType_>    _Mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm_mask_load_epi64(_mm_setzero_si128(), _Mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm_mask_load_epi32(_mm_setzero_si128(), _Mask, __address));

    else
        return _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _LoadAligned<_VectorType_>(__address),
            _SimdBroadcastZeros<__generation, __register_policy, _VectorType_>(), _Mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::xmm128>::__mask_load_unaligned(
    const void* __address,
    const _MaskVectorType_  _Mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8 || sizeof(_DesiredType_) == 4) {
        return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address,
            __simd_to_mask<__generation, __register_policy, _DesiredType_>(_Mask));
    }
    else {
        return _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _LoadAligned<_VectorType_>(__address),
            _SimdBroadcastZeros<__generation, __register_policy, _VectorType_>(),
            __intrin_bitcast<_VectorType_>(_Mask));
    }
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::xmm128>::__mask_load_aligned(
    const void* __address,
    const _MaskVectorType_  _Mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8 || sizeof(_DesiredType_) == 4) {
        return __mask_load_aligned<_VectorType_, _DesiredType_>(__address,
            __simd_to_mask<__generation, __register_policy, _DesiredType_>(_Mask));
    }
    else {
        return _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _LoadAligned<_VectorType_>(__address),
            _SimdBroadcastZeros<__generation, __register_policy, _VectorType_>(),
            __intrin_bitcast<_VectorType_>(_Mask));
    }
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::xmm128>::__mask_load_unaligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    _VectorType_                            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm_mask_loadu_epi64(__intrin_bitcast<__m128i>(__additional_source), _Mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm_mask_loadu_epi32(__intrin_bitcast<__m128i>(__additional_source), _Mask, __address));

    else
        return _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _LoadUnaligned<_VectorType_>(__address), __additional_source, _Mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::xmm128>::__mask_load_aligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    _VectorType_                            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm_mask_load_epi64(__intrin_bitcast<__m128i>(__additional_source), _Mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm_mask_load_epi32(__intrin_bitcast<__m128i>(__additional_source), _Mask, __address));

    else
        return _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _LoadAligned<_VectorType_>(__address), __additional_source, _Mask);
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::xmm128>::__mask_load_unaligned(
    const void*             __address,
    const _MaskVectorType_  _Mask,
    _VectorType_            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) >= 4)
        return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address,
            __simd_to_mask<__generation, __register_policy, _DesiredType_>(_Mask), __additional_source);
    else
        return _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _LoadUnaligned<_VectorType_>(__address),
            __additional_source,
            __intrin_bitcast<_VectorType_>(_Mask));
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::xmm128>::__mask_load_aligned(
    const void*             __address,
    const _MaskVectorType_  _Mask,
    _VectorType_            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) >= 4)
        return __mask_load_aligned<_VectorType_, _DesiredType_>(__address,
            __simd_to_mask<__generation, __register_policy, _DesiredType_>(_Mask), __additional_source);
    else
        return _SimdBlend<__generation, __register_policy, _DesiredType_>(
            _LoadAligned<_VectorType_>(__address),
            __additional_source,
            __intrin_bitcast<_VectorType_>(_Mask));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::xmm128>::__compress_store_unaligned(
    _DesiredType_*                      __address,
    __simd_mask_type<_DesiredType_>      _Mask,
    _VectorType_                        _Vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8) {
        const auto _Not = uint8(~_Mask);

        const auto _Compressed = _mm_mask_compress_epi64(__intrin_bitcast<__m128i>(_Vector), _Not, __intrin_bitcast<__m128i>(_Vector));
        _SimdStoreUnaligned<__generation, __register_policy>(__address, _Compressed);

        algorithm::__advance_bytes(__address, (math::PopulationCount(_Not) << 3));
    }
    else if constexpr (sizeof(_DesiredType_) == 4) {
        const auto _Not = uint16(~_Mask);

        const auto _Compressed = _mm_mask_compress_epi32(__intrin_bitcast<__m128i>(_Vector), _Not, __intrin_bitcast<__m128i>(_Vector));
        _SimdStoreUnaligned<__generation, __register_policy>(__address, _Compressed);

        algorithm::__advance_bytes(__address, (math::PopulationCount(_Not) << 2));
    }
    else {
        constexpr auto _Length = sizeof(_VectorType_) / sizeof(_DesiredType_);
        _DesiredType_ _Source[_Length];

        _SimdStoreUnaligned<__generation, __register_policy>(_Source, _Vector);
        auto _Start = __address;

        for (auto _Index = 0; _Index < _Length; ++_Index)
            if (!((_Mask >> _Index) & 1))
                *__address++ = _Source[_Index];

        const auto _Bytes = (__address - _Start);
        std::memcpy(__address, _Source + _Bytes, sizeof(_VectorType_) - _Bytes);
    }

    return __address;
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::AVX512VLF, numeric::xmm128>::__compress_store_aligned(
    _DesiredType_* __address,
    __simd_mask_type<_DesiredType_>      _Mask,
    _VectorType_                        _Vector) noexcept
{
    return __compress_store_unaligned<_DesiredType_>(__address, _Mask, _Vector);
}

template <typename _Type_>
simd_stl_always_inline auto __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::xmm128>::_MakeTailMask(uint32 _Bytes) noexcept {
    const auto _Elements = _Bytes / sizeof(_Type_);
    return (_Elements == 0) ? 0 : (static_cast<__simd_mask_type<_Type_>>((1ull << _Elements) - 1));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::xmm128>::__mask_store_unaligned(
    void* __address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    const _VectorType_                      _Vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        _mm_mask_storeu_epi64(__address, _Mask, __intrin_bitcast<__m128i>(_Vector));

    else if constexpr (sizeof(_DesiredType_) == 4)
        _mm_mask_storeu_epi32(__address, _Mask, __intrin_bitcast<__m128i>(_Vector));

    else if constexpr (sizeof(_DesiredType_) == 2)
        _mm_mask_storeu_epi16(__address, _Mask, __intrin_bitcast<__m128i>(_Vector));

    else if constexpr (sizeof(_DesiredType_) == 1)
        _mm_mask_storeu_epi8(__address, _Mask, __intrin_bitcast<__m128i>(_Vector));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::xmm128>::__mask_store_aligned(
    void* __address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    const _VectorType_                      _Vector) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        _mm_mask_store_epi64(__address, _Mask, __intrin_bitcast<__m128i>(_Vector));

    else if constexpr (sizeof(_DesiredType_) == 4)
        _mm_mask_store_epi32(__address, _Mask, __intrin_bitcast<__m128i>(_Vector));

    else
        return __mask_store_unaligned<_DesiredType_>(__address, _Mask, _Vector);
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::xmm128>::__mask_store_unaligned(
    void* __address,
    const _MaskVectorType_  _Mask,
    const _VectorType_      _Vector) noexcept
{
    return __mask_store_unaligned<_DesiredType_>(__address, __simd_to_mask<
        __generation, __register_policy, _DesiredType_>(_Mask), _Vector);
}

template <
    typename _DesiredType_,
    typename _MaskVectorType_,
    typename _VectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::xmm128>::__mask_store_aligned(
    void* __address,
    const _MaskVectorType_  _Mask,
    const _VectorType_      _Vector) noexcept
{
    return __mask_store_unaligned<_DesiredType_>(__address, __simd_to_mask<
        __generation, __register_policy, _DesiredType_>(_Mask), _Vector);
}


template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::xmm128>::__mask_load_unaligned(
    const void* __address,
    const __simd_mask_type<_DesiredType_>    _Mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm_mask_loadu_epi64(_mm_setzero_si128(), _Mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm_mask_loadu_epi32(_mm_setzero_si128(), _Mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 2)
        return __intrin_bitcast<_VectorType_>(_mm_mask_loadu_epi16(_mm_setzero_si128(), _Mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 1)
        return __intrin_bitcast<_VectorType_>(_mm_mask_loadu_epi8(_mm_setzero_si128(), _Mask, __address));
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::xmm128>::__mask_load_aligned(
    const void* __address,
    const __simd_mask_type<_DesiredType_>    _Mask) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm_mask_load_epi64(_mm_setzero_si128(), _Mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm_mask_load_epi32(_mm_setzero_si128(), _Mask, __address));

    else
        return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address, _Mask);
}


template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::xmm128>::__mask_load_unaligned(
    const void* __address,
    const _MaskVectorType_  _Mask) noexcept
{
    return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address,
        __simd_to_mask<__generation, __register_policy, _DesiredType_>(_Mask));
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::xmm128>::__mask_load_aligned(
    const void* __address,
    const _MaskVectorType_  _Mask) noexcept
{
    return __mask_load_aligned<_VectorType_, _DesiredType_>(__address,
        __simd_to_mask<__generation, __register_policy, _DesiredType_>(_Mask));
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::xmm128>::__mask_load_unaligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    _VectorType_                            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm_mask_loadu_epi64(__intrin_bitcast<__m128i>(__additional_source), _Mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm_mask_loadu_epi32(__intrin_bitcast<__m128i>(__additional_source), _Mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 2)
        return __intrin_bitcast<_VectorType_>(_mm_mask_loadu_epi16(__intrin_bitcast<__m128i>(__additional_source), _Mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 1)
        return __intrin_bitcast<_VectorType_>(_mm_mask_loadu_epi8(__intrin_bitcast<__m128i>(__additional_source), _Mask, __address));
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::xmm128>::__mask_load_aligned(
    const void*                             __address,
    const __simd_mask_type<_DesiredType_>    _Mask,
    _VectorType_                            __additional_source) noexcept
{
    if constexpr (sizeof(_DesiredType_) == 8)
        return __intrin_bitcast<_VectorType_>(_mm_mask_load_epi64(__intrin_bitcast<__m128i>(__additional_source), _Mask, __address));

    else if constexpr (sizeof(_DesiredType_) == 4)
        return __intrin_bitcast<_VectorType_>(_mm_mask_load_epi32(__intrin_bitcast<__m128i>(__additional_source), _Mask, __address));

    else
        return __mask_load_unaligned<_VectorType_, _DesiredType_>(__address, _Mask, __additional_source);
}


template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::xmm128>::__mask_load_unaligned(
    const void*             _Address,
    const _MaskVectorType_  _Mask,
    _VectorType_            __additional_source) noexcept
{
    return __mask_load_unaligned<_VectorType_, _DesiredType_>(_Address,
        __simd_to_mask<__generation, __register_policy, _DesiredType_>(_Mask), __additional_source);
}

template <
    typename _VectorType_,
    typename _DesiredType_,
    typename _MaskVectorType_,
    std::enable_if_t<__is_intrin_type_v<_MaskVectorType_>, int>>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, numeric::xmm128>::__mask_load_aligned(
    const void*             _Address,
    const _MaskVectorType_  _Mask,
    _VectorType_            __additional_source) noexcept
{
    return __mask_load_aligned<_VectorType_, _DesiredType_>(_Address,
        __simd_to_mask<__generation, __register_policy, _DesiredType_>(_Mask), __additional_source);
}

#pragma endregion

__SIMD_STL_NUMERIC_NAMESPACE_END
