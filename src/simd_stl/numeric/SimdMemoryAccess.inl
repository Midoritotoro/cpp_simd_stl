#pragma once 

__SIMD_STL_NUMERIC_NAMESPACE_BEGIN

#pragma region Sse2-Sse4.2 memory access 

template <
    int32 __first_,
    int32 _Second_>
static constexpr int32 __constexpr_max() noexcept {
    return (__first_ > _Second_) ? __first_ : _Second_;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    class               _RegisterPolicy_,
    class               _DesiredType_,
    class               _MaskTypeTo_,
    class               _MaskTypeFrom_>
simd_stl_always_inline _MaskTypeTo_ __mask_convert(_MaskTypeFrom_ __from) noexcept {
    if constexpr (std::is_integral_v<_MaskTypeFrom_> && std::is_integral_v<_MaskTypeTo_>)
        return static_cast<_MaskTypeTo_>(__from);

    else if constexpr (__is_intrin_type_v<_MaskTypeFrom_> && __is_intrin_type_v<_MaskTypeTo_>)
        return __intrin_bitcast<_MaskTypeTo_>(__from);

    else if constexpr (__is_intrin_type_v<_MaskTypeFrom_> && std::is_integral_v<_MaskTypeTo_>)
        return __simd_to_mask<_SimdGeneration_, _RegisterPolicy_, _DesiredType_>(__from);

    else if constexpr (std::is_integral_v<_MaskTypeFrom_> && __is_intrin_type_v<_MaskTypeTo_>)
        return __simd_to_vector<_SimdGeneration_, _RegisterPolicy_, _MaskTypeTo_, _DesiredType_>(__from);
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE2, xmm128>
    ::__non_temporal_load(const void* __address) noexcept
{
    return __load<_VectorType_>(__address, __aligned_policy{});
}

template <typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::SSE2, xmm128>
    ::__non_temporal_store(
        void*           __address,
        _VectorType_    __vector) noexcept
{
    _mm_stream_si128(static_cast<__m128i*>(__address), __intrin_bitcast<__m128i>(__vector));
}

simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::SSE2, xmm128>::__streaming_fence() noexcept {
    return _mm_sfence();
}

template <
    typename    _VectorType_,
    class       _AlignmentPolicy_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE2, xmm128>::__load(
    const void* __address,
    _AlignmentPolicy_&&) noexcept 
{
    if constexpr (std::remove_cvref<_AlignmentPolicy_>::__alignment) {
        if      constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_load_si128(reinterpret_cast<const __m128i*>(__address));

        else if constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_load_pd(reinterpret_cast<const double*>(__address));

        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_load_ps(reinterpret_cast<const float*>(__address));
    }
    else {
        if      constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_loadu_si128(reinterpret_cast<const __m128i*>(__address));

        else if constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_loadu_pd(reinterpret_cast<const double*>(__address));

        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_loadu_ps(reinterpret_cast<const float*>(__address));
    }
}

template <
    typename    _VectorType_,
    class       _AlignmentPolicy_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::SSE2, xmm128>::__store(
    void*           __address,
    _VectorType_    __vector,
    _AlignmentPolicy_&&) noexcept
{
    if constexpr (std::remove_cvref<_AlignmentPolicy_>::__alignment) {
        if      constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_store_si128(reinterpret_cast<__m128i*>(__address), __vector);

        else if constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_store_pd(reinterpret_cast<double*>(__address), __vector);

        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_store_ps(reinterpret_cast<float*>(__address), __vector);
    }
    else {
        if      constexpr (std::is_same_v<_VectorType_, __m128i>)
            return _mm_storeu_si128(reinterpret_cast<__m128i*>(__address), __vector);

        else if constexpr (std::is_same_v<_VectorType_, __m128d>)
            return _mm_storeu_pd(reinterpret_cast<double*>(__address), __vector);

        else if constexpr (std::is_same_v<_VectorType_, __m128>)
            return _mm_storeu_ps(reinterpret_cast<float*>(__address), __vector);
    }
}

template <
    typename    _DesiredType_,
    typename    _VectorType_,
    class       _MaskType_,
    class       _AlignmentPolicy_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::SSE2, xmm128>::__mask_store(
    void*               __address,
    _MaskType_          __mask,
    _VectorType_        __vector,
    _AlignmentPolicy_&& __policy) noexcept
{
    __store(__address, __simd_blend<__generation, __register_policy, _DesiredType_>(
        __vector, __load<_VectorType_>(__address, __policy), 
        __mask_convert<__generation, __register_policy, _VectorType_>(__mask)), __policy);
}

template <
    typename    _DesiredType_,
    typename    _VectorType_,
    class       _MaskType_,
    class       _AlignmentPolicy_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE2, xmm128>::__mask_load(
    const void*         __address,
    _MaskType_          __mask,
    _AlignmentPolicy_&& __policy) noexcept
{
    return __mask_load<_DesiredType_>(__address, __mask, __simd_broadcast_zeros<__generation, __register_policy, _VectorType_>(), __policy);
}

template <
    typename    _DesiredType_,
    typename    _VectorType_,
    class       _MaskType_,
    class       _AlignmentPolicy_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE2, xmm128>::__mask_load(
    const void*         __address,
    _MaskType_          __mask,
    _VectorType_        __additional_source,
    _AlignmentPolicy_&& __policy) noexcept
{
    return __simd_blend<__generation, __register_policy, _DesiredType_>(__load<_VectorType_>(__address),
        __additional_source, __mask_convert<__generation, __register_policy, _VectorType_>(__mask));
}

template <
    typename    _DesiredType_,
    typename    _VectorType_,
    class       _MaskType_,
    class       _AlignmentPolicy_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::SSE2, xmm128>::__compress_store(
    _DesiredType_*      __address,
    _MaskType_          __mask,
    _VectorType_        __vector,
    _AlignmentPolicy_&& __policy) noexcept
{
    const auto __compressed = __simd_compress<__generation, __register_policy, _DesiredType_>(__vector, __mask);
    __store(__address, __compressed.second, __policy);

    algorithm::__advance_bytes(__address, __compressed.first);
    return __address;
}

template <typename _Type_>
simd_stl_always_inline __m128i __simd_memory_access<arch::CpuFeature::SSE2, xmm128>::__make_tail_mask(uint32 __bytes) noexcept {
    constexpr uint8 __tail_mask[8] = { ~0u, ~0u, ~0u, ~0u, 0, 0, 0, 0 };
    return __load<__m128i>(reinterpret_cast<const uint8*>(__tail_mask) + (16 - __bytes));
}

template <
    typename    _VectorType_,
    class       _AlignmentPolicy_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE3, xmm128>::__load(
    const void*         __address,
    _AlignmentPolicy_&& __policy) noexcept 
{
    if constexpr (std::remove_cvref<_AlignmentPolicy_>::__alignment == false && std::is_same_v<_VectorType_, __m128i>)
        return _mm_lddqu_si128(reinterpret_cast<const __m128i*>(__address));
    else
        return __simd_load<arch::CpuFeature::SSE2, __register_policy, _VectorType_>(__address, __policy);
}

template <
    typename    _DesiredType_,
    typename    _VectorType_,
    class       _MaskType_,
    class       _AlignmentPolicy_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::SSSE3, xmm128>::__compress_store(
    _DesiredType_*      __address,
    _MaskType_          __mask,
    _VectorType_        __vector,
    _AlignmentPolicy_&& __policy) noexcept
{
    const auto __compressed = __simd_compress<__generation, __register_policy, _DesiredType_>(__vector, __mask);
    __store(__address, __compressed.second, __policy);

    algorithm::__advance_bytes(__address, __compressed.first);
    return __address;
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE41, xmm128>::__non_temporal_load(const void* __address) noexcept {
    return __intrin_bitcast<_VectorType_>(_mm_stream_load_si128(reinterpret_cast<const __m128i*>(__address)));
}

template <
    typename    _DesiredType_,
    typename    _VectorType_,
    class       _MaskType_,
    class       _AlignmentPolicy_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::SSE41, xmm128>::__mask_store(
    void*               __address,
    _MaskType_          __mask,
    _VectorType_        __vector,
    _AlignmentPolicy_&& __policy) noexcept
{
    __store(__address, __simd_blend<__generation, __register_policy, _DesiredType_>(__vector,
        __load<_VectorType_>(__address, __policy), __mask_convert<__generation, __register_policy, _VectorType_>(__mask)), __policy);
}

template <
    typename    _DesiredType_,
    typename    _VectorType_,
    class       _MaskType_,
    class       _AlignmentPolicy_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE41, xmm128>::__mask_load(
    const void*         __address,
    _MaskType_          __mask,
    _AlignmentPolicy_&& __policy) noexcept
{
    return __mask_load<_DesiredType_>(__address, __mask, __simd_broadcast_zeros<__generation, __register_policy, _VectorType_>(), __policy);
}

template <
    typename    _DesiredType_,
    typename    _VectorType_,
    class       _MaskType_,
    class       _AlignmentPolicy_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::SSE41, xmm128>::__mask_load(
    const void*         __address,
    _MaskType_          __mask,
    _VectorType_        __additional_source,
    _AlignmentPolicy_&& __policy) noexcept
{
    return __simd_blend<__generation, __register_policy, _DesiredType_>(__load<_VectorType_>(__address, __policy), 
        __additional_source, __mask_convert<__generation, __register_policy, _VectorType_>(__mask));
}

#pragma endregion

#pragma region Avx-Avx2 memory access

template <
    typename    _VectorType_,
    class       _AlignmentPolicy_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX2, ymm256>::__load(
    const void* __address,
    _AlignmentPolicy_&&) noexcept 
{
    if constexpr (std::remove_cvref<_AlignmentPolicy_>::__alignment) {
        if      constexpr (std::is_same_v<_VectorType_, __m256i>)
            return _mm256_load_si256(reinterpret_cast<const __m256i*>(__address));

        else if constexpr (std::is_same_v<_VectorType_, __m256d>)
            return _mm256_load_pd(reinterpret_cast<const double*>(__address));

        else if constexpr (std::is_same_v<_VectorType_, __m256>)
            return _mm256_load_ps(reinterpret_cast<const float*>(__address));
    }
    else {
        if      constexpr (std::is_same_v<_VectorType_, __m256i>)
            return _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(__address));

        else if constexpr (std::is_same_v<_VectorType_, __m256d>)
            return _mm256_loadu_pd(reinterpret_cast<const double*>(__address));

        else if constexpr (std::is_same_v<_VectorType_, __m256>)
            return _mm256_loadu_ps(reinterpret_cast<const float*>(__address));
    }
}

template <
    typename    _VectorType_,
    class       _AlignmentPolicy_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX2, ymm256>::__store(
    void*           __address,
    _VectorType_    __vector,
    _AlignmentPolicy_&&) noexcept
{
    if constexpr (std::remove_cvref<_AlignmentPolicy_>::__alignment) {
        if      constexpr (std::is_same_v<_VectorType_, __m256i>)
            return _mm256_store_si256(reinterpret_cast<__m256i*>(__address), __vector);

        else if constexpr (std::is_same_v<_VectorType_, __m256)
            return _mm256_store_pd(reinterpret_cast<double*>(__address), __vector);

        else if constexpr (std::is_same_v<_VectorType_, __m256>)
            return _mm256_store_ps(reinterpret_cast<float*>(__address), __vector);
    }
    else {
        if      constexpr (std::is_same_v<_VectorType_, __m256i>)
            return _mm256_storeu_si256(reinterpret_cast<__m256i*>(__address), __vector);

        else if constexpr (std::is_same_v<_VectorType_, __m256d>)
            return _mm256_storeu_pd(reinterpret_cast<double*>(__address), __vector);

        else if constexpr (std::is_same_v<_VectorType_, __m256>)
            return _mm256_storeu_ps(reinterpret_cast<float*>(__address), __vector);
    }
}

template <
    typename    _DesiredType_,
    typename    _VectorType_,
    class       _MaskType_,
    class       _AlignmentPolicy_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX2, ymm256>::__mask_load(
    const void*         __address,
    _MaskType_          __mask,
    _AlignmentPolicy_&& __policy) noexcept
{
    if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_maskload_epi64(reinterpret_cast<const long long*>(__address),
            __mask_convert<__generation, __register_policy, _DesiredType_, __m256i>(__mask)));

    else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_maskload_epi32(reinterpret_cast<const int*>(__address),
            __mask_convert<__generation, __register_policy, _DesiredType_, __m256i>(__mask)));

    else if constexpr (__is_ps_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_maskload_ps(reinterpret_cast<const float*>(__address),
            __mask_convert<__generation, __register_policy, _DesiredType_, __m256i>(__mask)));

    else if constexpr (__is_pd_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm256_maskload_ps(reinterpret_cast<const double*>(__address),
            __mask_convert<__generation, __register_policy, _DesiredType_, __m256i>(__mask)));

    else
        return __mask_load<_DesiredType_>(__address, __mask, __simd_broadcast_zeros<__generation, __register_policy, _VectorType_>(), __policy);
}


template <
    typename    _DesiredType_,
    typename    _VectorType_,
    class       _MaskType_,
    class       _AlignmentPolicy_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX2, ymm256>::__mask_load(
    const void*         __address,
    _MaskType_          __mask,
    _VectorType_        __additional_source,
    _AlignmentPolicy_&& __policy) noexcept
{
    return __simd_blend<__generation, __register_policy, _DesiredType_>(__load<_VectorType_>(__address, __policy), 
        __additional_source, __mask_convert<__generation, __register_policy, _VectorType_>(__mask));
}

template <
    typename    _DesiredType_,
    typename    _VectorType_,
    class       _MaskType_,
    class       _AlignmentPolicy_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX2, ymm256>::__mask_store(
    void*               __address,
    _MaskType_          __mask,
    _VectorType_        __vector,
    _AlignmentPolicy_&& __policy) noexcept
{
    if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>)
        _mm256_maskstore_epi64(reinterpret_cast<long long*>(__address),
            __mask_convert<__generation, __register_policy, _DesiredType_, __m256i>(__mask),
            __intrin_bitcast<__m256i>(__vector));

    else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>)
        _mm256_maskstore_epi32(reinterpret_cast<int*>(__address),
            __mask_convert<__generation, __register_policy, _DesiredType_, __m256i>(__mask),
            __intrin_bitcast<__m256i>(__vector));

    else if constexpr (__is_pd_v<_DesiredType_>)
        _mm256_maskstore_pd(reinterpret_cast<double*>(__address),
            __mask_convert<__generation, __register_policy, _DesiredType_, __m256i>(__mask),
            __intrin_bitcast<__m256d>(__vector));

    else if constexpr (__is_ps_v<_DesiredType_>)
        _mm256_maskstore_ps(reinterpret_cast<float*>(__address),
            __mask_convert<__generation, __register_policy, _DesiredType_, __m256i>(__mask),
            __intrin_bitcast<__m256>(__vector));

    else
        __store(__address, __simd_blend<__generation, __register_policy, _DesiredType_>(__vector, __load<_VectorType_>(__address, __policy), __mask));
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX2, ymm256>::__non_temporal_load(const void* __address) noexcept {
    return __intrin_bitcast<_VectorType_>(_mm256_stream_load_si256(reinterpret_cast<const __m256i*>(__address)));
}

template <typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX2, ymm256>::__non_temporal_store(
    void* __address,
    _VectorType_    __vector) noexcept
{
    _mm256_stream_si256(reinterpret_cast<__m256i*>(__address), __intrin_bitcast<__m256i>(__vector));
}

simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX2, ymm256>::__streaming_fence() noexcept {
    return _mm_sfence();
}

template <
    typename    _DesiredType_,
    typename    _VectorType_,
    class       _MaskType_,
    class       _AlignmentPolicy_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::AVX2, ymm256>::__compress_store(
    _DesiredType_*      __address,
    _MaskType_          __mask,
    _VectorType_        __vector,
    _AlignmentPolicy_&& __policy) noexcept
{
    const auto __compressed = __simd_compress<__generation, __register_policy, _DesiredType_>(__vector, __mask);
    __store(__address, __compressed.second, __policy);

    algorithm::__advance_bytes(__address, __compressed.first);
    return __address;
}

#pragma endregion

#pragma region Avx512 memory access

    
template <typename _Type_>
simd_stl_always_inline auto __simd_memory_access<arch::CpuFeature::AVX512F, zmm512>::__make_tail_mask(uint32 __bytes) noexcept {
    return (__bytes == 0) ? 0 : (static_cast<__simd_mask_type<_Type_>>((1ull << (__bytes / sizeof(_Type_))) - 1));
}

template <typename _VectorType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512F, zmm512>::__non_temporal_load(const void* __address) noexcept {
    return __intrin_bitcast<_VectorType_>(_mm512_stream_load_si512(__address));
}

template <typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512F, zmm512>::__non_temporal_store(
    void*           __address,
    _VectorType_    __vector) noexcept
{
    _mm512_stream_si512(__address, __intrin_bitcast<__m512i>(__vector));
}

simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512F, zmm512>::__streaming_fence() noexcept {
    return _mm_sfence();
}

template <
    typename    _VectorType_,
    class       _AlignmentPolicy_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512F, zmm512>::__load(
    const void* __address,
    _AlignmentPolicy_&&) noexcept 
{
    if constexpr (std::remove_cvref_t<_AlignmentPolicy_>::__alignment) {
        if      constexpr (std::is_same_v<_VectorType_, __m512i>)
            return _mm512_load_si512(__address);

        else if constexpr (std::is_same_v<_VectorType_, __m512d>)
            return _mm512_load_pd(__address);

        else if constexpr (std::is_same_v<_VectorType_, __m512>)
            return _mm512_load_ps(__address);
    }
    else {
        if      constexpr (std::is_same_v<_VectorType_, __m512i>)
            return _mm512_loadu_si512(__address);

        else if constexpr (std::is_same_v<_VectorType_, __m512d>)
            return _mm512_loadu_pd(__address);

        else if constexpr (std::is_same_v<_VectorType_, __m512>)
            return _mm512_loadu_ps(__address);
    }
}

template <
    typename    _VectorType_,
    class       _AlignmentPolicy_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512F, zmm512>::__store(
    void*           __address,
    _VectorType_    __vector,
    _AlignmentPolicy_&&) noexcept
{
    if constexpr (std::remove_cvref_t<_AlignmentPolicy_>::__alignment) {
        if      constexpr (std::is_same_v<_VectorType_, __m512i>)
            return _mm512_store_si512(__address, __vector);

        else if constexpr (std::is_same_v<_VectorType_, __m512d>)
            return _mm512_store_pd(__address, __vector);

        else if constexpr (std::is_same_v<_VectorType_, __m512>)
            return _mm512_store_ps(__address, __vector);
    }
    else {
        if      constexpr (std::is_same_v<_VectorType_, __m512i>)
            return _mm512_storeu_si512(__address, __vector);

        else if constexpr (std::is_same_v<_VectorType_, __m512d>)
            return _mm512_storeu_pd(__address, __vector);

        else if constexpr (std::is_same_v<_VectorType_, __m512>)
            return _mm512_storeu_ps(__address, __vector);
    }
}

template <
    typename    _DesiredType_,
    typename    _VectorType_,
    class       _MaskType_,
    class       _AlignmentPolicy_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512F, zmm512>::__mask_store(
    void*               __address,
    _MaskType_          __mask,
    _VectorType_        __vector,
    _AlignmentPolicy_&& __policy) noexcept
{
    if constexpr (std::remove_cvref_t<_AlignmentPolicy_>::__alignment) {
        if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>)
            return _mm512_mask_store_epi64(__address, __mask_convert<__generation, __register_policy, _DesiredType_, uint8>(__mask), __intrin_bitcast<__m512i>(__vector));

        else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>)
            return _mm512_mask_store_epi32(__address, __mask_convert<__generation, __register_policy, _DesiredType_, uint16>(__mask), __intrin_bitcast<__m512i>(__vector));

        else if constexpr (__is_pd_v<_DesiredType_>)
            return _mm512_mask_store_pd(__address, __mask_convert<__generation, __register_policy, _DesiredType_, uint8>(__mask), __intrin_bitcast<__m512d>(__vector));

        else if constexpr (__is_ps_v<_DesiredType_>)
            return _mm512_mask_store_ps(__address, __mask_convert<__generation, __register_policy, _DesiredType_, uint16>(__mask), __intrin_bitcast<__m512>(__vector));
    }
    else {
        if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>)
            return _mm512_mask_storeu_epi64(__address, __mask_convert<__generation, __register_policy, _DesiredType_, uint8>(__mask), __intrin_bitcast<__m512i>(__vector));

        else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>)
            return _mm512_mask_storeu_epi32(__address, __mask_convert<__generation, __register_policy, _DesiredType_, uint16>(__mask), __intrin_bitcast<__m512i>(__vector));

        else if constexpr (__is_pd_v<_DesiredType_>)
            return _mm512_mask_storeu_pd(__address, __mask_convert<__generation, __register_policy, _DesiredType_, uint8>(__mask), __intrin_bitcast<__m512d>(__vector));

        else if constexpr (__is_ps_v<_DesiredType_>)
            return _mm512_mask_storeu_ps(__address, __mask_convert<__generation, __register_policy, _DesiredType_, uint16>(__mask), __intrin_bitcast<__m512>(__vector));
    }

    __store(__address, __simd_blend<__generation, __register_policy, _DesiredType_>(__vector, __load<_VectorType_>(__address, __policy),
        __mask_convert<__generation, __register_policy, _DesiredType_, _VectorType_>(__mask)), __policy);
}

template <
    typename    _DesiredType_,
    typename    _VectorType_,
    class       _MaskType_,
    class       _AlignmentPolicy_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512F, zmm512>::__mask_load(
    const void* __address,
    _MaskType_          __mask,
    _AlignmentPolicy_&& __policy) noexcept
{
    if constexpr (std::remove_cvref_t<_AlignmentPolicy_>::__alignment) {
        if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>)
            return __intrin_bitcast<_VectorType_>(_mm512_maskz_load_epi64(__mask_convert<__generation, __register_policy, _DesiredType_, uint8>(__mask), __address));

        else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>)
            return __intrin_bitcast<_VectorType_>(_mm512_maskz_load_epi32(__mask_convert<__generation, __register_policy, _DesiredType_, uint16>(__mask), __address));

        else if constexpr (__is_pd_v<_DesiredType_>)
            return __intrin_bitcast<_VectorType_>(_mm512_maskz_load_pd(__mask_convert<__generation, __register_policy, _DesiredType_, uint8>(__mask), __address));

        else if constexpr (__is_ps_v<_DesiredType_>)
            return __intrin_bitcast<_VectorType_>(_mm512_maskz_load_ps(__mask_convert<__generation, __register_policy, _DesiredType_, uint16>(__mask), __address));
    }
    else {
        if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>)
            return __intrin_bitcast<_VectorType_>(_mm512_maskz_loadu_epi64(__mask_convert<__generation, __register_policy, _DesiredType_, uint8>(__mask), __address));

        else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>)
            return __intrin_bitcast<_VectorType_>(_mm512_maskz_loadu_epi32(__mask_convert<__generation, __register_policy, _DesiredType_, uint16>(__mask), __address));

        else if constexpr (__is_pd_v<_DesiredType_>)
            return __intrin_bitcast<_VectorType_>(_mm512_maskz_loadu_pd(__mask_convert<__generation, __register_policy, _DesiredType_, uint8>(__mask), __address));

        else if constexpr (__is_ps_v<_DesiredType_>)
            return __intrin_bitcast<_VectorType_>(_mm512_maskz_loadu_ps(__mask_convert<__generation, __register_policy, _DesiredType_, uint16>(__mask), __address));
    }

    return __mask_load<_DesiredType_>(__address, __mask, __simd_broadcast_zeros<__generation, __register_policy, _VectorType_>(), __policy);
}

template <
    typename    _DesiredType_,
    typename    _VectorType_,
    class       _MaskType_,
    class       _AlignmentPolicy_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512F, zmm512>::__mask_load(
    const void*         __address,
    _MaskType_          __mask,
    _VectorType_        __additional_source,
    _AlignmentPolicy_&& __policy) noexcept
{
    if constexpr (std::remove_cvref_t<_AlignmentPolicy_>::__alignment) { 
        if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>)
            return __intrin_bitcast<_VectorType_>(_mm512_mask_load_epi64(__intrin_bitcast<__m512i>(__additional_source),
                __mask_convert<__generation, __register_policy, _DesiredType_, uint8>(__mask), __address));

        else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>)
            return __intrin_bitcast<_VectorType_>(_mm512_mask_load_epi32(__intrin_bitcast<__m512i>(__additional_source),
                __mask_convert<__generation, __register_policy, _DesiredType_, uint16>(__mask), __address));

        else if constexpr (__is_pd_v<_DesiredType_>)
            return __intrin_bitcast<_VectorType_>(_mm512_mask_load_pd(__intrin_bitcast<__m512d>(__additional_source),
                __mask_convert<__generation, __register_policy, _DesiredType_, uint8>(__mask), __address));

        else if constexpr (__is_ps_v<_DesiredType_>)
            return __intrin_bitcast<_VectorType_>(_mm512_mask_load_ps(__intrin_bitcast<__m512>(__additional_source),
                __mask_convert<__generation, __register_policy, _DesiredType_, uint16>(__mask), __address));
    }
    else {
        if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>)
            return __intrin_bitcast<_VectorType_>(_mm512_mask_loadu_epi64(__intrin_bitcast<__m512i>(__additional_source),
                __mask_convert<__generation, __register_policy, _DesiredType_, uint8>(__mask), __address));

        else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>)
            return __intrin_bitcast<_VectorType_>(_mm512_mask_loadu_epi32(__intrin_bitcast<__m512i>(__additional_source),
                __mask_convert<__generation, __register_policy, _DesiredType_, uint16>(__mask), __address));

        else if constexpr (__is_pd_v<_DesiredType_>)
            return __intrin_bitcast<_VectorType_>(_mm512_mask_loadu_pd(__intrin_bitcast<__m512d>(__additional_source),
                __mask_convert<__generation, __register_policy, _DesiredType_, uint8>(__mask), __address));

        else if constexpr (__is_ps_v<_DesiredType_>)
            return __intrin_bitcast<_VectorType_>(_mm512_mask_loadu_ps(__intrin_bitcast<__m512>(__additional_source),
                __mask_convert<__generation, __register_policy, _DesiredType_, uint16>(__mask), __address));
    }

    return __simd_blend<__generation, __register_policy, _DesiredType_>(__load<_VectorType_>(__address, __policy),
        __additional_source, __mask_convert<__generation, __register_policy, _VectorType_>(__mask));
}

template <
    typename    _DesiredType_,
    typename    _VectorType_,
    class       _MaskType_,
    class       _AlignmentPolicy_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::AVX512F, zmm512>::__compress_store(
    _DesiredType_*      __address,
    _MaskType_          __mask,
    _VectorType_        __vector,
    _AlignmentPolicy_&& __policy) noexcept
{
    const auto __compressed = __simd_compress<__generation, __register_policy, _DesiredType_>(__vector, __mask);
    __store(__address, __compressed.second, __policy);

    algorithm::__advance_bytes(__address, __compressed.first);
    return __address;
}

template <
    typename    _DesiredType_,
    typename    _VectorType_,
    class       _MaskType_,
    class       _AlignmentPolicy_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512BW, zmm512>::__mask_store(
    void*               __address,
    _MaskType_          __mask,
    _VectorType_        __vector,
    _AlignmentPolicy_&& __policy) noexcept
{
    if constexpr (__is_epi16_v<_DesiredType_> || __is_epu16_v<_DesiredType_>)
        return _mm512_mask_storeu_epi16(__address, __mask_convert<__generation, __register_policy, _DesiredType_, uint32>(__mask), __intrin_bitcast<__m512i>(__vector));

    else if constexpr (__is_epi8_v<_DesiredType_> || __is_epu8_v<_DesiredType_>)
        return _mm512_mask_storeu_epi8(__address, __mask_convert<__generation, __register_policy, _DesiredType_, uint64>(__mask), __intrin_bitcast<__m512i>(__vector));

    else
        return __simd_mask_store<arch::CpuFeature::AVX512F, __register_policy, _DesiredType_>(__address, __mask, __vector, __policy);
}


template <
    typename    _DesiredType_,
    typename    _VectorType_,
    class       _MaskType_,
    class       _AlignmentPolicy_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512BW, zmm512>::__mask_load(
    const void*         __address,
    _MaskType_          __mask,
    _AlignmentPolicy_&& __policy) noexcept
{
    if constexpr (__is_epi16_v<_DesiredType_> || __is_epu16_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm512_maskz_loadu_epi16(__mask_convert<__generation, __register_policy, _DesiredType_, uint32>(__mask), __address));

    else if constexpr (__is_epi8_v<_DesiredType_> || __is_epu8_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm512_maskz_loadu_epi8(__mask_convert<__generation, __register_policy, _DesiredType_, uint64>(__mask), __address));

    else
        return __simd_mask_load<arch::CpuFeature::AVX512F, __register_policy, _DesiredType_>(__address, __mask, __policy);
}

template <
    typename    _DesiredType_,
    typename    _VectorType_,
    class       _MaskType_,
    class       _AlignmentPolicy_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512BW, zmm512>::__mask_load(
    const void*         __address,
    _MaskType_          __mask,
    _VectorType_        __additional_source,
    _AlignmentPolicy_&& __policy) noexcept
{
    if constexpr (__is_epi16_v<_DesiredType_> || __is_epu16_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_loadu_epi16(__intrin_bitcast<__m512i>(__additional_source),
            __mask_convert<__generation, __register_policy, _DesiredType_, uint32>(__mask), __address));

    else if constexpr (__is_epi8_v<_DesiredType_> || __is_epu8_v<_DesiredType_>)
        return __intrin_bitcast<_VectorType_>(_mm512_mask_loadu_epi8(__intrin_bitcast<__m512i>(__additional_source),
            __mask_convert<__generation, __register_policy, _DesiredType_, uint64>(__mask), __address));

    else
        return __simd_mask_load<arch::CpuFeature::AVX512F, __register_policy, _DesiredType_>(__address, __mask, __additional_source, __policy);
}

template <typename _Type_>
simd_stl_always_inline auto __simd_memory_access<arch::CpuFeature::AVX512VLF, ymm256>::__make_tail_mask(uint32 __bytes) noexcept {
    return (__bytes == 0) ? 0 : (static_cast<__simd_mask_type<_Type_>>((1ull << (__bytes / sizeof(_Type_))) - 1));
}

template <
    typename    _DesiredType_,
    typename    _VectorType_,
    class       _MaskType_,
    class       _AlignmentPolicy_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLF, ymm256>::__mask_store(
    void*               __address,
    _MaskType_          __mask,
    _VectorType_        __vector,
    _AlignmentPolicy_&& __policy) noexcept
{
    if constexpr (__is_intrin_type_v<_MaskType_>) {
        return __simd_mask_store<arch::CpuFeature::AVX2, __register_policy, _DesiredType_>(__address, __mask, __vector, __policy);
    }
    else {
        if constexpr (std::remove_cvref_t<_AlignmentPolicy_>::__alignment) {
            if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>)
                return _mm256_mask_store_epi64(__address, static_cast<uint8>(__mask), __intrin_bitcast<__m256i>(__vector));

            else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>)
                return _mm256_mask_store_epi32(__address, static_cast<uint8>(__mask), __intrin_bitcast<__m256i>(__vector));

            else if constexpr (__is_pd_v<_DesiredType_>)
                return _mm256_mask_store_pd(__address, static_cast<uint8>(__mask), __intrin_bitcast<__m256d>(__vector));

            else if constexpr (__is_ps_v<_DesiredType_>)
                return _mm256_mask_store_ps(__address, static_cast<uint8>(__mask), __intrin_bitcast<__m256>(__vector));
        }
        else {
            if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>)
                return _mm256_mask_storeu_epi64(__address, static_cast<uint8>(__mask), __intrin_bitcast<__m256i>(__vector));

            else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>)
                return _mm256_mask_storeu_epi32(__address, static_cast<uint8>(__mask), __intrin_bitcast<__m256i>(__vector));

            else if constexpr (__is_pd_v<_DesiredType_>)
                return _mm256_mask_storeu_pd(__address, static_cast<uint8>(__mask), __intrin_bitcast<__m256d>(__vector));

            else if constexpr (__is_ps_v<_DesiredType_>)
                return _mm256_mask_storeu_ps(__address, static_cast<uint8>(__mask), __intrin_bitcast<__m256>(__vector));
        }
    }

    __store(__address, __simd_blend<__generation, __register_policy, _DesiredType_>(__vector, __load<_VectorType_>(__address, __policy),
        __mask_convert<__generation, __register_policy, _DesiredType_, _VectorType_>(__mask)), __policy);
}

template <
    typename    _DesiredType_,
    typename    _VectorType_,
    class       _MaskType_,
    class       _AlignmentPolicy_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, ymm256>::__mask_load(
    const void*         __address,
    _MaskType_          __mask,
    _AlignmentPolicy_&& __policy) noexcept
{
    if constexpr (__is_intrin_type_v<_MaskType_>) {
        return __simd_mask_load<arch::CpuFeature::AVX2, __register_policy, _DesiredType_>(__address, __mask, __policy);
    }
    else {
        if constexpr (std::remove_cvref_t<_AlignmentPolicy_>::__alignment) {
            if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>)
                return __intrin_bitcast<_VectorType_>(_mm256_maskz_load_epi64(static_cast<uint8>(__mask), __address));

            else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>)
                return __intrin_bitcast<_VectorType_>(_mm256_maskz_load_epi32(static_cast<uint16>(__mask), __address));

            else if constexpr (__is_pd_v<_DesiredType_>)
                return __intrin_bitcast<_VectorType_>(_mm256_maskz_load_pd(static_cast<uint8>(__mask), __address));

            else if constexpr (__is_ps_v<_DesiredType_>)
                return __intrin_bitcast<_VectorType_>(_mm256_maskz_load_ps(static_cast<uint16>(__mask), __address));
        }
        else {
            if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>)
                return __intrin_bitcast<_VectorType_>(_mm256_maskz_loadu_epi64(static_cast<uint8>(__mask), __address));

            else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>)
                return __intrin_bitcast<_VectorType_>(_mm256_maskz_loadu_epi32(static_cast<uint16>(__mask), __address));

            else if constexpr (__is_pd_v<_DesiredType_>)
                return __intrin_bitcast<_VectorType_>(_mm256_maskz_loadu_pd(static_cast<uint8>(__mask), __address));

            else if constexpr (__is_ps_v<_DesiredType_>)
                return __intrin_bitcast<_VectorType_>(_mm256_maskz_loadu_ps(static_cast<uint16>(__mask), __address));
        }
    }

    return __mask_load<_DesiredType_>(__address, __mask, __simd_broadcast_zeros<__generation, __register_policy, _VectorType_>(), __policy);
}

template <
    typename    _DesiredType_,
    typename    _VectorType_,
    class       _MaskType_,
    class       _AlignmentPolicy_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, ymm256>::__mask_load(
    const void*         __address,
    _MaskType_          __mask,
    _VectorType_        __additional_source,
    _AlignmentPolicy_&& __policy) noexcept
{
    if constexpr (__is_intrin_type_v<_MaskType_>) {
        return __simd_mask_load<arch::CpuFeature::AVX2, __register_policy, _DesiredType_>(__address, __mask, __additional_source, __policy);
    }
    else {
        if constexpr (std::remove_cvref_t<_AlignmentPolicy_>::__alignment) {
            if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>)
                return __intrin_bitcast<_VectorType_>(_mm256_mask_load_epi64(__intrin_bitcast<__m256i>(__additional_source), static_cast<uint8>(__mask), __address));

            else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>)
                return __intrin_bitcast<_VectorType_>(_mm256_mask_load_epi32(__intrin_bitcast<__m256i>(__additional_source), static_cast<uint8>(__mask), __address));

            else if constexpr (__is_pd_v<_DesiredType_>)
                return __intrin_bitcast<_VectorType_>(_mm256_mask_load_pd(__intrin_bitcast<__m256d>(__additional_source), static_cast<uint8>(__mask), __address));

            else if constexpr (__is_ps_v<_DesiredType_>)
                return __intrin_bitcast<_VectorType_>(_mm256_mask_load_ps(__intrin_bitcast<__m256>(__additional_source), static_cast<uint8>(__mask), __address));
        }
        else {
            if constexpr (__is_epi64_v<_DesiredType_> || __is_epu64_v<_DesiredType_>)
                return __intrin_bitcast<_VectorType_>(_mm256_mask_loadu_epi64(__intrin_bitcast<__m256i>(__additional_source), static_cast<uint8>(__mask), __address));

            else if constexpr (__is_epi32_v<_DesiredType_> || __is_epu32_v<_DesiredType_>)
                return __intrin_bitcast<_VectorType_>(_mm256_mask_loadu_epi32(__intrin_bitcast<__m256i>(__additional_source), static_cast<uint8>(__mask), __address));

            else if constexpr (__is_pd_v<_DesiredType_>)
                return __intrin_bitcast<_VectorType_>(_mm256_mask_loadu_pd(__intrin_bitcast<__m256d>(__additional_source), static_cast<uint8>(__mask), __address));

            else if constexpr (__is_ps_v<_DesiredType_>)
                return __intrin_bitcast<_VectorType_>(_mm256_mask_loadu_ps(__intrin_bitcast<__m256>(__additional_source), static_cast<uint8>(__mask), __address));
        }
    }

    return __simd_blend<__generation, __register_policy, _DesiredType_>(__load<_VectorType_>(__address, __policy), __additional_source, __mask);
}

template <
    typename    _DesiredType_,
    typename    _VectorType_,
    class       _MaskType_,
    class       _AlignmentPolicy_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::AVX512VLF, ymm256>::__compress_store(
    _DesiredType_*      __address,
    _MaskType_          __mask,
    _VectorType_        __vector,
    _AlignmentPolicy_&& __policy) noexcept
{
    const auto __compressed = __simd_compress<__generation, __register_policy, _DesiredType_>(__vector, __mask);
    __store(__address, __compressed.second);

    algorithm::__advance_bytes(__address, __compressed.first);
    return __address;
}

template <
    typename    _DesiredType_,
    typename    _VectorType_,
    class       _MaskType_,
    class       _AlignmentPolicy_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLBW, ymm256>::__mask_store(
    void*               __address,
    _MaskType_          __mask,
    _VectorType_        __vector,
    _AlignmentPolicy_&& __policy) noexcept
{
    if constexpr (sizeof(_DesiredType_) >= 4)
        return __simd_mask_store<arch::CpuFeature::AVX512VLF, __register_policy, _DesiredType_>(__address, __mask, __vector, __policy);

    else if constexpr (sizeof(_DesiredType_) == 2)
        return _mm256_mask_storeu_epi16(__address, __mask_convert<__generation, __register_policy, _DesiredType_, uint16>(__mask), __intrin_bitcast<__m256i>(__vector));

    else if constexpr (sizeof(_DesiredType_) == 1)
        return _mm256_mask_storeu_epi8(__address, __mask_convert<__generation, __register_policy, _DesiredType_, uint32>(__mask), __intrin_bitcast<__m256i>(__vector));
    
}

template <
    typename    _DesiredType_,
    typename    _VectorType_,
    class       _MaskType_,
    class       _AlignmentPolicy_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, ymm256>::__mask_load(
    const void*         __address,
    _MaskType_          __mask,
    _AlignmentPolicy_&& __policy) noexcept
{
    if constexpr (sizeof(_DesiredType_) >= 4)
        return __simd_mask_load<arch::CpuFeature::AVX512VLF, __register_policy, _DesiredType_>(__address, __mask, __policy);

    else if constexpr (sizeof(_DesiredType_) == 2)
        return __intrin_bitcast<_VectorType_>(_mm256_maskz_loadu_epi16(__mask_convert<__generation, __register_policy, _DesiredType_, uint16>(__mask), __address));

    else if constexpr (sizeof(_DesiredType_) == 1)
        return __intrin_bitcast<_VectorType_>(_mm256_maskz_loadu_epi8(__mask_convert<__generation, __register_policy, _DesiredType_, uint32>(__mask), __address));
}

template <
    typename    _DesiredType_,
    typename    _VectorType_,
    class       _MaskType_,
    class       _AlignmentPolicy_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, ymm256>::__mask_load(
    const void*         __address,
    _MaskType_          __mask,
    _VectorType_        __additional_source,
    _AlignmentPolicy_&& __policy) noexcept
{
    if constexpr (sizeof(_DesiredType_) >= 4)
        return __simd_mask_load<arch::CpuFeature::AVX512VLF, __register_policy, _DesiredType_>(__address, __mask, __additional_source, __policy);

    else if constexpr (sizeof(_DesiredType_) == 2)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_loadu_epi16(__intrin_bitcast<__m256i>(__additional_source), __mask_convert<__generation, __register_policy, _DesiredType_, uint16>(__mask), __address));

    else if constexpr (sizeof(_DesiredType_) == 1)
        return __intrin_bitcast<_VectorType_>(_mm256_mask_loadu_epi8(__intrin_bitcast<__m256i>(__additional_source), __mask_convert<__generation, __register_policy, _DesiredType_, uint32>(__mask), __address));
}

template <typename _Type_>
simd_stl_always_inline auto __simd_memory_access<arch::CpuFeature::AVX512VLF, xmm128>::__make_tail_mask(uint32 __bytes) noexcept {
    const auto __elements = __bytes / sizeof(_Type_);
    return (__elements == 0) ? 0 : (static_cast<__simd_mask_type<_Type_>>((1ull << __elements) - 1));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLF, xmm128>::__mask_store_unaligned(
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
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLF, xmm128>::__mask_store_aligned(
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
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLF, xmm128>::__mask_store_unaligned(
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
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLF, xmm128>::__mask_store_aligned(
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
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, xmm128>::__mask_load_unaligned(
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
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, xmm128>::__mask_load_aligned(
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
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, xmm128>::__mask_load_unaligned(
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
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, xmm128>::__mask_load_aligned(
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
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, xmm128>::__mask_load_unaligned(
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
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, xmm128>::__mask_load_aligned(
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
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, xmm128>::__mask_load_unaligned(
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
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLF, xmm128>::__mask_load_aligned(
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
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::AVX512VLF, xmm128>::__compress_store_unaligned(
    _DesiredType_*                      __address,
    __simd_mask_type<_DesiredType_>     __mask,
    _VectorType_                        __vector) noexcept
{
    const auto __compressed = __simd_compress<__generation, __register_policy, _DesiredType_>(__vector, __mask);
    __store_unaligned(__address, __compressed.second);

    algorithm::__advance_bytes(__address, __compressed.first);
    return __address;
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline _DesiredType_* __simd_memory_access<arch::CpuFeature::AVX512VLF, xmm128>::__compress_store_aligned(
    _DesiredType_*                      __address,
    __simd_mask_type<_DesiredType_>     __mask,
    _VectorType_                        __vector) noexcept
{
    const auto __compressed = __simd_compress<__generation, __register_policy, _DesiredType_>(__vector, __mask);
    __store_aligned(__address, __compressed.second);

    algorithm::__advance_bytes(__address, __compressed.first);
    return __address;
}

template <typename _Type_>
simd_stl_always_inline auto __simd_memory_access<arch::CpuFeature::AVX512VLBW, xmm128>::__make_tail_mask(uint32 __bytes) noexcept {
    return (__bytes == 0) ? 0 : (static_cast<__simd_mask_type<_Type_>>((1ull << (__bytes / sizeof(_Type_))) - 1));
}

template <
    typename _DesiredType_,
    typename _VectorType_>
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLBW, xmm128>::__mask_store_unaligned(
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
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLBW, xmm128>::__mask_store_aligned(
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
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLBW, xmm128>::__mask_store_unaligned(
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
simd_stl_always_inline void __simd_memory_access<arch::CpuFeature::AVX512VLBW, xmm128>::__mask_store_aligned(
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
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, xmm128>::__mask_load_unaligned(
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
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, xmm128>::__mask_load_aligned(
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
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, xmm128>::__mask_load_unaligned(
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
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, xmm128>::__mask_load_aligned(
    const void*             __address,
    const _MaskVectorType_  __mask) noexcept
{
    return __mask_load_aligned<_VectorType_, _DesiredType_>(__address,
        __simd_to_mask<__generation, __register_policy, _DesiredType_>(__mask));
}

template <
    typename _VectorType_,
    typename _DesiredType_>
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, xmm128>::__mask_load_unaligned(
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
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, xmm128>::__mask_load_aligned(
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
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, xmm128>::__mask_load_unaligned(
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
simd_stl_always_inline _VectorType_ __simd_memory_access<arch::CpuFeature::AVX512VLBW, xmm128>::__mask_load_aligned(
    const void*             __address,
    const _MaskVectorType_  __mask,
    _VectorType_            __additional_source) noexcept
{
    return __mask_load_aligned<_VectorType_, _DesiredType_>(__address,
        __simd_to_mask<__generation, __register_policy, _DesiredType_>(__mask), __additional_source);
}

#pragma endregion

__SIMD_STL_NUMERIC_NAMESPACE_END
