#pragma once

#include <simd_stl/numeric/Simd.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
simd_stl_declare_const_function bool simd_stl_stdcall __contains_scalar(
    const void* __first,
    const void* __last,
    _Type_      __value) noexcept
{
    auto __current = static_cast<const _Type_*>(__first);

    while (__current != __last) {
        if (*__current == __value)
            return true;

        ++__current;
    }

    return false;
}

template <class _Simd_>
simd_stl_declare_const_function bool simd_stl_stdcall __contains_vectorized_internal(
    const void*                 __first,
    const void*                 __last,
    typename _Simd_::value_type __value) noexcept
{
    numeric::zero_upper_at_exit_guard<_Simd_::__generation> __guard;

    constexpr auto __is_masked_memory_access_supported = _Simd_::template is_native_mask_store_supported_v<> &&
        _Simd_::template is_native_mask_load_supported_v<>;

    const auto __size   = __byte_length(__first, __last);
    auto __aligned_size = __size & (~(sizeof(_Simd_) - 1));

    const auto __comparand = _Simd_(__value);

    while (__aligned_size != 0) {
        const auto __loaded = _Simd_::load(__first);
        const auto __mask   = __comparand.mask_compare<numeric::simd_comparison::equal>(__loaded);

        if (__mask.any_of())
            return true;

        __advance_bytes(__first, sizeof(_Simd_));
        __aligned_size -= sizeof(_Simd_);
    }

    const auto __tail_size = __size & (sizeof(_Simd_) - sizeof(typename _Simd_::value_type));

    if constexpr (__is_masked_memory_access_supported) {
        const auto __tail_size = __size & (sizeof(_Simd_) - sizeof(typename _Simd_::value_type));

        if (__tail_size != 0) {
            const auto __tail_mask  = _Simd_::make_tail_mask(__tail_size);
            const auto __loaded     = _Simd_::mask_load(__first, __tail_mask);

            const auto __compared = __comparand.native_compare<numeric::simd_comparison::equal>(__loaded) & __tail_mask;
            const auto __mask = numeric::simd_mask<_Simd_::__generation, typename _Simd_::value_type>(numeric::__simd_to_native_mask<_Simd_::__generation,
                    typename _Simd_::policy_type, std::remove_cv_t<decltype(__compared)>>(__compared));

            if (__mask.any_of())
                return true;
        }
    }
    else {
        if (__first != __last)
            return __contains_scalar(__first, __last, __value);
    }
}

template <class _Type_>
simd_stl_declare_const_function bool simd_stl_stdcall __contains_vectorized(
    const void* __first,
    const void* __last,
    _Type_      __value) noexcept
{
    if constexpr (sizeof(_Type_) <= 2) {
        if (arch::ProcessorFeatures::AVX512BW())
            return __contains_vectorized_internal<numeric::simd512_avx512bw<_Type_>>(__first, __last, __value);
    }
    else {
        if (arch::ProcessorFeatures::AVX512F())
            return __contains_vectorized_internal<numeric::simd512_avx512f<_Type_>>(__first, __last, __value);
    }

    if (arch::ProcessorFeatures::AVX2())
         return __contains_vectorized_internal<numeric::simd256_avx2<_Type_>>(__first, __last, __value);
    else if (arch::ProcessorFeatures::SSE2())
        return __contains_vectorized_internal<numeric::simd128_sse2<_Type_>>(__first, __last, __value);

    return __contains_scalar(__first, __last, __value);
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
