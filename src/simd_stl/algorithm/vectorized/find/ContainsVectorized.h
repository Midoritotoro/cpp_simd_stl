#pragma once

#include <src/simd_stl/numeric/SizedSimdDispatcher.h>
#include <src/simd_stl/numeric/CachePrefetcher.h>


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
struct __contains_vectorized_internal {
    template <class _CachePrefetcher_>
    simd_stl_always_inline bool operator()(
        sizetype                    __aligned_size,
        sizetype                    __tail_size,
        const void*                 __first,
        const void*                 __last,
        typename _Simd_::value_type __value,
        _CachePrefetcher_&&         __prefetcher) const noexcept
    {
        const auto __guard      = numeric::make_guard<_Simd_>();
        const auto __comparand  = _Simd_(__value);

        do {
            __prefetcher(__bytes_pointer_offset(__first, _CachePrefetcher_::__prefetch_stride));

            const auto __loaded = _Simd_::load(__first);
            const auto __mask   = __comparand.mask_compare<numeric::simd_comparison::equal>(__loaded);

            if (__mask.any_of())
                return true;

            __advance_bytes(__first, sizeof(_Simd_));
            __aligned_size -= sizeof(_Simd_);
        } while (__aligned_size != 0);

        if (__tail_size != 0) {
            if constexpr (_Simd_::template is_native_mask_load_supported_v<>) {
                const auto __tail_mask  = _Simd_::make_tail_mask(__tail_size);
                const auto __loaded     = _Simd_::mask_load(__first, __tail_mask);

                const auto __mask = typename _Simd_::mask_type(__comparand.native_compare<numeric::simd_comparison::equal>(__loaded) & __tail_mask);

                if (__mask.any_of())
                    return true;
            }
            else {
                return __contains_scalar(__first, __last, __value);
            }
        }
    }
};

template <class _Type_>
simd_stl_declare_const_function bool simd_stl_stdcall __contains_vectorized(
    const void* __first,
    const void* __last,
    _Type_      __value) noexcept
{
    const auto __fallback_args  = std::forward_as_tuple(__first, __last, __value);
    const auto __simd_args      = std::forward_as_tuple(__first, __last, __value, numeric::__cache_prefetcher<numeric::__prefetch_hint::NTA>());

    return numeric::__simd_sized_dispatcher<__contains_vectorized_internal>::__apply<_Type_>(
        __byte_length(__first, __last), &__contains_scalar<_Type_>, std::move(__simd_args), std::move(__fallback_args));
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
