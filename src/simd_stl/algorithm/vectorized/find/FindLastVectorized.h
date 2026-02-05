#pragma once

#include <src/simd_stl/datapar/SizedSimdDispatcher.h>
#include <src/simd_stl/datapar/CachePrefetcher.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
simd_stl_always_inline const _Type_* simd_stl_stdcall __find_last_scalar(
    const void* __first,
    const void* __last,
    _Type_      __value) noexcept
{
    auto __current = static_cast<const _Type_*>(__last);

    while (__current != __first && *__current != __value)
        --__current;

    return __current;
}

template <class _Simd_>
struct __find_last_vectorized_internal {
    template <class _CachePrefetcher_>
    simd_stl_always_inline const typename _Simd_::value_type* simd_stl_stdcall operator()(
        sizetype                        __aligned_size,
        sizetype                        __tail_size,
        const void*                     __first,
        const void*                     __last,
        typename _Simd_::value_type     __value,
        _CachePrefetcher_&&             __prefetcher) const noexcept
    {
        const auto __guard = datapar::make_guard<_Simd_>();

        const void* __cached_last = __last;
        const auto __comparand = _Simd_(__value);

        do {
            __rewind_bytes(__last, sizeof(_Simd_));
            __aligned_size -= sizeof(_Simd_);
            
            __prefetcher(__bytes_pointer_offset(__last, -sizeof(_Simd_)));

            const auto __loaded  = _Simd_::load(__last);
            const auto __mask    = __comparand.mask_compare<datapar::simd_comparison::equal>(__loaded);

            if (__mask.any_of())
                return static_cast<const typename _Simd_::value_type*>(__last) + __mask.count_trailing_zero_bits();
        } while (__aligned_size != 0);

        if (__tail_size == 0)
            return static_cast<const typename _Simd_::value_type*>(__cached_last);

        if constexpr (_Simd_::template is_native_mask_load_supported_v<>) {
            __rewind_bytes(__last, __tail_size);

            const auto __tail_mask  = _Simd_::make_tail_mask(__tail_size);
            const auto __loaded     = _Simd_::maskz_load(__last, __tail_mask);

            const auto __mask = typename _Simd_::mask_type(__comparand.native_compare<datapar::simd_comparison::equal>(__loaded) & __tail_mask);

            if (__mask.any_of())
                return static_cast<const typename _Simd_::value_type*>(__last) + __mask.count_trailing_zero_bits() + 1;

            return static_cast<const typename _Simd_::value_type*>(__cached_last);
        }
        else {
            return __find_last_scalar(__first, __last, __value);
        }
    }

};

template <class _Type_>
simd_stl_declare_const_function _Type_* simd_stl_stdcall __find_last_vectorized(
    const void* __first,
    const void* __last,
    _Type_      __value) noexcept
{
    const auto __fallback_args  = std::forward_as_tuple(__first, __last, __value);
    const auto __simd_args      = std::forward_as_tuple(__first, __last, __value, datapar::__cache_prefetcher<datapar::__prefetch_hint::NTA>());

    return const_cast<_Type_*>(datapar::__simd_sized_dispatcher<__find_last_vectorized_internal>::__apply<_Type_>(
        __byte_length(__first, __last), &__find_last_scalar<_Type_>, std::move(__simd_args), std::move(__fallback_args)));
}
__SIMD_STL_ALGORITHM_NAMESPACE_END