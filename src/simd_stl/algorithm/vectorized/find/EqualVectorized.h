#pragma once

#include <src/simd_stl/numeric/SizedSimdDispatcher.h>
#include <src/simd_stl/numeric/CachePrefetcher.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <typename _Type_> 
simd_stl_always_inline bool simd_stl_stdcall __equal_scalar(
    const void* __first,
    const void* __second,
    sizetype    __size) noexcept
{
    __size /= sizeof(_Type_);

    auto __first_pointer    = static_cast<const _Type_*>(__first);
    auto __second_pointer   = static_cast<const _Type_*>(__second);

    while (__size--)
        if (*__first_pointer++ != *__second_pointer++)
            return false;

    return true;
}

template <class _Simd_>
struct __equal_vectorized_internal {
    template <class _CachePrefetcher_>
    simd_stl_always_inline bool simd_stl_stdcall operator()(
        sizetype            __aligned_size,
        sizetype            __tail_size,
        const void*         __first,
        const void*         __second,
        const sizetype      __length,
        _CachePrefetcher_&& __prefetcher) noexcept
    {
        const auto __guard = numeric::make_guard<_Simd_>();

        do {
            __prefetcher(__bytes_pointer_offset(__first, sizeof(_Simd_)));
            __prefetcher(__bytes_pointer_offset(__second, sizeof(_Simd_)));

            const auto __loaded_first   = _Simd_::load(__first);
            const auto __loaded_second  = _Simd_::load(__second);

            const auto __mask = __loaded_first.mask_compare<numeric::simd_comparison::equal>(__loaded_second);

            if (__mask.all_of() == false)
                return false;

            __advance_bytes(__first, sizeof(_Simd_));
            __advance_bytes(__second, sizeof(_Simd_));

            __aligned_size -= sizeof(_Simd_);
        } while (__aligned_size != 0);

        if (__tail_size == 0)
            return true;

        if constexpr (_Simd_::template is_native_mask_load_supported_v<>) {
            const auto __tail_mask = _Simd_::make_tail_mask(__tail_size);

            const auto __loaded_first   = _Simd_::mask_load(__first, __tail_mask);
            const auto __loaded_second  = _Simd_::mask_load(__second, __tail_mask);

            const auto __combined_native_mask = __loaded_first.native_compare<numeric::simd_comparison::equal>(__loaded_second) & __tail_mask;
            const auto __mask = typename _Simd_::mask_type(__combined_native_mask);

            const auto __tail_length = __tail_size / sizeof(typename _Simd_::value_type);
            const auto __all_equal_mask = (typename _Simd_::mask_type::mask_type(1) << __tail_length) - 1;

            return (__mask == __all_equal_mask);
        }
        else {
            return __equal_scalar<typename _Simd_::value_type>(__first, __second, __tail_size);
        }
    }
};

template <typename _Type_>
simd_stl_declare_const_function bool simd_stl_stdcall __equal_vectorized(
    const void*     __first,
    const void*     __second,
    const sizetype  __size) noexcept
{
    const auto __bytes = __size * sizeof(_Type_);

    const auto __fallback_args  = std::forward_as_tuple(__first, __second, __bytes);
    const auto __simd_args      = std::forward_as_tuple(__first, __second, __bytes, numeric::__cache_prefetcher<numeric::__prefetch_hint::NTA>());

    return numeric::__simd_sized_dispatcher<__equal_vectorized_internal>::__apply<_Type_>(
        __bytes, &__equal_scalar<_Type_>, std::move(__simd_args), std::move(__fallback_args));
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
