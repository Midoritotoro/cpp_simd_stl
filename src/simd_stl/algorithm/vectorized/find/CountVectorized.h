#pragma once

#include <src/simd_stl/numeric/SizedSimdDispatcher.h>
#include <src/simd_stl/numeric/CachePrefetcher.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
simd_stl_always_inline sizetype simd_stl_stdcall __count_scalar(
    const void*     __first,
    const sizetype  __bytes,
    _Type_          __value) noexcept
{
    auto __current      = static_cast<const _Type_*>(__first);
    const auto __length = __bytes / sizeof(_Type_);

    auto __count = sizetype(0);

    for (auto __index = sizetype(0); __index < __length; ++__index)
        __count += (*__current++ == __value);

    return __count;
}


template <class _Simd_>
struct __count_vectorized_internal {
    template <class _CachePrefetcher_>
    simd_stl_always_inline sizetype simd_stl_stdcall operator()(
        sizetype                    __aligned_size,
        sizetype                    __tail_size,
        const void*                 __first,
        typename _Simd_::value_type __value,
        _CachePrefetcher_&&         __prefetcher) noexcept
    {
        const auto __guard = numeric::make_guard<_Simd_>();

        constexpr auto __is_native_compare_return_number = std::is_integral_v<numeric::__native_compare_return_type<_Simd_, 
            typename _Simd_::value_type, numeric::simd_comparison::equal>>;

        constexpr auto __is_safe_reducible = std::is_integral_v<typename _Simd_::value_type> && !__is_native_compare_return_number;

        auto __count            = sizetype(0);

        const auto __comparand  = _Simd_(__value);
        auto __zeros            = _Simd_();

        if constexpr (__is_safe_reducible)
            __zeros.clear();

        do {
            const auto __loaded     = _Simd_::load(__first);
            const auto __compared   = __comparand.native_compare<numeric::simd_comparison::equal>(__loaded);

            if constexpr (__is_safe_reducible)
                __count += (__zeros - __compared).reduce_add();
            else
                __count += typename _Simd_::mask_type(__compared).count_set();

            __advance_bytes(__first, sizeof(_Simd_));
            __aligned_size -= sizeof(_Simd_);
        } while (__aligned_size != 0);

        if constexpr (_Simd_::template is_native_mask_load_supported_v<>) {
            if (__tail_size != 0) {
                const auto __tail_mask  = _Simd_::make_tail_mask(__tail_size);
                const auto __loaded     = _Simd_::mask_load(__first, __tail_mask);

                const auto __mask = typename _Simd_::mask_type(__comparand.native_compare<numeric::simd_comparison::equal>(__loaded) & __tail_mask);
                __count += __mask.count_set();
            }

            return __count;
        }
        else {
            return __count + __count_scalar(__first, __tail_size, __value);
        }
    }
};

template <class _Type_>
simd_stl_declare_const_function sizetype simd_stl_stdcall __count_vectorized(
    const void*     __first,
    const sizetype  __bytes,
    _Type_          __value) noexcept
{
    const auto __fallback_args  = std::forward_as_tuple(__first, __bytes, __value);
    const auto __simd_args      = std::forward_as_tuple(__first, __value, numeric::__cache_prefetcher<numeric::__prefetch_hint::NTA>());

    return numeric::__simd_sized_dispatcher<__count_vectorized_internal>::__apply<_Type_>(
        __bytes, &__count_scalar<_Type_>, std::move(__simd_args), std::move(__fallback_args));
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
