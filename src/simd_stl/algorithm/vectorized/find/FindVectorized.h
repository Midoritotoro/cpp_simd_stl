#pragma once

#include <src/simd_stl/numeric/SizedSimdDispatcher.h>
#include <src/simd_stl/numeric/CachePrefetcher.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline const _Type_* __find_scalar(
    const void* __first,
    const void* __last,
    _Type_      __value) noexcept
{
    auto __current = static_cast<const _Type_*>(__first);

    while (__current != __last && *__current != __value)
        ++__current;

    return __current;
}

template <class _Simd_>
struct __find_vectorized_internal {
    template <class _CachePrefetcher_>
    simd_stl_declare_const_function simd_stl_always_inline const typename _Simd_::value_type* operator()(
        const sizetype              __aligned_size,
        const sizetype              __tail_size,
        const void*                 __first,
        const void*                 __last,
        typename _Simd_::value_type __value,
        _CachePrefetcher_&&         __prefetcher) noexcept
    {
        const auto __guard = numeric::make_guard<_Simd_>();

        constexpr auto __is_masked_memory_access_supported = _Simd_::template is_native_mask_store_supported_v<> &&
            _Simd_::template is_native_mask_load_supported_v<>;

        auto __comparand = _Simd_(__value);

        const void* __stop_at = __first;
        __advance_bytes(__stop_at, __aligned_size);

        do {
            __prefetcher(reinterpret_cast<const char*>(__first) + (sizeof(_Simd_)));

            const auto __loaded = _Simd_::load(__first);
            const auto __mask = __comparand.mask_compare<numeric::simd_comparison::equal>(__loaded);

            if (__mask.any_of())
                return static_cast<const typename _Simd_::value_type*>(__first) + __mask.count_trailing_zero_bits();

            __advance_bytes(__first, sizeof(_Simd_));
        } while (__first != __stop_at);

        if constexpr (__is_masked_memory_access_supported) {
            if (__tail_size != 0) {
                const auto __tail_mask = _Simd_::make_tail_mask(__tail_size);
                const auto __loaded = _Simd_::mask_load(__first, __tail_mask);

                const auto __compared = __comparand.native_compare<numeric::simd_comparison::equal>(__loaded) & __tail_mask;
                const auto __mask = numeric::simd_mask<_Simd_::__generation,
                    typename _Simd_::value_type>(numeric::__simd_to_native_mask<_Simd_::__generation,
                        typename _Simd_::policy_type, std::remove_cv_t<decltype(__compared)>>(__compared));

                if (__mask.any_of())
                    return static_cast<const typename _Simd_::value_type*>(__first) + __mask.count_trailing_zero_bits();
            }
        }
        else {
            if (__tail_size != 0)
                __last = __find_scalar(__first, __last, __value);
        }

        return static_cast<const typename _Simd_::value_type*>(__last);
    }
};

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline _Type_* simd_stl_stdcall __find_vectorized(
    const void* __first,
    const void* __last,
    _Type_      __value) noexcept
{
    return const_cast<_Type_*>(numeric::__simd_sized_dispatcher<__find_vectorized_internal>::__apply<_Type_>(
        __byte_length(__first, __last), &__find_scalar<_Type_>,
        std::make_tuple(__first, __last, __value, numeric::__cache_prefetcher<numeric::__prefetch_hint::NTA>{}),
        std::make_tuple(__first, __last, __value)));
}

__SIMD_STL_ALGORITHM_NAMESPACE_END