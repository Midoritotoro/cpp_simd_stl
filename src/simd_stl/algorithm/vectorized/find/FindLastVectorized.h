#pragma once

#include <src/simd_stl/numeric/SizedSimdDispatcher.h>
#include <src/simd_stl/numeric/CachePrefetcher.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline const _Type_* simd_stl_stdcall __find_last_scalar(
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
    simd_stl_declare_const_function simd_stl_always_inline const typename _Simd_::value_type* simd_stl_stdcall operator()(
        const sizetype                  __aligned_size,
        const sizetype                  __tail_size,
        const void*                     __first,
        const void*                     __last,
        typename _Simd_::value_type     __value,
        _CachePrefetcher_&&             __prefetcher) noexcept
    {
        numeric::zero_upper_at_exit_guard<_Simd_::__generation> __guard;

        constexpr auto __is_masked_memory_access_supported = _Simd_::template is_native_mask_store_supported_v<> &&
            _Simd_::template is_native_mask_load_supported_v<>;

        const void* __cached_last = __last;
        const auto __comparand = _Simd_(__value);

        const void* __stop_at = __last;
        __rewind_bytes(__stop_at, __aligned_size);

        do {
            __rewind_bytes(__last, sizeof(_Simd_));
            __prefetcher(reinterpret_cast<const char*>(__last) - sizeof(_Simd_));

            const auto __loaded  = _Simd_::load(__last);
            const auto __mask    = __comparand.mask_compare<numeric::simd_comparison::equal>(__loaded);

            if (__mask.any_of())
                return static_cast<const typename _Simd_::value_type*>(__last) + __mask.count_trailing_zero_bits();
        } while (__last != __stop_at);

        if (__tail_size == 0)
            return static_cast<const typename _Simd_::value_type*>(__cached_last);

        if constexpr (__is_masked_memory_access_supported) {
            __rewind_bytes(__last, __tail_size);

            const auto __tail_mask  = _Simd_::make_tail_mask(__tail_size);
            const auto __loaded     = _Simd_::mask_load(__last, __tail_mask);

            const auto __compared = __comparand.native_compare<numeric::simd_comparison::equal>(__loaded) & __tail_mask;

            const auto __mask = numeric::simd_mask<_Simd_::__generation, typename _Simd_::value_type>(numeric::__simd_to_native_mask<_Simd_::__generation,
                typename _Simd_::policy_type, std::remove_cv_t<decltype(__compared)>>(__compared));

            if (__mask.any_of())
                return static_cast<const typename _Simd_::value_type*>(__last) + __mask.count_trailing_zero_bits() + 1;
        }
        else {
            return __find_last_scalar(__first, __last, __value);
        }
    }

};

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline _Type_* simd_stl_stdcall __find_last_vectorized(
    const void* __first,
    const void* __last,
    _Type_      __value) noexcept
{
    return const_cast<_Type_*>(numeric::__simd_sized_dispatcher<__find_last_vectorized_internal>::__apply<_Type_>(
        __byte_length(__first, __last), &__find_last_scalar<_Type_>,
        std::make_tuple(__first, __last, __value, numeric::__cache_prefetcher<numeric::__prefetch_hint::NTA>{}),
        std::make_tuple(__first, __last, __value)));
}

__SIMD_STL_ALGORITHM_NAMESPACE_END