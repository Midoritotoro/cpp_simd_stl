#pragma once

#include <src/simd_stl/numeric/SizedSimdDispatcher.h>
#include <src/simd_stl/numeric/CachePrefetcher.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <typename _Type_> 
simd_stl_declare_const_function simd_stl_always_inline sizetype simd_stl_stdcall __mismatch_scalar(
    const void* __first,
    const void* __second,
    sizetype    __size) noexcept
{
    const _Type_* __first_pointer   = static_cast<const _Type_*>(__first);
    const _Type_* __second_pointer  = static_cast<const _Type_*>(__second);

    while (__size--)
        if (*__first_pointer++ != *__second_pointer++)
            return (__first_pointer - static_cast<const _Type_*>(__first));
   
    return (__first_pointer - static_cast<const _Type_*>(__first));
}

template <class _Simd_>
struct __mismatch_vectorized_internal {
    template <class _CachePrefetcher_>
    simd_stl_declare_const_function simd_stl_always_inline sizetype operator()(
        sizetype            __aligned_size,
        sizetype            __tail_size,
        const void*         __first,
        const void*         __second,
        const sizetype      __length,
        _CachePrefetcher_&& __prefetcher) noexcept
    {
        numeric::zero_upper_at_exit_guard<_Simd_::__generation> __guard;

        constexpr auto __is_masked_memory_access_supported = _Simd_::template is_native_mask_store_supported_v<> &&
            _Simd_::template is_native_mask_load_supported_v<>;

        auto __cached_first = static_cast<const typename _Simd_::value_type*>(__first);

        auto __stop_at = __first;
        __advance_bytes(__stop_at, __aligned_size);

        do {
            __prefetcher(static_cast<const char*>(__first) + sizeof(_Simd_));
            __prefetcher(static_cast<const char*>(__second) + sizeof(_Simd_));

            const auto __loaded_first   = _Simd_::load(__first);
            const auto __loaded_second  = _Simd_::load(__second);

            const auto __mask = __loaded_first.mask_compare<numeric::simd_comparison::equal>(__loaded_second);

            if (__mask.all_of() == false)
                return (static_cast<const typename _Simd_::value_type*>(__first) - __cached_first) + __mask.count_trailing_one_bits();

            __advance_bytes(__first, sizeof(_Simd_));
            __advance_bytes(__second, sizeof(_Simd_));
        } while (__first != __stop_at);

        if (__tail_size == 0)
            return __length;

        if constexpr (__is_masked_memory_access_supported) {
            const auto __tail_mask = _Simd_::make_tail_mask(__tail_size);

            const auto __loaded_first   = _Simd_::mask_load(__first, __tail_mask);
            const auto __loaded_second  = _Simd_::mask_load(__second, __tail_mask);

            const auto __compared = __loaded_first.native_compare<numeric::simd_comparison::equal>(__loaded_second) & __tail_mask;
            const auto __mask = numeric::simd_mask<_Simd_::__generation, typename _Simd_::value_type>(numeric::__simd_to_native_mask<_Simd_::__generation,
                    typename _Simd_::policy_type, std::remove_cv_t<decltype(__compared)>>(__compared));

            const auto __all_equal_mask = (1u << (__tail_size / sizeof(typename _Simd_::value_type))) - 1;

            if (__mask != __all_equal_mask)
                return (static_cast<const typename _Simd_::value_type*>(__first) - __cached_first) + __mask.count_trailing_one_bits();

            return __length;
        }
        else {
            return __mismatch_scalar<typename _Simd_::value_type>(__first, __second, __tail_size / sizeof(typename _Simd_::value_type));
        }
    }
};


template <typename _Type_>
simd_stl_declare_const_function sizetype simd_stl_stdcall __mismatch_vectorized(
    const void*     __first,
    const void*     __second,
    const sizetype  __size) noexcept
{
    const auto __bytes = __size * sizeof(_Type_);

    return numeric::__simd_sized_dispatcher<__mismatch_vectorized_internal>::__apply<_Type_>(
        __bytes, &__mismatch_scalar<_Type_>,
        std::make_tuple(__first, __second, __size, numeric::__cache_prefetcher<numeric::__prefetch_hint::NTA>{}),
        std::make_tuple(__first, __second, __size));
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
