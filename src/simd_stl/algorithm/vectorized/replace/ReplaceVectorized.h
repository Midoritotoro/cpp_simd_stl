#pragma once


#include <simd_stl/datapar/BasicSimd.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <typename _Type_>
simd_stl_always_inline void simd_stl_stdcall __replace_scalar(
    void*           __first,
    void*           __last,
    const _Type_    __old_value,
    const _Type_    __new_value) noexcept
{
    auto __first_pointer = static_cast<_Type_*>(__first);

    for (; __first_pointer != __last; ++__first_pointer)
        if (*__first_pointer == __old_value)
            *__first_pointer = __new_value;
}

template <class _Simd_>
struct __replace_vectorized_internal {
    template <class _CachePrefetcher_>
    simd_stl_always_inline void simd_stl_stdcall operator()(
        sizetype                            __aligned_size,
        sizetype                            __tail_size,
        void*                               __first,
        void*                               __last,
        const typename _Simd_::value_type   __old_value,
        const typename _Simd_::value_type   __new_value,
        _CachePrefetcher_&&                 __prefetcher) noexcept
    {
        const auto __guard = datapar::make_guard<_Simd_>();

        constexpr auto __is_masked_store_supported = _Simd_::template is_native_mask_store_supported_v<>;
        constexpr auto __is_masked_memory_access_supported = __is_masked_store_supported &&
            _Simd_::template is_native_mask_load_supported_v<>;

        void* __stop_at = __first;
        __advance_bytes(__stop_at, __aligned_size);

        const auto __comparand      = _Simd_(__old_value);
        const auto __replacement    = _Simd_(__new_value);

        do {
            __prefetcher(static_cast<const char*>(__first) + sizeof(_Simd_));

            const auto __loaded = _Simd_::load(__first);
            const auto __mask   = __loaded.native_compare<datapar::simd_comparison::equal>(__comparand);

            __replacement.mask_store(__first, __mask);
            __advance_bytes(__first, sizeof(_Simd_));
        } while (__first != __stop_at);

        if (__tail_size == 0)
            return;

        if constexpr (__is_masked_memory_access_supported) {
            const auto __tail_mask  = _Simd_::make_tail_mask(__tail_size);
            const auto __loaded     = _Simd_::mask_load(__first, __tail_mask);

            const auto __mask                   = __loaded.native_compare<datapar::simd_comparison::equal>(__comparand);
            const auto __mask_for_native_store  = datapar::__simd_convert_to_mask_for_native_store<_Simd_::__generation,
                typename _Simd_::policy_type, typename _Simd_::value_type>(__mask);

            const auto __store_mask = __mask_for_native_store & __tail_mask;
            __replacement.mask_store(__first, __store_mask);
        }
        else {
            __replace_scalar<typename _Simd_::value_type>(__first, __last, __old_value, __new_value);
        }
    }
};

template <typename _Type_>
void simd_stl_stdcall __replace_vectorized(
    void*           __first,
    void*           __last,
    const _Type_    __old_value,
    const _Type_    __new_value) noexcept
{
    
}

__SIMD_STL_ALGORITHM_NAMESPACE_END

