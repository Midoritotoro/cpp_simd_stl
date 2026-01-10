#pragma once

#include <src/simd_stl/numeric/SizedSimdDispatcher.h>
#include <src/simd_stl/numeric/CachePrefetcher.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline _Type_* __remove_scalar(
    void*       __first,
    const void* __current,
    const void* __last,
    _Type_      __value) noexcept
{
    auto __current_pointer  = static_cast<const _Type_*>(__current);
    auto __first_pointer    = static_cast<_Type_*>(__first);

    for (; __current_pointer != __last; ++__current_pointer) {
        const auto __current_value = *__current_pointer;

        if (__current_value != __value)
            *__first_pointer++ = __current_value;
    }

    return __first_pointer;
}


template <class _Simd_>
struct __remove_vectorized_internal {
    template <class _CachePrefetcher_>
    simd_stl_always_inline typename _Simd_::value_type* operator()(
        sizetype                    __aligned_size,
        sizetype                    __tail_size,
        void*                       __first,
        const void*                 __last,
        typename _Simd_::value_type __value,
        _CachePrefetcher_&&         __prefetcher) noexcept
    {
        const auto __guard = numeric::make_guard<_Simd_>();
        auto __current = __first;

        const auto __comparand = _Simd_(__value);

        auto __stop_at = __first;
        __advance_bytes(__stop_at, __aligned_size);

        do {
            __prefetcher(static_cast<const char*>(__current) + sizeof(_Simd_));
            
            const auto __loaded = _Simd_::load(__current);
            const auto __mask = __comparand.mask_compare<numeric::simd_comparison::equal>(__loaded);

            __first = __loaded.compress_store(__first, __mask);
            __advance_bytes(__current, sizeof(_Simd_));
        } while (__current != __stop_at);

        return (__tail_size == 0)
            ? static_cast<typename _Simd_::value_type*>(__first)
            : __remove_scalar<typename _Simd_::value_type>(__first, __current, __last, __value);
    }
};

template <class _Type_>
simd_stl_declare_const_function _Type_* simd_stl_stdcall __remove_vectorized(
    void*       __first,
    const void* __last,
    _Type_      __value) noexcept
{
    return numeric::__simd_sized_dispatcher<__remove_vectorized_internal>::__apply<_Type_>(
        __byte_length(__first, __last), &__remove_scalar<_Type_>,
        std::make_tuple(__first, __last, __value, numeric::__cache_prefetcher<numeric::__prefetch_hint::NTA>{}),
        std::make_tuple(__first, __first, __last, __value));
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
