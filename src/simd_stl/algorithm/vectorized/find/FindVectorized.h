#pragma once

#include <src/simd_stl/numeric/SizedSimdDispatcher.h>
#include <src/simd_stl/numeric/CachePrefetcher.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

extern "C" {
    template <class _Type_>
    _Type_* simd_stl_stdcall __find_trivial_1(
        const void* __first,
        const void* __last,
        _Type_      __value) noexcept
}

template <class _Type_>
const _Type_* simd_stl_stdcall __find_scalar(
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
    const typename _Simd_::value_type* simd_stl_stdcall operator()(
        sizetype                    __aligned_size,
        sizetype                    __tail_size,
        const void*                 __first,
        const void*                 __last,
        typename _Simd_::value_type __value) const noexcept
    {
        const auto __guard      = numeric::make_guard<_Simd_>();
        const auto __comparand  = _Simd_(__value);

        do {
            const auto __loaded = _Simd_::load(__first);
            const auto __mask   = (__comparand == __loaded) | numeric::as_index_mask;

            if (__mask.any_of())
                return static_cast<const typename _Simd_::value_type*>(__first) + __mask.count_trailing_zero_bits();

            __advance_bytes(__first, sizeof(_Simd_));
            __aligned_size -= sizeof(_Simd_);
        } while (__aligned_size != 0);

        if (__tail_size != 0) {
            if constexpr (_Simd_::template is_native_mask_load_supported_v<>) {
                const auto __tail_mask  = _Simd_::make_tail_mask(__tail_size);
                const auto __loaded     = _Simd_::mask_load(__first, __tail_mask);

                const auto __mask = typename _Simd_::mask_type(((__comparand == __loaded) | numeric::as_native) & __tail_mask);

                if (__mask.any_of())
                    return static_cast<const typename _Simd_::value_type*>(__first) + __mask.count_trailing_zero_bits();
            }
            else {
                __last = __find_scalar(__first, __last, __value);
            }
        }

        return static_cast<const typename _Simd_::value_type*>(__last);
    }
};

template <class _Type_>
_Type_* simd_stl_stdcall __find_vectorized(
    const void* __first,
    const void* __last,
    _Type_      __value) noexcept
{
    return const_cast<_Type_*>(numeric::__simd_sized_dispatcher<__find_vectorized_internal>::__apply<_Type_>(
        __byte_length(__first, __last), &__find_scalar<_Type_>, __first, __last, __value));
}

__SIMD_STL_ALGORITHM_NAMESPACE_END