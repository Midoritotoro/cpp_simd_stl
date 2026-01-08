#pragma once

#include <src/simd_stl/numeric/SimdDispatcher.h>


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
    simd_stl_always_inline typename _Simd_::value_type* operator()(
        void*                       __first,
        const void*                 __last,
        typename _Simd_::value_type __value) noexcept
    {
        numeric::zero_upper_at_exit_guard<_Simd_::__generation> __guard;

        const auto __aligned_size = __byte_length(__first, __last) & (~(sizeof(_Simd_) - 1));
        auto __current = __first;

        if (__aligned_size != 0) {
            const auto __comparand = _Simd_(__value);

            auto __stop_at = __first;
            __advance_bytes(__stop_at, __aligned_size);

            do {
                const auto __loaded = _Simd_::load(__current);
                const auto __mask = __comparand.mask_compare<numeric::simd_comparison::equal>(__loaded);

                __first = __loaded.compress_store(__first, __mask);
                __advance_bytes(__current, sizeof(_Simd_));
            } while (__current != __stop_at);
        }

        return (__current == __last) 
            ? static_cast<typename _Simd_::value_type*>(__first)
            : __remove_scalar<typename _Simd_::value_type>(__first, __current, __last, __value);
    }
};

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline _Type_* __remove_vectorized(
    void*       __first,
    const void* __last,
    _Type_      __value) noexcept
{
    auto __first_pointer = static_cast<_Type_*>(__first);
    auto __last_pointer = static_cast<const _Type_*>(__last);

    return numeric::__simd_dispatcher<__remove_vectorized_internal>::__apply<_Type_>(
        &__remove_scalar<_Type_>, std::make_tuple(__first_pointer, __last_pointer, __value),
        std::make_tuple(__first_pointer, __first_pointer, __last_pointer, __value));
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
