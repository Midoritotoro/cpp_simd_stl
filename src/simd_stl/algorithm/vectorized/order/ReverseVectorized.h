#pragma once

#include <src/simd_stl/numeric/SimdDispatcher.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
simd_stl_always_inline void __reverse_scalar(
    void* __first,
    void* __last) noexcept
{
    auto __first_pointer  = static_cast<_Type_*>(__first);
    auto __last_pointer   = static_cast<_Type_*>(__last);

    for (; __first_pointer != __last_pointer && __first_pointer != --__last_pointer; ++__first_pointer) {
        _Type_ _Temp = *__last_pointer;

        *__last_pointer = *__first_pointer;
        *__first_pointer = _Temp;
    }
}

template <class _Simd_>
struct __reverse_vectorized_internal {
    simd_stl_declare_const_function simd_stl_always_inline void operator()(
        void* __first,
        void* __last) noexcept
    {
        const auto __guard = numeric::make_guard<_Simd_>();
        const auto __aligned_size = __byte_length(__first, __last) & (~((sizeof(_Simd_) << 1) - 1));

        if (__aligned_size != 0) {
            void* __stop_at = __first;
            __advance_bytes(__stop_at, __aligned_size >> 1);

            do {
                __rewind_bytes(__last, sizeof(_Simd_));

                auto __loaded_begin = _Simd_::load(__first);
                auto __loaded_end = _Simd_::load(__last);

                __loaded_begin.reverse();
                __loaded_end.reverse();

                __loaded_begin.store(__last);
                __loaded_end.store(__first);

                __advance_bytes(__first, sizeof(_Simd_));
            } while (__first != __stop_at);
        }

        if (__first != __last)
            __reverse_scalar<typename _Simd_::value_type>(__first, __last);
    }
};

template <class _Type_>
void __reverse_vectorized(
    void* __first,
    void* __last) noexcept
{
    return numeric::__simd_dispatcher<__reverse_vectorized_internal>::__apply<_Type_>(
        &__reverse_scalar<_Type_>, __first, __last);
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
