#pragma once

#include <src/simd_stl/datapar/SimdDispatcher.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
void simd_stl_stdcall __reverse_copy_scalar(
    const void* __first,
    const void* __last,
    void*       __destination) noexcept
{
    auto __first_pointer        = static_cast<const _Type_*>(__first);
    auto __last_pointer         = static_cast<const _Type_*>(__last);

    auto __destination_pointer  = static_cast<_Type_*>(__destination);

    for (; __first_pointer != __last_pointer; ++__destination_pointer)
        *__destination_pointer = *--__last_pointer;
}

template <class _Simd_>
struct __reverse_copy_vectorized_internal {
    void simd_stl_stdcall operator()(
        const void* __first,
        const void* __last,
        void*       __destination) noexcept
    {
        const auto __guard = datapar::make_guard<_Simd_>();
        const auto __aligned_size = __byte_length(__first, __last) & (~((sizeof(_Simd_)) - 1));

        if (__aligned_size != 0) {
            const void* __stop_at = __last;
            __rewind_bytes(__stop_at, __aligned_size);

            do {
                auto __loaded = _Simd_::load(static_cast<const char*>(__last) - sizeof(_Simd_));
                __loaded.reverse();
                __loaded.store(__destination);

                __advance_bytes(__destination, sizeof(_Simd_));
                __rewind_bytes(__last, sizeof(_Simd_));
            } while (__last != __stop_at);
        }

        if (__first != __last)
            __reverse_copy_scalar<typename _Simd_::value_type>(__first, __last, __destination);
    }
};

template <class _Type_>
void simd_stl_stdcall __reverse_copy_vectorized(
    const void* __first,
    const void* __last,
    void*       __destination) noexcept
{
    return datapar::__simd_dispatcher<__reverse_copy_vectorized_internal>::__apply<_Type_>(
        &__reverse_copy_scalar<_Type_>, __first, __last, __destination);
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
