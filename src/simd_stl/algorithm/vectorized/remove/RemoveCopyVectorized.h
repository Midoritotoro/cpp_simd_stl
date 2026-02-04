#pragma once

#include <src/simd_stl/datapar/SizedSimdDispatcher.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
simd_stl_declare_const_function _Type_* __remove_copy_scalar(
    const void* __first,
    const void* __last,
    void*       __destination,
    _Type_      __value) noexcept
{
    auto __first_pointer       = static_cast<const _Type_*>(__first);
    auto __destination_pointer = static_cast<_Type_*>(__destination);

    for (; __first_pointer != __last; ++__first_pointer) {
        const auto __current_value = *__first_pointer;

        if (__current_value != __value)
            *__destination_pointer++ = __current_value;
    }

    return __destination_pointer;
}

template <class _Simd_>
struct __remove_copy_vectorized_internal {
    simd_stl_declare_const_function typename _Simd_::value_type* operator()(
        sizetype                    __aligned_size,
        sizetype                    __tail_size,
        const void*                 __first,
        const void*                 __last,
        void*                       __destination,
        typename _Simd_::value_type __value) noexcept
    {
        const auto __guard = datapar::make_guard<_Simd_>();

        const void* __stop_at = __first;
        __advance_bytes(__stop_at, __aligned_size);

        const auto __comparand = _Simd_(__value);

        do {
            const auto __loaded = _Simd_::load(__first);
            const auto __mask = __loaded.mask_compare<datapar::simd_comparison::equal>(__comparand);

            __destination = __loaded.compress_store(__destination, __mask);
            __advance_bytes(__first, sizeof(_Simd_));
        } while (__first != __stop_at);

        return (__first == __last)
            ? static_cast<typename _Simd_::value_type*>(__destination)
            : __remove_copy_scalar(__first, __last, __destination, __value);
    }
};

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline _Type_* simd_stl_stdcall __remove_copy_vectorized(
    const void* __first,
    const void* __last,
    void*       __destination,
    _Type_      __value) noexcept
{
    return datapar::__simd_sized_dispatcher<__remove_copy_vectorized_internal>::__apply<_Type_>(
        __byte_length(__first, __last), &__remove_copy_scalar<_Type_>,
        __first, __last, __destination, __value);
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
