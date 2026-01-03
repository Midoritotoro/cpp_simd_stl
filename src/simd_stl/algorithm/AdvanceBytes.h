#pragma once 

#include <simd_stl/compatibility/Inline.h>
#include <simd_stl/SimdStlNamespace.h>

#include <src/simd_stl/type_traits/IteratorCheck.h>
#include <simd_stl/Types.h>

#include <src/simd_stl/utility/Assert.h>
#include <src/simd_stl/algorithm/MsvcIteratorUnwrap.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    typename    _Type_,
    class       _Integral_>
simd_stl_always_inline void __rewind_bytes(
    _Type_*&    __target,
    _Integral_  __offset) noexcept
{
    __target = reinterpret_cast<_Type_*>(const_cast<unsigned char*>(
        reinterpret_cast<const volatile unsigned char*>(__target)) - __offset);
}

template <
    typename    _Type_,
    class       _Integral_>
simd_stl_always_inline void __rewind_bytes(
    const _Type_*&  __target,
    _Integral_      __offset) noexcept
{
    __target = reinterpret_cast<const _Type_*>(const_cast<const unsigned char*>(
        reinterpret_cast<const volatile unsigned char*>(__target)) - __offset);
}

template <
    typename    _Type_,
    class       _Integral_>
simd_stl_always_inline void __advance_bytes(
    _Type_*&    __target,
    _Integral_  __offset) noexcept
{
    __target = reinterpret_cast<_Type_*>(const_cast<unsigned char*>(
        reinterpret_cast<const volatile unsigned char*>(__target)) + __offset);
}

template <
    typename    _Type_,
    class       _Integral_>
simd_stl_always_inline void __advance_bytes(
    const _Type_*&  __target,
    _Integral_      __offset) noexcept
{
    __target = reinterpret_cast<const _Type_*>(const_cast<const unsigned char*>(
        reinterpret_cast<const volatile unsigned char*>(__target)) + __offset);
}

simd_stl_always_inline sizetype __byte_length(
    const volatile void* ___first,
    const volatile void* __last) noexcept
{
    return static_cast<sizetype>(
        const_cast<const unsigned char*>(reinterpret_cast<const volatile unsigned char*>(__last)) - 
        const_cast<const unsigned char*>(reinterpret_cast<const volatile unsigned char*>(___first)));
}

template <class _ContiguousIterator_>
constexpr inline type_traits::IteratorDifferenceType<_ContiguousIterator_> __iterators_difference(
    const _ContiguousIterator_& ___first,
    const _ContiguousIterator_& __last) noexcept
{
    using _DifferenceType_ = type_traits::IteratorDifferenceType<_ContiguousIterator_>;

    if constexpr (std::is_pointer_v<_ContiguousIterator_> || type_traits::is_iterator_random_ranges_v<_ContiguousIterator_>)
        return static_cast<_DifferenceType_>(__last - ___first);

    const auto ___first_address  = std::to_address(___first);
    const auto __second_address = std::to_address(__last);

    using _IteratorValueType_ = type_traits::IteratorValueType<_ContiguousIterator_>;

    const auto ___first_iterator_address = const_cast<const _IteratorValueType_*>(
        reinterpret_cast<const volatile _IteratorValueType_*>(___first_address));

    const auto __last_iterator_address = const_cast<const _IteratorValueType_*>(
        reinterpret_cast<const volatile _IteratorValueType_*>(__second_address));

    return static_cast<_DifferenceType_>(__last_iterator_address - ___first_iterator_address);
}

template <class _InputIterator_> 
constexpr inline bool __is_nothrow_distance_v = type_traits::is_iterator_random_ranges_v<_InputIterator_> 
    || std::bool_constant<noexcept(std::declval<std::remove_reference_t<_InputIterator_>&>()++)>::value;

template <
    class _InputIterator_,
    class _DifferenceType_ = type_traits::IteratorDifferenceType<_InputIterator_>>
simd_stl_nodiscard simd_stl_always_inline constexpr type_traits::IteratorDifferenceType<_InputIterator_> distance(
    _InputIterator_ ___first,
    _InputIterator_ __last) noexcept(__is_nothrow_distance_v<_InputIterator_>)
{
    if constexpr (type_traits::is_iterator_random_ranges_v<_InputIterator_>) {
        return static_cast<_DifferenceType_>(__last - ___first);
    }
    else {
        __verifyRange(___first, __last);

        auto ___first_unwrapped        = _UnwrapIterator(___first);
        const auto __last_unwrapped   = _UnwrapIterator(__last);

        auto __distance = _DifferenceType_(0);

        for (; ___first_unwrapped != __last_unwrapped; ++___first_unwrapped)
            ++__distance;

        return __distance;
    }
}

template <
    class _Char_, 
    class _UnsignedIntegralType_>
simd_stl_nodiscard _Char_* __unsigned_integral_to_buffer(
    _Char_*                 __end,
    _UnsignedIntegralType_  __value) noexcept
{ 
    static_assert(std::is_unsigned_v<_UnsignedIntegralType_>);

    auto __truncated = __value;

    do {
        *--__end = static_cast<_Char_>('0' + __truncated % 10);
        __truncated /= 10;
    } while (__truncated != 0);

    return __end;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
