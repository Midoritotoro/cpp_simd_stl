#pragma once 

#include <simd_stl/compatibility/Inline.h>
#include <simd_stl/SimdStlNamespace.h>

#include <src/simd_stl/type_traits/IteratorCheck.h>
#include <simd_stl/Types.h>

#include <src/simd_stl/utility/Assert.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    typename    _Type_,
    class       _Integral_>
simd_stl_always_inline simd_stl_constexpr_cxx20 void RewindBytes(
    _Type_*&    target,
    _Integral_  offset) noexcept
{
    target = reinterpret_cast<_Type_*>(const_cast<unsigned char*>(
        reinterpret_cast<const volatile unsigned char*>(target)) - offset);
}

template <
    typename    _Type_,
    class       _Integral_>
simd_stl_always_inline simd_stl_constexpr_cxx20 void RewindBytes(
    const _Type_*&  target,
    _Integral_      offset) noexcept
{
    target = reinterpret_cast<const _Type_*>(const_cast<const unsigned char*>(
        reinterpret_cast<const volatile unsigned char*>(target)) - offset);
}

template <
    typename    _Type_,
    class       _Integral_>
simd_stl_always_inline simd_stl_constexpr_cxx20 void AdvanceBytes(
    _Type_*&    target,
    _Integral_  offset) noexcept
{
    target = reinterpret_cast<_Type_*>(const_cast<unsigned char*>(
        reinterpret_cast<const volatile unsigned char*>(target)) + offset);
}

template <
    typename    _Type_,
    class       _Integral_>
simd_stl_always_inline simd_stl_constexpr_cxx20 void AdvanceBytes(
    const _Type_*&  target,
    _Integral_      offset) noexcept
{
    target = reinterpret_cast<const _Type_*>(const_cast<const unsigned char*>(
        reinterpret_cast<const volatile unsigned char*>(target)) + offset);
}

simd_stl_always_inline size_t ByteLength(
    const void* first,
    const void* last) noexcept
{
    const auto firstChar    = const_cast<const unsigned char*>(
        reinterpret_cast<const volatile unsigned char*>(first));

    const auto lastChar     = const_cast<const unsigned char*>(
        reinterpret_cast<const volatile unsigned char*>(last));

    return static_cast<std::size_t>(lastChar - firstChar);
}

template <class _ContiguousIterator_>
constexpr inline type_traits::IteratorDifferenceType<_ContiguousIterator_> IteratorsDifference(
    const _ContiguousIterator_& firstIterator,
    const _ContiguousIterator_& lastIterator) noexcept
{
    if constexpr (std::is_pointer_v<_ContiguousIterator_> && type_traits::is_iterator_random_ranges_v<_ContiguousIterator_>)
        return static_cast<type_traits::IteratorDifferenceType<_ContiguousIterator_>>(lastIterator - firstIterator);

    AssertUnreachable();
    return -1;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
