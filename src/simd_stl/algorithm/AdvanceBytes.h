#pragma once 

#include <simd_stl/compatibility/Inline.h>
#include <simd_stl/SimdStlNamespace.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <
    typename    _Type_,
    class       _Integral_>
simd_stl_always_inline void AdvanceBytes(
    _Type_*&    target,
    _Integral_  offset) noexcept
{
    target = reinterpret_cast<_Type_*>(const_cast<unsigned char*>(reinterpret_cast<const volatile unsigned char*>(target)) + offset);
}

template <
    typename    _Type_,
    class       _Integral_>
simd_stl_always_inline void AdvanceBytes(
    const _Type_*&  target,
    _Integral_      offset) noexcept
{
    target = reinterpret_cast<const _Type_*>(const_cast<const unsigned char*>(reinterpret_cast<const volatile unsigned char*>(target)) + offset);
}

simd_stl_always_inline size_t ByteLength(
    const void* first,
    const void* last) noexcept
{
    const auto firstChar    = const_cast<const unsigned char*>(reinterpret_cast<const volatile unsigned char*>(first));
    const auto lastChar     = const_cast<const unsigned char*>(reinterpret_cast<const volatile unsigned char*>(last));

    return static_cast<std::size_t>(lastChar - firstChar);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
