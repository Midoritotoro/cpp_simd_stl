#pragma once 

#include <simd_stl/compatibility/Inline.h>
#include <simd_stl/SimdStlNamespace.h>
#include <src/simd_stl/type_traits/IteratorCheck.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

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
inline simd_stl_constexpr_cxx20 sizetype IteratorsDifference(
    _ContiguousIterator_ firstIterator,
    _ContiguousIterator_ lastIterator) noexcept
{
    if constexpr (std::is_pointer_v<_ContiguousIterator_>)
        return static_cast<sizetype>(lastIterator - firstIterator);

    const auto pointerLikeAddress1 = std::to_address(firstIterator);
    const auto pointerLikeAddress2 = std::to_address(lastIterator);

    using _IteratorValueType_ = type_traits::IteratorValueType<_ContiguousIterator_>;

    return static_cast<sizetype>(
        const_cast<const _IteratorValueType_*>(
            reinterpret_cast<const volatile _IteratorValueType_*>(pointerLikeAddress1)) -
        const_cast<const _IteratorValueType_*>(
            reinterpret_cast<const volatile _IteratorValueType_*>(pointerLikeAddress2))
    );
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
