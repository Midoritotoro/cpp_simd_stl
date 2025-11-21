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

simd_stl_always_inline sizetype ByteLength(
    const volatile void* first,
    const volatile void* last) noexcept
{
    const auto firstChar    = const_cast<const unsigned char*>(
        reinterpret_cast<const volatile unsigned char*>(first));

    const auto lastChar     = const_cast<const unsigned char*>(
        reinterpret_cast<const volatile unsigned char*>(last));

    return static_cast<sizetype>(lastChar - firstChar);
}

template <class _ContiguousIterator_>
constexpr inline type_traits::IteratorDifferenceType<_ContiguousIterator_> IteratorsDifference(
    const _ContiguousIterator_& firstIterator,
    const _ContiguousIterator_& lastIterator) noexcept
{
    using _DifferenceType_ = type_traits::IteratorDifferenceType<_ContiguousIterator_>;

    if constexpr (std::is_pointer_v<_ContiguousIterator_> || type_traits::is_iterator_random_ranges_v<_ContiguousIterator_>)
        return static_cast<_DifferenceType_>(lastIterator - firstIterator);

    const auto pointerLikeAddress1 = std::to_address(firstIterator);
    const auto pointerLikeAddress2 = std::to_address(lastIterator);

    using _IteratorValueType_ = type_traits::IteratorValueType<_ContiguousIterator_>;

    const auto firstIteratorAddress = const_cast<const _IteratorValueType_*>(
        reinterpret_cast<const volatile _IteratorValueType_*>(pointerLikeAddress1));

    const auto lastIteratorAddress = const_cast<const _IteratorValueType_*>(
        reinterpret_cast<const volatile _IteratorValueType_*>(pointerLikeAddress2));

    return static_cast<_DifferenceType_>(lastIteratorAddress - firstIteratorAddress);
}

template <class _InputIterator_> 
constexpr inline bool is_nothrow_distance_v = type_traits::is_iterator_random_ranges_v<_InputIterator_> 
    || std::bool_constant<noexcept(std::declval<std::remove_reference_t<_InputIterator_>&>()++)>::value;

template <
    class _InputIterator_,
    class _DifferenceType_ = type_traits::IteratorDifferenceType<_InputIterator_>>
simd_stl_nodiscard simd_stl_always_inline constexpr type_traits::IteratorDifferenceType<_InputIterator_> distance(
    _InputIterator_ first,
    _InputIterator_ last) noexcept(is_nothrow_distance_v<_InputIterator_>)
{
    if constexpr (type_traits::is_iterator_random_ranges_v<_InputIterator_>) {
        return static_cast<_DifferenceType_>(last - first);
    }
    else {
        __verifyRange(first, last);

        auto firstUnwrapped        = _UnwrapIterator(first);
        const auto lastUnwrapped   = _UnwrapIterator(last);

        auto distance = _DifferenceType_(0);

        for (; firstUnwrapped != lastUnwrapped; ++firstUnwrapped)
            ++distance;

        return distance;
    }
}

template <
    class _Char_, 
    class _UnsignedIntegralType_>
simd_stl_nodiscard _Char_* UnsignedIntegralToBuffer(
    _Char_*                 end,
    _UnsignedIntegralType_  value) noexcept
{ 
    static_assert(std::is_unsigned_v<_UnsignedIntegralType_>);

#if defined(simd_stl_os_win)
    auto truncated = value;
#else

    if constexpr (sizeof(_UnsignedIntegralType_) > 4) {
        while (value > 0xFFFFFFFFU) {
            auto chunk = static_cast<unsigned long>(value % 1000000000);
            value /= 1000000000;

            for (int current = 0; current != 9; ++current) {
                *--end = static_cast<_Char_>('0' + chunk % 10);
                chunk /= 10;
            }
        }
    }

    auto truncated = static_cast<unsigned long>(_UVal);
#endif // ^^^ !defined(_WIN64) ^^^

    do {
        *--end = static_cast<_Char_>('0' + truncated % 10);
        truncated /= 10;
    } while (truncated != 0);

    return end;
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
