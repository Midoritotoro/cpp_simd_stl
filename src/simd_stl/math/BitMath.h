#pragma once 

#include <simd_stl/Types.h>
#include <simd_stl/compatibility/Inline.h>

#include <simd_stl/arch/ProcessorFeatures.h>
#include <simd_stl/arch/ProcessorDetection.h>

#include <simd_stl/compatibility/SimdCompatibility.h>
#include <src/simd_stl/type_traits/IsConstantEvaluated.h>


#if __has_include(<bit>) && __cplusplus > 201703L
    #include <bit>
#endif

#if defined(simd_stl_cpp_msvc)
    #include <intrin.h>
#endif

__SIMD_STL_MATH_NAMESPACE_BEGIN


template <typename _Type_>
_Type_ ClearLeftMostSet(const _Type_ value) {
    DebugAssert(value != _Type_());
    return value & (value - 1);
}

constexpr simd_stl::uint32 ConstexprCountTrailingZeroBits(simd_stl::uint32 v) noexcept
{
    if (!type_traits::is_constant_evaluated()) {
#if !defined(simd_stl_processor_arm)

#  if defined(simd_stl_processor_x86_64)
        if (arch::ProcessorFeatures::AVX2())
            return _tzcnt_u32(v);
        else
#  endif

#  if defined (simd_stl_processor_x86)
        {
            ulong index = 0;
            _BitScanForward(&index, v);

            return index;
        }
#  endif
#endif
    } 
    
    uint32 c = 32;

    v &= -signed(v);
    if (v) c--;

    if (v & 0x0000FFFF) c -= 16;
    if (v & 0x00FF00FF) c -= 8;
    if (v & 0x0F0F0F0F) c -= 4;

    if (v & 0x33333333) c -= 2;
    if (v & 0x55555555) c -= 1;

    return c;
}

constexpr simd_stl::uint32 ConstexprCountTrailingZeroBits(simd_stl::uint64 v) noexcept
{
    simd_stl::uint32 x = static_cast<simd_stl::uint32>(v);
    return x ? ConstexprCountTrailingZeroBits(x)
        : 32 + ConstexprCountTrailingZeroBits(static_cast<simd_stl::uint32>(v >> 32));
}

constexpr simd_stl::uint32 ConstexprCountTrailingZeroBits(uint8 v) noexcept
{
    unsigned int c = 8;

    v &= simd_stl::uint8(-signed(v));
    if (v) c--;

    if (v & 0x0000000F) c -= 4;
    if (v & 0x00000033) c -= 2;
    if (v & 0x00000055) c -= 1;

    return c;
}

constexpr simd_stl::uint32 ConstexprCountTrailingZeroBits(simd_stl::uint16 v) noexcept
{
    unsigned int c = 16;

    v &= simd_stl::uint16(-signed(v));
    if (v) c--;

    if (v & 0x000000FF) c -= 8;
    if (v & 0x00000F0F) c -= 4;

    if (v & 0x00003333) c -= 2;
    if (v & 0x00005555) c -= 1;

    return c;
}

constexpr inline simd_stl::uint32 ConstexprCountTrailingZeroBits(unsigned long v) noexcept
{
    return ConstexprCountTrailingZeroBits(IntegerForSizeof<long>::Unsigned(v));
}

constexpr inline simd_stl::uint32 CountTrailingZeroBits(simd_stl::uint32 v) noexcept
{
    return ConstexprCountTrailingZeroBits(v);
}

constexpr inline simd_stl::uint32 CountTrailingZeroBits(simd_stl::uint8 v) noexcept
{
    return ConstexprCountTrailingZeroBits(v);
}

constexpr inline simd_stl::uint32 CountTrailingZeroBits(simd_stl::uint16 v) noexcept
{
    return ConstexprCountTrailingZeroBits(v);
}

constexpr inline simd_stl::uint32 CountTrailingZeroBits(simd_stl::uint64 v) noexcept
{
    return ConstexprCountTrailingZeroBits(v);
}

constexpr inline simd_stl::uint32 CountTrailingZeroBits(simd_stl::int32 v) noexcept
{
    return ConstexprCountTrailingZeroBits(static_cast<uint32>(v));
}

constexpr inline simd_stl::uint32 CountTrailingZeroBits(simd_stl::int8 v) noexcept
{
    return ConstexprCountTrailingZeroBits(static_cast<uint8>(v));
}

constexpr inline simd_stl::uint32 CountTrailingZeroBits(simd_stl::int16 v) noexcept
{
    return ConstexprCountTrailingZeroBits(static_cast<uint16>(v));
}

constexpr inline simd_stl::uint32 CountTrailingZeroBits(simd_stl::int64 v) noexcept
{
    return ConstexprCountTrailingZeroBits(static_cast<uint64>(v));
}

constexpr inline simd_stl::uint32 CountTrailingZeroBits(unsigned long v) noexcept
{
    return CountTrailingZeroBits(IntegerForSizeof<long>::Unsigned(v));
}

constexpr simd_stl::uint32 CountLeadingZeroBits(simd_stl::uint32 v) noexcept
{
    v = v | (v >> 1);
    v = v | (v >> 2);
    v = v | (v >> 4);

    v = v | (v >> 8);
    v = v | (v >> 16);

    return PopulationCount(~v);
}

constexpr simd_stl::uint32 CountLeadingZeroBits(simd_stl::uint8 v) noexcept
{
    v = v | (v >> 1);
    v = v | (v >> 2);
    v = v | (v >> 4);

    return PopulationCount(static_cast<simd_stl::uint8>(~v));
}

constexpr simd_stl::uint32 CountLeadingZeroBits(simd_stl::uint16 v) noexcept
{
    v = v | (v >> 1);
    v = v | (v >> 2);

    v = v | (v >> 4);
    v = v | (v >> 8);

    return PopulationCount(static_cast<simd_stl::uint16>(~v));
}

constexpr simd_stl::uint32 CountLeadingZeroBits(simd_stl::uint64 v) noexcept
{
    v = v | (v >> 1);
    v = v | (v >> 2);
    v = v | (v >> 4);

    v = v | (v >> 8);
    v = v | (v >> 16);
    v = v | (v >> 32);

    return PopulationCount(~v);
}

constexpr simd_stl::uint32 CountLeadingZeroBits(unsigned long v) noexcept
{
    return CountLeadingZeroBits(IntegerForSizeof<long>::Unsigned(v));
}

static simd_stl_always_inline simd_stl::uint32 __BitScanReverse(unsigned v) noexcept
{
    uint32 result = CountLeadingZeroBits(v);

    result ^= sizeof(unsigned) * 8 - 1;
    return result;
}

__SIMD_STL_MATH_NAMESPACE_END