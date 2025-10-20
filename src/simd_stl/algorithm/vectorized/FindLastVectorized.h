#pragma once

#include <simd_stl/SimdStlNamespace.h>

#include <simd_stl/compatibility/Inline.h>
#include <simd_stl/compatibility/FunctionAttributes.h>

#include <simd_stl/compatibility/SimdCompatibility.h>
#include <simd_stl/arch/ProcessorFeatures.h>

#include <src/simd_stl/algorithm/AdvanceBytes.h>

#include <simd_stl/math/BitMath.h>
#include <simd_stl/compatibility/Inline.h>

#include <simd_stl/numeric/BasicSimd.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline const void* FindLastScalar(
    const void* firstPointer,
    const void* lastPointer,
    _Type_      value) noexcept
{
    auto pointer = static_cast<const _Type_*>(lastPointer);

    while (pointer != firstPointer && *pointer != value)
        --pointer;

    return (pointer == firstPointer) ? lastPointer : pointer;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_>
simd_stl_declare_const_function simd_stl_always_inline const void* FindLastVectorizedInternal(
    const void* firstPointer,
    const void* lastPointer,
    _Type_      value) noexcept
{
    using _SimdType_ = numeric::basic_simd<_SimdGeneration_, _Type_>;

    const auto size = ByteLength(firstPointer, lastPointer);
    const auto alignedSize = size & (~(_SimdType_::template width() - 1));

    const void* cachedLast = lastPointer;

    if (alignedSize != 0) {
        const auto comparand = _SimdType_(value);

        const void* stopAt = firstPointer;
        AdvanceBytes(stopAt, (size - alignedSize));

        do {
            const auto mask = comparand.maskEqual<_Type_>(_SimdType_::loadUnaligned(lastPointer));

            if (mask.unwrap() != 0)
                return static_cast<const _Type_*>(lastPointer) + mask.countTrailingZeroBits();

            RewindBytes(lastPointer, _SimdType_::template width());
        } while (lastPointer != stopAt);
    }

    return (firstPointer == lastPointer) ? cachedLast : FindLastScalar(firstPointer, lastPointer, value);
}

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline const void* FindLastVectorized(
    const void* firstPointer,
    const void* lastPointer,
    _Type_      value) noexcept
{
    if (arch::ProcessorFeatures::SSE2())
        return FindLastVectorizedInternal<arch::CpuFeature::SSE2, _Type_>(firstPointer, lastPointer, value);

    return FindLastScalar(firstPointer, lastPointer, value);
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
