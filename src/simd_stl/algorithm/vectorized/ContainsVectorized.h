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
simd_stl_declare_const_function simd_stl_always_inline bool ContainsScalar(
    const void* firstPointer,
    const void* lastPointer,
    _Type_      value) noexcept
{
    auto pointer = static_cast<const _Type_*>(firstPointer);

    while (pointer != lastPointer) {
        if (*pointer == value)
            return true;

        ++pointer;
    }

    return false;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_>
simd_stl_declare_const_function simd_stl_always_inline bool ContainsVectorizedInternal(
    const void* firstPointer,
    const void* lastPointer,
    _Type_      value) noexcept
{
    using _SimdType_ = numeric::basic_simd<_SimdGeneration_, _Type_>;

    const auto size = ByteLength(firstPointer, lastPointer);
    const auto alignedSize = size & (~(_SimdType_::template width() - 1));

    if (alignedSize != 0) {
        const auto comparand = _SimdType_(value);

        const void* stopAt = firstPointer;
        AdvanceBytes(stopAt, alignedSize);

        do {
            const auto mask = comparand.maskEqual<_Type_>(_SimdType_::loadUnaligned(firstPointer));

            if (mask.unwrap() != 0)
                return true;

            AdvanceBytes(firstPointer, _SimdType_::template width());
        } while (firstPointer != stopAt);
    }

    return (firstPointer == lastPointer) ? false : ContainsScalar(firstPointer, lastPointer, value);
}

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline bool ContainsVectorized(
    const void* firstPointer,
    const void* lastPointer,
    _Type_      value) noexcept
{
    /* if (arch::ProcessorFeatures::AVX512F())
         return FindVectorizedInternal<arch::CpuFeature::AVX512F, _Type_>(firstPointer, lastPointer, value);
     else if (arch::ProcessorFeatures::AVX2())
         return FindVectorizedInternal<arch::CpuFeature::AVX2, _Type_>(firstPointer, lastPointer, value);*/
    /*else*/ if (arch::ProcessorFeatures::SSE2())
        return ContainsVectorizedInternal<arch::CpuFeature::SSE2, _Type_>(firstPointer, lastPointer, value);

    return ContainsScalar(firstPointer, lastPointer, value);
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
