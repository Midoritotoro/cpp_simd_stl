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
simd_stl_declare_const_function simd_stl_always_inline sizetype CountScalar(
    const void*     firstPointer,
    const sizetype  bytes,
    sizetype&       count,
    _Type_          value) noexcept
{
    auto pointer = static_cast<const _Type_*>(firstPointer);
    const auto length = bytes / sizeof(_Type_);

    for (sizetype current = 0; current < length; ++current)
        count += (*pointer++ == value);

    return count;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_>
simd_stl_declare_const_function simd_stl_always_inline sizetype CountVectorizedInternal(
    const void*     firstPointer,
    const sizetype  bytes,
    _Type_          value) noexcept
{
    using _SimdType_        = numeric::basic_simd<_SimdGeneration_, _Type_>;
    auto alignedSize        = bytes & (~(_SimdType_::template width() - 1));

    sizetype count = 0;

    if (alignedSize != 0) {
        const auto comparand = _SimdType_(value);

        for (sizetype current = 0; current < alignedSize; current += _SimdType_::template width()) {
            const auto loaded   = _SimdType_::loadUnaligned(static_cast<const char*>(firstPointer) + current);
            const auto compared = comparand.maskEqual(loaded);

            count += compared.countSet();
        }
    }
   
    AdvanceBytes(firstPointer, alignedSize);
    return CountScalar(firstPointer, (bytes - alignedSize), count, value);
}

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline sizetype CountVectorized(
    const void*     firstPointer,
    const sizetype  bytes,
    _Type_          value) noexcept
{
    /* if (arch::ProcessorFeatures::AVX512F())
         return FindVectorizedInternal<arch::CpuFeature::AVX512F, _Type_>(firstPointer, lastPointer, value);
     else if (arch::ProcessorFeatures::AVX2())
         return FindVectorizedInternal<arch::CpuFeature::AVX2, _Type_>(firstPointer, lastPointer, value);*/
    /*else*/ if (arch::ProcessorFeatures::SSE2())
        return CountVectorizedInternal<arch::CpuFeature::SSE2, _Type_>(firstPointer, bytes, value);

    sizetype count = 0;
    return CountScalar(firstPointer, bytes, count, value);
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
