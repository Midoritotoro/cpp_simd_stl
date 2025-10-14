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

template <typename _Type_> 
constexpr int maximumCount() noexcept {
    if      constexpr (numeric::is_epu32_v<_Type_> || numeric::is_epi32_v<_Type_>)
        return 0x1FFF'FFFF;
    else if constexpr (numeric::is_epi16_v<_Type_> || numeric::is_epu16_v<_Type_>)
        return 0x7FFF;
    else if constexpr (numeric::is_epi8_v<_Type_> || numeric::is_epu8_v<_Type_>)
        return 0xFF;
}

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline sizetype CountScalar(
    const void* firstPointer,
    const void* lastPointer,
    sizetype&   count,
    _Type_      value) noexcept
{
    auto pointer = static_cast<const _Type_*>(firstPointer);

    while (pointer != lastPointer) {
        count += (*pointer == value);
        ++pointer;
    }

    return count;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_>
simd_stl_declare_const_function simd_stl_always_inline sizetype CountVectorizedInternal(
    const void* firstPointer,
    const void* lastPointer,
    _Type_      value) noexcept
{
    using _SimdType_        = numeric::basic_simd<_SimdGeneration_, _Type_>;

    const auto size         = ByteLength(firstPointer, lastPointer);
    auto alignedSize        = size & (~(_SimdType_::template width() - 1));

    sizetype count = 0;


    if (alignedSize != 0) {
        const auto comparand = _SimdType_(value);
        const void* stopAt = firstPointer;

        AdvanceBytes(stopAt, alignedSize);

        do {
            const auto compared = comparand.maskEqual(_SimdType_::loadUnaligned(firstPointer));
            count += compared.countSet();

            AdvanceBytes(firstPointer, _SimdType_::template width());
        } while (firstPointer != stopAt);
    }
   

    return CountScalar(firstPointer, lastPointer, count, value);
}

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline sizetype CountVectorized(
    const void* firstPointer,
    const void* lastPointer,
    _Type_      value) noexcept
{
    /* if (arch::ProcessorFeatures::AVX512F())
         return FindVectorizedInternal<arch::CpuFeature::AVX512F, _Type_>(firstPointer, lastPointer, value);
     else if (arch::ProcessorFeatures::AVX2())
         return FindVectorizedInternal<arch::CpuFeature::AVX2, _Type_>(firstPointer, lastPointer, value);*/
    /*else*/ if (arch::ProcessorFeatures::SSE42())
        return CountVectorizedInternal<arch::CpuFeature::SSE42, _Type_>(firstPointer, lastPointer, value);

    sizetype count = 0;
    return CountScalar(firstPointer, lastPointer, count, value);
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
