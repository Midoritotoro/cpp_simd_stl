#pragma once

#include <simd_stl/SimdStlNamespace.h>

#include <simd_stl/compatibility/Inline.h>
#include <simd_stl/compatibility/FunctionAttributes.h>

#include <simd_stl/compatibility/SimdCompatibility.h>
#include <simd_stl/arch/ProcessorFeatures.h>

#include <src/simd_stl/algorithm/vectorized/traits/FindVectorizedTraits.h>
#include <src/simd_stl/algorithm/AdvanceBytes.h>

#include <src/simd_stl/math/BitMath.h>
#include <simd_stl/compatibility/Inline.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline simd_stl_constexpr_cxx20 const void* FindScalar(
    const void* firstPointer,
    const void* lastPointer,
    _Type_      value) noexcept
{
    auto pointer = static_cast<const _Type_*>(firstPointer);

    while (pointer != lastPointer && *pointer != value) 
        ++pointer;

    return pointer;
}

template <
    class _Traits_,
    class _Type_>
simd_stl_declare_const_function simd_stl_always_inline simd_stl_constexpr_cxx20 const void* FindVectorizedInternal(
    const void* firstPointer,
    const void* lastPointer,
    _Type_      value) noexcept
{
    const auto size         = ByteLength(firstPointer, lastPointer);
    const auto alignedSize  = size & (~(_Traits_::portionSize - 1));

    if (alignedSize != 0) {
        const auto comparand = _Traits_::Set(value);

        const void* stopAt = firstPointer;
        AdvanceBytes(stopAt, alignedSize);

        do {
            const auto mask = _Traits_::ToMask(
                _Traits_::Compare<sizeof(_Type_)>(comparand, _Traits_::LoadUnaligned(firstPointer)));

            if (mask != 0)
                return static_cast<const char*>(firstPointer) + math::CountTrailingZeroBits(mask);

            AdvanceBytes(firstPointer, _Traits_::portionSize);
        } while (firstPointer != stopAt);
    }

    return (firstPointer == lastPointer) ? nullptr : FindScalar(firstPointer, lastPointer, value);
}

template <class _Type_>
simd_stl_declare_const_function simd_stl_constexpr_cxx20 const void* FindVectorized(
    const void* firstPointer,
    const void* lastPointer,
    _Type_      value) noexcept
{
    if (arch::ProcessorFeatures::AVX512F())
        return FindVectorizedInternal<FindTraits<arch::CpuFeature::AVX512F>, _Type_>(firstPointer, lastPointer, value);
    else if (arch::ProcessorFeatures::AVX2())
        return FindVectorizedInternal<FindTraits<arch::CpuFeature::AVX2>, _Type_>(firstPointer, lastPointer, value);
    else if (arch::ProcessorFeatures::SSE2())
        return FindVectorizedInternal<FindTraits<arch::CpuFeature::SSE2>, _Type_>(firstPointer, lastPointer, value);

    return FindScalar(firstPointer, lastPointer, value);
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
