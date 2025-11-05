#pragma once

#include <simd_stl/SimdStlNamespace.h>

#include <simd_stl/compatibility/Inline.h>
#include <simd_stl/compatibility/FunctionAttributes.h>

#include <simd_stl/compatibility/SimdCompatibility.h>
#include <simd_stl/arch/ProcessorFeatures.h>

#include <src/simd_stl/algorithm/AdvanceBytes.h>

#include <simd_stl/math/BitMath.h>
#include <simd_stl/numeric/BasicSimd.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline const void* _RemoveCopyScalar(
    void*       first,
    const void* last,
    void*       destination,
    _Type_      value) noexcept
{
    auto firstCasted = static_cast<_Type_*>(first);

    for (; firstCasted != last; ++firstCasted) {
        const auto currentValue = *firstCasted;

        if (currentValue != value)
            *destination++ = currentValue;
    }

    return firstCasted;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_>
simd_stl_always_inline const void* _RemoveVectorizedInternal(
    const void* first,
    const void* last,
    void*       destination,
    _Type_      value) noexcept
{
    using _SimdType_ = numeric::basic_simd<_SimdGeneration_, _Type_>;

    const auto size = ByteLength(first, last);
    const auto alignedSize = size & (~(sizeof(_SimdType_) - 1));

    if (alignedSize != 0) {
        const void* stopAt = first;
        AdvanceBytes(stopAt, alignedSize);

        const auto comparand = _SimdType_(value);

        do {
            const auto loaded   = _SimdType_::loadUnaligned(first);
            const auto compared = loaded.maskEqual(comparand);

            loaded.compressStoreUnaligned(destination, compared.unwrap());
        } while (first != stopAt);
    }

}

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline const void* _RemoveCopyVectorized(
    const void* first,
    const void* last,
    void*       destination,
    _Type_      value) noexcept
{
    if (arch::ProcessorFeatures::SSSE3())
        return _RemoveVectorizedInternal<arch::CpuFeature::SSSE3, _Type_>(first, last, value);

    return _RemoveCopyScalar<_Type_>(first, last, destination, value);
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
