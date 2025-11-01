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
simd_stl_declare_const_function simd_stl_always_inline const void* _RemoveScalar(
    const void*     firstPointer,
    const void*     lastPointer,
    _Type_          value) noexcept
{
    auto pointer = static_cast<const _Type_*>(firstPointer);

    while (pointer != lastPointer && *pointer != value)
        ++pointer;

    return (pointer == lastPointer) ? lastPointer : pointer;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_>
simd_stl_declare_const_function simd_stl_always_inline const void* _RemoveVectorizedInternal(
    void*       first,
    const void* last,
    _Type_      value) noexcept
{
    using _SimdType_ = numeric::basic_simd<_SimdGeneration_, _Type_>;

    const auto size         = ByteLength(first, last);
    const auto alignedSize  = size & (~(sizeof(_SimdType_) - 1));

    if (alignedSize != 0) {
        void* current           = first;
        const auto comparand    = _SimdType_(value);

        const void* stopAt = first;
        AdvanceBytes(stopAt, alignedSize);

        do {
            const auto loaded = _SimdType_::loadUnaligned(current);
            const auto mask = comparand.maskEqual<_Type_>(loaded);

           /* loaded.compressStore(first, mask);*/
            AdvanceBytes(current, sizeof(_SimdType_));
        } while (current != stopAt);
    }

    return (first == last) ? last : FindScalar(first, last, value);
}

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline const void* _RemoveVectorized(
    void*       first,
    const void* last,
    _Type_      value) noexcept
{
    if (arch::ProcessorFeatures::SSE2())
        return _RemoveVectorizedInternal<arch::CpuFeature::SSE2, _Type_>(first, last, value);

    return _RemoveScalar(first, last, value);
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
