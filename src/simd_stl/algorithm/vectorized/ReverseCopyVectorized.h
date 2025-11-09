#pragma once

#include <simd_stl/numeric/BasicSimd.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
simd_stl_always_inline void __ReverseCopyScalar__(
    const void* firstPointer,
    const void* lastPointer,
    void*       destinationPointer,
    sizetype    bytes) noexcept
{
    auto first = static_cast<const _Type_*>(firstPointer);
    auto last = static_cast<const _Type_*>(lastPointer);

    auto destination = static_cast<_Type_*>(destinationPointer);

    for (; first != last; ++destination)
        *destination = *--last;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_>
simd_stl_always_inline void __ReverseCopyVectorized__(
    const void* firstPointer,
    const void* lastPointer,
    void*       destination) noexcept
{
    using _SimdType_ = numeric::basic_simd<_SimdGeneration_, _Type_>;

    const auto size = ByteLength(firstPointer, lastPointer);
    const auto alignedSize = size & (~((sizeof(_SimdType_)) - 1));

    if (alignedSize != 0) {
        const void* stopAt = lastPointer;
        RewindBytes(stopAt, alignedSize);

        do {
            auto loaded = _SimdType_::loadUnaligned(static_cast<const char*>(lastPointer) - sizeof(_SimdType_));
            loaded.reverse();
            loaded.storeUnaligned(destination);

            AdvanceBytes(destination, sizeof(_SimdType_));
            RewindBytes(lastPointer, sizeof(_SimdType_));
        } while (lastPointer != stopAt);
    }

    if (firstPointer != lastPointer)
        __ReverseCopyScalar__<_Type_>(firstPointer, lastPointer, destination, size);
}

template <class _Type_>
void _ReverseCopyVectorized(
    const void* firstPointer,
    const void* lastPointer,
    void*       destination) noexcept
{
    if (arch::ProcessorFeatures::SSSE3())
        return __ReverseCopyVectorized__<arch::CpuFeature::SSSE3, _Type_>(firstPointer, lastPointer, destination);

    __ReverseCopyScalar__<_Type_>(firstPointer, lastPointer, destination, ByteLength(firstPointer, lastPointer));
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
