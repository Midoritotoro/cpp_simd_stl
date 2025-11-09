#pragma once

#include <simd_stl/numeric/BasicSimd.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
simd_stl_always_inline void __ReverseScalar__(
    void*   firstPointer,
    void*   lastPointer) noexcept
{
    auto first  = static_cast<_Type_*>(firstPointer);
    auto last   = static_cast<_Type_*>(lastPointer);

    for (; first != last && first != --last; ++first) {
        _Type_ temp = *last;

        *last = *first;
        *first = temp;
    }
        
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_>
simd_stl_declare_const_function simd_stl_always_inline void __ReverseVectorized__(
    void* firstPointer,
    void* lastPointer) noexcept
{
    using _SimdType_ = numeric::basic_simd<_SimdGeneration_, _Type_>;

    const auto size         = ByteLength(firstPointer, lastPointer);
    const auto alignedSize  = size & (~((sizeof(_SimdType_) << 1) - 1));

    if (alignedSize != 0) {
        void* stopAt = firstPointer;
        AdvanceBytes(stopAt, alignedSize / 2);

        do {
            auto loadedBegin  = _SimdType_::loadUnaligned(firstPointer);
            auto loadedEnd    = _SimdType_::loadUnaligned(static_cast<char*>(lastPointer) - sizeof(_SimdType_));

            loadedBegin.reverse();
            loadedEnd.reverse();

            loadedBegin.storeUnaligned(static_cast<char*>(lastPointer) - sizeof(_SimdType_));
            loadedEnd.storeUnaligned(firstPointer);

            AdvanceBytes(firstPointer, sizeof(_SimdType_));
            RewindBytes(lastPointer, sizeof(_SimdType_));
        } while (firstPointer != stopAt);
    }

    if (firstPointer != lastPointer)
        __ReverseScalar__<_Type_>(firstPointer, lastPointer);
}

template <class _Type_>
void _ReverseVectorized(
    void* firstPointer,
    void* lastPointer) noexcept
{
    if (arch::ProcessorFeatures::SSE2())
        return __ReverseVectorized__<arch::CpuFeature::SSE2, _Type_>(firstPointer, lastPointer);

    __ReverseScalar__<_Type_>(firstPointer, lastPointer);
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
