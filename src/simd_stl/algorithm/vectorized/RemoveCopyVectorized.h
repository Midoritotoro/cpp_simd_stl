#pragma once

#include <src/simd_stl/algorithm/vectorized/RemoveVectorized.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline void* _RemoveCopyScalar(
    const void* first,
    const void* last,
    void*       destination,
    _Type_      value) noexcept
{
    auto firstCasted = static_cast<const _Type_*>(first);
    auto destinationCasted = static_cast<_Type_*>(destination);

    for (; firstCasted != last; ++firstCasted) {
        const auto currentValue = *firstCasted;

        if (currentValue != value)
            *destinationCasted++ = currentValue;
    }

    return destinationCasted;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_>
simd_stl_always_inline void* _RemoveCopyVectorizedInternal(
    const void* first,
    const void* last,
    void*       destination,
    _Type_      value) noexcept
{
    using _SimdType_ = numeric::basic_simd<_SimdGeneration_, _Type_>;

    constexpr auto step     = removeStep<_SimdType_>();

    const auto size         = ByteLength(first, last);
    const auto alignedSize  = size & (~(step - 1));

    if (alignedSize != 0) {
        const void* stopAt = first;
        AdvanceBytes(stopAt, alignedSize);

        const auto comparand = _SimdType_(value);

        do {
            _SimdType_ loaded;
            
            if constexpr (numeric::is_epi8_v<_Type_> || numeric::is_epu8_v<_Type_>)
                loaded = _SimdType_::loadLowerHalf(first);
            else
                loaded = _SimdType_::loadUnaligned(first);

            const auto compared = loaded.maskEqual(comparand);

            destination = loaded.compressStoreUnaligned(destination, compared.unwrap());
            AdvanceBytes(first, step);
        } while (first != stopAt);
    }

    return (first == last) ? destination : _RemoveCopyScalar(first, last, destination, value);
}

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline void* _RemoveCopyVectorized(
    const void* first,
    const void* last,
    void*       destination,
    _Type_      value) noexcept
{
    if (arch::ProcessorFeatures::SSSE3())
        return _RemoveCopyVectorizedInternal<arch::CpuFeature::SSSE3, _Type_>(first, last, destination, value);

    return _RemoveCopyScalar<_Type_>(first, last, destination, value);
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
