#pragma once

#include <simd_stl/numeric/BasicSimd.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
void simd_stl_stdcall _ReverseCopyScalar(
    const void* _First,
    const void* _Last,
    void*       _Destination) noexcept
{
    auto _FirstPointer = static_cast<const _Type_*>(_First);
    auto _LastPointer = static_cast<const _Type_*>(_Last);

    auto _DestinationPointer = static_cast<_Type_*>(_Destination);

    for (; _FirstPointer != _LastPointer; ++_DestinationPointer)
        *_DestinationPointer = *--_LastPointer;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_>
void simd_stl_stdcall _ReverseCopyVectorized(
    const void* _First,
    const void* _Last,
    void*       _Destination) noexcept
{
    using _SimdType_ = numeric::simd<_SimdGeneration_, _Type_>;
    numeric::zero_upper_at_exit_guard<_SimdGeneration_> _Guard;

    const auto _AlignedSize = __byte_length(_First, _Last) & (~((sizeof(_SimdType_)) - 1));

    if (_AlignedSize != 0) {
        const void* _StopAt = _Last;
        __rewind_bytes(_StopAt, _AlignedSize);

        do {
            auto _Loaded = _SimdType_::loadUnaligned(static_cast<const char*>(_Last) - sizeof(_SimdType_));
            _Loaded.reverse();
            _Loaded.storeUnaligned(_Destination);

            __advance_bytes(_Destination, sizeof(_SimdType_));
            __rewind_bytes(_Last, sizeof(_SimdType_));
        } while (_Last != _StopAt);
    }

    if (_First != _Last)
        _ReverseCopyScalar<_Type_>(_First, _Last, _Destination);
}

template <class _Type_>
void simd_stl_stdcall _ReverseCopyVectorized(
    const void* _First,
    const void* _Last,
    void*       _Destination) noexcept
{
    if constexpr (sizeof(_Type_) == 2) {
        if (arch::ProcessorFeatures::AVX512BW())
            return _ReverseCopyVectorized<arch::CpuFeature::AVX512BW, _Type_>(_First, _Last, _Destination);
    }
    else {
        if (arch::ProcessorFeatures::AVX512F())
            return _ReverseCopyVectorized<arch::CpuFeature::AVX512F, _Type_>(_First, _Last, _Destination);
    }

    if (arch::ProcessorFeatures::AVX2())
        return _ReverseCopyVectorized<arch::CpuFeature::AVX2, _Type_>(_First, _Last, _Destination);
    else if (arch::ProcessorFeatures::SSSE3())
        return _ReverseCopyVectorized<arch::CpuFeature::SSSE3, _Type_>(_First, _Last, _Destination);

    _ReverseCopyScalar<_Type_>(_First, _Last, _Destination);
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
