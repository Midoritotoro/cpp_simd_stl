#pragma once

#include <src/simd_stl/algorithm/vectorized/remove/RemoveVectorized.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
simd_stl_declare_const_function void* _RemoveCopyScalar(
    const void* _First,
    const void* _Last,
    void*       _Destination,
    _Type_      _Value) noexcept
{
    auto _FirstCasted       = static_cast<const _Type_*>(_First);
    auto _DestinationCasted = static_cast<_Type_*>(_Destination);

    for (; _FirstCasted != _Last; ++_FirstCasted) {
        const auto _CurrentValue = *_FirstCasted;

        if (_CurrentValue != _Value)
            *_DestinationCasted++ = _CurrentValue;
    }

    return _DestinationCasted;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_>
simd_stl_declare_const_function void* _RemoveCopyVectorizedInternal(
    const void* _First,
    const void* _Last,
    void*       _Destination,
    _Type_      _Value) noexcept
{
    using _SimdType_ = numeric::simd<_SimdGeneration_, _Type_>;

    const auto _AlignedSize  = __byte_length(_First, _Last) & (~(sizeof(_SimdType_) - 1));

    if (_AlignedSize != 0) {
        const void* _StopAt = _First;
        __advance_bytes(_StopAt, _AlignedSize);

        const auto _Comparand = _SimdType_(_Value);

        do {
            const auto _Loaded = _SimdType_::loadUnaligned(_First);
            const auto _Mask = _Loaded.maskEqual(_Comparand);

            _Destination = _Loaded.compressStoreUnaligned(_Destination, _Mask.unwrap());
            __advance_bytes(_First, sizeof(_SimdType_));
        } while (_First != _StopAt);
    }

    return (_First == _Last) ? _Destination : _RemoveCopyScalar(_First, _Last, _Destination, _Value);
}

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline _Type_* simd_stl_stdcall _RemoveCopyVectorized(
    const void* _First,
    const void* _Last,
    void*       _Destination,
    _Type_      _Value) noexcept
{
    if constexpr (sizeof(_Type_) <= 2) {
        if (arch::ProcessorFeatures::AVX512BW())
            return static_cast<_Type_*>(_RemoveCopyVectorizedInternal<arch::CpuFeature::AVX512BW, _Type_>(
                _First, _Last, _Destination, _Value));
    }
    else {
        if (arch::ProcessorFeatures::AVX512F())
            return static_cast<_Type_*>(_RemoveCopyVectorizedInternal<arch::CpuFeature::AVX512F, _Type_>(
                _First, _Last, _Destination, _Value));
    }

    if (arch::ProcessorFeatures::AVX2())
        return static_cast<_Type_*>(
            _RemoveCopyVectorizedInternal<arch::CpuFeature::AVX2, _Type_>(_First, _Last, _Destination, _Value));
    else if (arch::ProcessorFeatures::SSSE3())
        return static_cast<_Type_*>(
            _RemoveCopyVectorizedInternal<arch::CpuFeature::SSSE3, _Type_>(_First, _Last, _Destination, _Value));

    return static_cast<_Type_*>(_RemoveCopyScalar<_Type_>(_First, _Last, _Destination, _Value));
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
