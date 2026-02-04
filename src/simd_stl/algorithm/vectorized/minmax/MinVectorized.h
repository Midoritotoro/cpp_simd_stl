#pragma once

#include <simd_stl/datapar/BasicSimd.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline _Type_ _MinScalar(
    const void* _First,
    const void* _Last) noexcept
{
    if (_First == _Last)
        return -1;

    const _Type_* _FirstCasted = static_cast<const _Type_*>(_First);
    auto _Min = _FirstCasted;

    for (; ++_FirstCasted != _Last; )
        if (*_FirstCasted < *_Min)
            _Min = _FirstCasted;

    return *_Min;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_>
simd_stl_declare_const_function simd_stl_always_inline _Type_ _MinVectorizedInternal(
    const void*     _First,
    const void*     _Last) noexcept
{
    using _SimdType_ = datapar::simd<_SimdGeneration_, _Type_>;
    datapar::zero_upper_at_exit_guard<_SimdGeneration_> _Guard;

    constexpr auto _Is_masked_memory_access_supported = _SimdType_::template is_native_mask_store_supported_v<> &&
        _SimdType_::template is_native_mask_load_supported_v<>;

    const auto _Size        = __byte_length(_First, _Last);
    const auto _AlignedSize = _Size & (~(sizeof(_SimdType_) - 1));

    auto _MinimumValues = _SimdType_();
    _MinimumValues.broadcastZeros();

    if (_AlignedSize != 0) {
        const void* _StopAt = _First;
        __advance_bytes(_StopAt, _AlignedSize);

        _MinimumValues = _SimdType_::loadUnaligned(_First);
        __advance_bytes(_First, sizeof(_SimdType_));

        while (_First != _StopAt) {
            const auto _Loaded = _SimdType_::loadUnaligned(_First);
            _MinimumValues = _MinimumValues.verticalMin(_Loaded);

            __advance_bytes(_First, sizeof(_SimdType_));
        };
    }

    const auto _TailSize = _Size & (sizeof(_SimdType_) - sizeof(_Type_));

    if constexpr (_Is_masked_memory_access_supported) {
        if (_TailSize != 0) {
            const auto _TailMask = _SimdType_::makeTailMask(_TailSize);

            if (_AlignedSize != 0) {
                const auto _Loaded = _SimdType_::maskLoadUnaligned(_First, _TailMask);
                _MinimumValues = _MinimumValues.verticalMin(_Loaded);
            }
            else {
                _MinimumValues = _SimdType_::maskLoadUnaligned(_First, _TailMask);
            }
        }
    }
    else {
        if (_AlignedSize != 0) {
            const auto _Min = _MinScalar<_Type_>(_First, _Last);
            const auto _HorizontalMin = _MinimumValues.horizontalMin();

            return (_Min < _HorizontalMin) ? _Min : _HorizontalMin;
        }
        else {
            return _MinScalar<_Type_>(_First, _Last);
        }
    }

    return _MinimumValues.horizontalMin();
}

template <class _Type_>
simd_stl_declare_const_function _Type_ simd_stl_stdcall _MinVectorized(
    const void* _First,
    const void* _Last) noexcept
{
    if constexpr (sizeof(_Type_) <= 2) {
        if (arch::ProcessorFeatures::AVX512BW()) 
            return _MinVectorizedInternal<arch::CpuFeature::AVX512BW, _Type_>(_First, _Last);
    }
    else {
        if (arch::ProcessorFeatures::AVX512F())
            return _MinVectorizedInternal<arch::CpuFeature::AVX512F, _Type_>(_First, _Last);
    }

    if (arch::ProcessorFeatures::AVX2())
        return _MinVectorizedInternal<arch::CpuFeature::AVX2, _Type_>(_First, _Last);
    else if (arch::ProcessorFeatures::SSE41())
        return _MinVectorizedInternal<arch::CpuFeature::SSE41, _Type_>(_First, _Last);
    else if (arch::ProcessorFeatures::SSSE3())
        return _MinVectorizedInternal<arch::CpuFeature::SSSE3, _Type_>(_First, _Last);
    else if (arch::ProcessorFeatures::SSE2())
        return _MinVectorizedInternal<arch::CpuFeature::SSE2, _Type_>(_First, _Last);

    return _MinScalar<_Type_>(_First, _Last);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END