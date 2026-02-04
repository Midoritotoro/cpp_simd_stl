#pragma once

#include <simd_stl/datapar/BasicSimd.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline _Type_ _MaxScalar(
    const void* _First,
    const void* _Last) noexcept
{
    if (_First == _Last)
        return -1;

    const _Type_* _FirstCasted = static_cast<const _Type_*>(_First);
    auto _Max = _FirstCasted;

    for (; ++_FirstCasted != _Last; )
        if (*_FirstCasted > *_Max)
            _Max = _FirstCasted;

    return *_Max;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_>
simd_stl_declare_const_function simd_stl_always_inline _Type_ _MaxVectorizedInternal(
    const void*     _First,
    const void*     _Last) noexcept
{
    using _SimdType_ = datapar::simd<_SimdGeneration_, _Type_>;
    datapar::zero_upper_at_exit_guard<_SimdGeneration_> _Guard;

    constexpr auto _Is_masked_memory_access_supported = _SimdType_::template is_native_mask_store_supported_v<> &&
        _SimdType_::template is_native_mask_load_supported_v<>;

    const auto _Size        = __byte_length(_First, _Last);
    const auto _AlignedSize = _Size & (~(sizeof(_SimdType_) - 1));

    auto _MaximumValues = _SimdType_();
    _MaximumValues.broadcastZeros();

    if (_AlignedSize != 0) {
        const void* _StopAt = _First;
        __advance_bytes(_StopAt, _AlignedSize);

        _MaximumValues = _SimdType_::loadUnaligned(_First);
        __advance_bytes(_First, sizeof(_SimdType_));

        while (_First != _StopAt) {
            const auto _Loaded = _SimdType_::loadUnaligned(_First);
            _MaximumValues = _MaximumValues.verticalMax(_Loaded);

            __advance_bytes(_First, sizeof(_SimdType_));
        };
    }

    const auto _TailSize = _Size & (sizeof(_SimdType_) - sizeof(_Type_));

    if constexpr (_Is_masked_memory_access_supported) {
        if (_TailSize != 0) {
            const auto _TailMask = _SimdType_::makeTailMask(_TailSize);

            if (_AlignedSize != 0) {
                const auto _Loaded = _SimdType_::maskLoadUnaligned(_First, _TailMask);
                _MaximumValues = _MaximumValues.verticalMax(_Loaded);
            }
            else {
                _MaximumValues = _SimdType_::maskLoadUnaligned(_First, _TailMask);
            }
        }
    }
    else {
        if (_AlignedSize != 0) {
            const auto _Max = _MaxScalar<_Type_>(_First, _Last);
            const auto _HorizontalMax = _MaximumValues.horizontalMax();

            return (_Max > _HorizontalMax) ? _Max : _HorizontalMax;
        }
        else {
            return _MaxScalar<_Type_>(_First, _Last);
        }
    }

    return _MaximumValues.horizontalMax();
}

template <class _Type_>
simd_stl_declare_const_function _Type_ simd_stl_stdcall _MaxVectorized(
    const void* _First,
    const void* _Last) noexcept
{
    if constexpr (sizeof(_Type_) <= 2) {
        if (arch::ProcessorFeatures::AVX512BW()) 
            return _MaxVectorizedInternal<arch::CpuFeature::AVX512BW, _Type_>(_First, _Last);
    }
    else {
        if (arch::ProcessorFeatures::AVX512F())
            return _MaxVectorizedInternal<arch::CpuFeature::AVX512F, _Type_>(_First, _Last);
    }

    if (arch::ProcessorFeatures::AVX2())
        return _MaxVectorizedInternal<arch::CpuFeature::AVX2, _Type_>(_First, _Last);
    else if (arch::ProcessorFeatures::SSE41())
        return _MaxVectorizedInternal<arch::CpuFeature::SSE41, _Type_>(_First, _Last);
    else if (arch::ProcessorFeatures::SSSE3())
        return _MaxVectorizedInternal<arch::CpuFeature::SSSE3, _Type_>(_First, _Last);
    else if (arch::ProcessorFeatures::SSE2())
        return _MaxVectorizedInternal<arch::CpuFeature::SSE2, _Type_>(_First, _Last);

    return _MaxScalar<_Type_>(_First, _Last);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END