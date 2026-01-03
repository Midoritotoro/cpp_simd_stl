#pragma once


#include <simd_stl/numeric/BasicSimd.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <typename _Type_>
simd_stl_always_inline void simd_stl_stdcall _ReplaceCopyScalar(
    const void*     _First,
    const void*     _Last,
    void*           _Destination,
    const _Type_    _OldValue,
    const _Type_    _NewValue) noexcept
{
    auto _Current               = static_cast<const _Type_*>(_First);
    auto _DestinationCurrent    = static_cast<_Type_*>(_Destination);

    for (; _Current != _Last; ++_Current, ++_DestinationCurrent)
        *_DestinationCurrent = (*_Current == _OldValue) ? _NewValue : (*_Current);
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_>
simd_stl_always_inline void simd_stl_stdcall _ReplaceCopyVectorizedInternal(
    const void*     _First,
    const void*     _Last,
    void*           _Destination,
    const _Type_    _OldValue,
    const _Type_    _NewValue) noexcept
{
    using _SimdType_ = numeric::simd<_SimdGeneration_, _Type_>;
    numeric::zero_upper_at_exit_guard<_SimdGeneration_> _Guard;

    constexpr auto _Is_masked_memory_access_supported = _SimdType_::template is_native_mask_store_supported_v<> &&
        _SimdType_::template is_native_mask_load_supported_v<>;

    const auto _Size        = __byte_length(_First, _Last);
    const auto _AlignedSize = _Size & (~(sizeof(_SimdType_) - 1));

    const void* _StopAt = _First;
    __advance_bytes(_StopAt, _AlignedSize);

    const auto _Comparand   = _SimdType_(_OldValue);
    const auto _Replacement = _SimdType_(_NewValue);

    while (_First != _StopAt) {
        const auto _Loaded      = _SimdType_::loadUnaligned(_First);
        const auto _NativeMask  = _Loaded.nativeEqual(_Comparand);

        _Replacement.maskBlendStoreUnaligned(_Destination, _NativeMask, _Loaded);

        __advance_bytes(_First, sizeof(_SimdType_));
        __advance_bytes(_Destination, sizeof(_SimdType_));
    }
    
    if constexpr (_Is_masked_memory_access_supported) {
        const auto _TailSize = _Size & (sizeof(_SimdType_) - sizeof(_Type_));

        if (_TailSize != 0) {
            const auto _TailMask    = _SimdType_::makeTailMask(_TailSize);
            auto _Loaded            = _SimdType_::maskLoadUnaligned(_First, _TailMask);

            const auto _Mask                = _Loaded.nativeEqual(_Comparand);
            const auto _MaskForNativeStore  = numeric::_SimdConvertToMaskForNativeStore<_SimdGeneration_,
                typename _SimdType_::policy_type, _Type_>(_Mask);

            const auto _StoreMask   = _MaskForNativeStore & _TailMask;

            _Loaded.blend(_Replacement, ~_StoreMask);
            _Loaded.maskStoreUnaligned(_Destination, _StoreMask);
        }
    }
    else {
        if (_First != _Last)
            _ReplaceCopyScalar<_Type_>(_First, _Last, _Destination, _OldValue, _NewValue);
    }
}

template <typename _Type_>
void simd_stl_stdcall _ReplaceCopyVectorized(
    const void*     _First,
    const void*     _Last,
    void*           _Destination,
    const _Type_    _OldValue,
    const _Type_    _NewValue) noexcept
{
    
    if constexpr (sizeof(_Type_) <= 2) {
        if (arch::ProcessorFeatures::AVX512BW())
            return _ReplaceCopyVectorizedInternal<arch::CpuFeature::AVX512BW, _Type_>(_First, _Last, _Destination, _OldValue, _NewValue);
    }
    else {
        if (arch::ProcessorFeatures::AVX512F())
            return _ReplaceCopyVectorizedInternal<arch::CpuFeature::AVX512F, _Type_>(_First, _Last, _Destination, _OldValue, _NewValue);
    }

    if (arch::ProcessorFeatures::AVX2())
        return _ReplaceCopyVectorizedInternal<arch::CpuFeature::AVX2, _Type_>(_First, _Last, _Destination, _OldValue, _NewValue);
    else if (arch::ProcessorFeatures::SSE41())
        return _ReplaceCopyVectorizedInternal<arch::CpuFeature::SSE41, _Type_>(_First, _Last, _Destination, _OldValue, _NewValue);
    else if (arch::ProcessorFeatures::SSE2())
        return _ReplaceCopyVectorizedInternal<arch::CpuFeature::SSE2, _Type_>(_First, _Last, _Destination, _OldValue, _NewValue);

    return _ReplaceCopyScalar<_Type_>(_First, _Last, _Destination, _OldValue, _NewValue);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END

