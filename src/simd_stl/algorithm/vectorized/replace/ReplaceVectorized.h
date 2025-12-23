#pragma once


#include <simd_stl/numeric/BasicSimd.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <typename _Type_>
simd_stl_always_inline void simd_stl_stdcall _ReplaceScalar(
    void*           _First,
    void*           _Last,
    const _Type_    _OldValue,
    const _Type_    _NewValue) noexcept
{
    auto _Current = static_cast<_Type_*>(_First);

    for (; _Current != _Last; ++_Current)
        if (*_Current == _OldValue)
            *_Current = _NewValue;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_>
simd_stl_always_inline void simd_stl_stdcall _ReplaceVectorizedInternal(
    void*           _First,
    void*           _Last,
    const _Type_    _OldValue,
    const _Type_    _NewValue) noexcept
{
    using _SimdType_ = numeric::simd<_SimdGeneration_, _Type_>;
    numeric::zero_upper_at_exit_guard<_SimdGeneration_> _Guard;

    constexpr auto _Is_masked_store_supported = _SimdType_::template is_native_mask_store_supported_v<>;
    constexpr auto _Is_masked_memory_access_supported = _Is_masked_store_supported &&
        _SimdType_::template is_native_mask_load_supported_v<>;

    const auto _Size        = ByteLength(_First, _Last);
    const auto _AlignedSize = _Size & (~(sizeof(_SimdType_) - 1));

    void* _StopAt = _First;
    AdvanceBytes(_StopAt, _AlignedSize);

    const auto _Comparand   = _SimdType_(_OldValue);
    const auto _Replacement = _SimdType_(_NewValue);

    while (_First != _StopAt) {
        const auto _Loaded      = _SimdType_::loadUnaligned(_First);
        const auto _NativeMask  = _Loaded.nativeEqual(_Comparand);

        if constexpr (_Is_masked_store_supported)
            _Replacement.maskStoreUnaligned(_First, _NativeMask);
        else
            _Replacement.maskBlendStoreUnaligned(_First, _NativeMask, _Loaded);

        AdvanceBytes(_First, sizeof(_SimdType_));
    }
    
    if constexpr (_Is_masked_memory_access_supported) {
        const auto _TailSize = _Size & (sizeof(_SimdType_) - sizeof(_Type_));

        if (_TailSize != 0) {
            const auto _TailMask    = _SimdType_::makeTailMask(_TailSize);
            const auto _Loaded      = _SimdType_::maskLoadUnaligned(_First, _TailMask);

            const auto _Mask                = _Loaded.nativeEqual(_Comparand);
            const auto _MaskForNativeStore  = numeric::_SimdConvertToMaskForNativeStore<_SimdGeneration_,
                typename _SimdType_::policy_type, _Type_>(_Mask);

            const auto _StoreMask   = _MaskForNativeStore & _TailMask;
            _Replacement.maskStoreUnaligned(_First, _StoreMask);
        }
    }
    else {
        if (_First != _Last)
            _ReplaceScalar<_Type_>(_First, _Last, _OldValue, _NewValue);
    }
}

template <typename _Type_>
void simd_stl_stdcall _ReplaceVectorized(
    void*           _First,
    void*           _Last,
    const _Type_    _OldValue,
    const _Type_    _NewValue) noexcept
{
    
    if constexpr (sizeof(_Type_) <= 2) {
        if (arch::ProcessorFeatures::AVX512BW())
            return _ReplaceVectorizedInternal<arch::CpuFeature::AVX512BW, _Type_>(_First, _Last, _OldValue, _NewValue);
    }
    else {
        if (arch::ProcessorFeatures::AVX512F())
            return _ReplaceVectorizedInternal<arch::CpuFeature::AVX512F, _Type_>(_First, _Last, _OldValue, _NewValue);
    }

    if (arch::ProcessorFeatures::AVX2())
        return _ReplaceVectorizedInternal<arch::CpuFeature::AVX2, _Type_>(_First, _Last, _OldValue, _NewValue);
    else if (arch::ProcessorFeatures::SSE41())
        return _ReplaceVectorizedInternal<arch::CpuFeature::SSE41, _Type_>(_First, _Last, _OldValue, _NewValue);
    else if (arch::ProcessorFeatures::SSE2())
        return _ReplaceVectorizedInternal<arch::CpuFeature::SSE2, _Type_>(_First, _Last, _OldValue, _NewValue);

    return _ReplaceScalar<_Type_>(_First, _Last, _OldValue, _NewValue);
}

__SIMD_STL_ALGORITHM_NAMESPACE_END

