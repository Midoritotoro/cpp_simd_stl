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
    using _SimdType_ = numeric::basic_simd<_SimdGeneration_, _Type_>;

    const auto _AlignedSize = ByteLength(_First, _Last) & (~(sizeof(_SimdType_) - 1));

    if (_AlignedSize != 0) {
        void* _StopAt = _First;
        AdvanceBytes(_StopAt, _AlignedSize);

        const auto _Comparand   = _SimdType_(_OldValue);
        const auto _Replacement = _SimdType_(_NewValue);

        do {
            const auto _Loaded = _SimdType_::loadUnaligned(_First);
            const auto _NativeMask = _Loaded.nativeEqual(_Comparand);

            _Replacement.maskBlendStoreUnaligned(_First, _NativeMask, _Loaded);
            AdvanceBytes(_First, sizeof(_SimdType_));
        } while (_First != _StopAt);
    }

    if (_First == _Last)
        _ReplaceScalar(_First, _Last, _OldValue, _NewValue);
}

template <typename _Type_>
void simd_stl_stdcall _ReplaceVectorized(
    void*           _First,
    void*           _Last,
    const _Type_    _OldValue,
    const _Type_    _NewValue) noexcept
{
    if (sizeof(_Type_) <= 2) {
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

