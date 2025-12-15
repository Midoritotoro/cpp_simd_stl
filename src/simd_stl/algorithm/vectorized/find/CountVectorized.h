#pragma once

#include <simd_stl/numeric/BasicSimd.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline sizetype simd_stl_stdcall _CountScalar(
    const void*     _FirstPointer,
    const sizetype  _Bytes,
    sizetype&       _Count,
    _Type_          _Value) noexcept
{
    auto _Current = static_cast<const _Type_*>(_FirstPointer);
    const auto _Length = _Bytes / sizeof(_Type_);

    for (sizetype _Index = 0; _Index < _Length; ++_Index)
        _Count += (*_Current++ == _Value);

    return _Count;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_>
simd_stl_declare_const_function simd_stl_always_inline sizetype simd_stl_stdcall _CountVectorizedInternal(
    const void*     _First,
    const sizetype  _Bytes,
    _Type_          _Value) noexcept
{
    using _SimdType_    = numeric::basic_simd<_SimdGeneration_, _Type_>;
    numeric::zero_upper_at_exit_guard<_SimdGeneration_> _Guard;

    constexpr auto _Is_masked_memory_access_supported = _SimdType_::template is_native_mask_store_supported_v<> &&
        _SimdType_::template is_native_mask_load_supported_v<>;

    auto _AlignedSize   = _Bytes & (~(sizeof(_SimdType_) - 1));
    sizetype _Count = 0;

    const auto _Comparand = _SimdType_(_Value);

    const void* _StopAt = _First; 
    AdvanceBytes(_StopAt, _AlignedSize);

    while (_First != _StopAt) {
        const auto _Loaded   = _SimdType_::loadUnaligned(_First);
        const auto _Compared = _Comparand.maskEqual(_Loaded);

        _Count += _Compared.countSet();
        AdvanceBytes(_First, sizeof(_SimdType_));
    }
   
    if constexpr (_Is_masked_memory_access_supported) {
        const auto _TailSize = _Bytes & (sizeof(_SimdType_) - sizeof(_Type_));

        if (_TailSize != 0) {
            const auto _TailMask = _SimdType_::makeTailMask(_TailSize);
            const auto _Loaded = _SimdType_::maskLoadUnaligned(_First, _TailMask);

            const auto _Compared = _Comparand.nativeEqual(_Loaded) & _TailMask;
            const auto _Mask = numeric::basic_simd_mask<_SimdGeneration_, _Type_>(
                numeric::_SimdToNativeMask<_SimdGeneration_, typename _SimdType_::policy_type, 
                std::remove_cv_t<decltype(_Compared)>>(_Compared));

            _Count += _Mask.countSet();
        }

        return _Count;
    }
    else {
        return _CountScalar(_First, (_Bytes - _AlignedSize), _Count, _Value);
    }
}

template <class _Type_>
simd_stl_declare_const_function sizetype simd_stl_stdcall _CountVectorized(
    const void*     _FirstPointer,
    const sizetype  _Bytes,
    _Type_          _Value) noexcept
{
    if constexpr (sizeof(_Type_) <= 2) {
        if (arch::ProcessorFeatures::AVX512BW())
            return _CountVectorizedInternal<arch::CpuFeature::AVX512BW, _Type_>(_FirstPointer, _Bytes, _Value);
    }
    else {
        if (arch::ProcessorFeatures::AVX512F())
            return _CountVectorizedInternal<arch::CpuFeature::AVX512F, _Type_>(_FirstPointer, _Bytes, _Value);
    }

    if (arch::ProcessorFeatures::AVX2())
        return _CountVectorizedInternal<arch::CpuFeature::AVX2, _Type_>(_FirstPointer, _Bytes, _Value);
    else if (arch::ProcessorFeatures::SSE2())
        return _CountVectorizedInternal<arch::CpuFeature::SSE2, _Type_>(_FirstPointer, _Bytes, _Value);

    sizetype _Count = 0;
    return _CountScalar(_FirstPointer, _Bytes, _Count, _Value);
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
