#pragma once

#include <simd_stl/SimdStlNamespace.h>

#include <simd_stl/compatibility/Inline.h>
#include <simd_stl/compatibility/FunctionAttributes.h>

#include <simd_stl/compatibility/SimdCompatibility.h>
#include <simd_stl/arch/ProcessorFeatures.h>

#include <src/simd_stl/algorithm/AdvanceBytes.h>

#include <simd_stl/math/BitMath.h>
#include <simd_stl/compatibility/Inline.h>

#include <simd_stl/numeric/BasicSimd.h>


__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline const void* simd_stl_stdcall _FindLastScalar(
    const void* _First,
    const void* _Last,
    _Type_      _Value) noexcept
{
    auto _Current = static_cast<const _Type_*>(_Last);

    while (_Current != _First && *_Current != _Value)
        --_Current;

    return (_Current == _First) ? _Last : _Current;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_>
simd_stl_declare_const_function simd_stl_always_inline const void* simd_stl_stdcall _FindLastVectorizedInternal(
    const void* _First,
    const void* _Last,
    _Type_      _Value) noexcept
{
    using _SimdType_ = numeric::basic_simd<_SimdGeneration_, _Type_>;
    numeric::zero_upper_at_exit_guard<_SimdGeneration_> _Guard;

    constexpr auto _Is_masked_memory_access_supported = _SimdType_::template is_native_mask_store_supported_v<> &&
        _SimdType_::template is_native_mask_load_supported_v<>;

    const auto _Size        = ByteLength(_First, _Last);
    const auto _AlignedSize = _Size & (~(sizeof(_SimdType_) - 1));

    const void* _CachedLast = _Last;
    const auto _Comparand = _SimdType_(_Value);

    if (_AlignedSize != 0) {
        const void* _StopAt = _Last;
        RewindBytes(_StopAt, _AlignedSize);

        do {
            RewindBytes(_Last, sizeof(_SimdType_));

            const auto _Loaded  = _SimdType_::loadUnaligned(_Last);
            const auto _Mask    = _Comparand.maskEqual(_Loaded);

            if (_Mask.anyOf())
                return static_cast<const _Type_*>(_Last) + _Mask.countTrailingZeroBits();
        } while (_Last != _StopAt);
    }

    if constexpr (_Is_masked_memory_access_supported) {
        const auto _TailSize = _Size & (sizeof(_SimdType_) - sizeof(_Type_));

        if (_TailSize != 0) {
            RewindBytes(_Last, _TailSize);

            const auto _TailMask = _SimdType_::makeTailMask(_TailSize);
            const auto _Loaded = _SimdType_::maskLoadUnaligned(_Last, _TailMask);

            const auto _Compared = _Comparand.nativeEqual(_Loaded) & _TailMask;
            const auto _Mask = numeric::basic_simd_mask<_SimdGeneration_,
                _Type_>(numeric::_SimdToNativeMask<_SimdGeneration_,
                    typename _SimdType_::policy_type, std::remove_cv_t<decltype(_Compared)>>(_Compared));

            if (_Mask.anyOf())
                return static_cast<const _Type_*>(_Last) + _Mask.countTrailingZeroBits() + 1;
        }

        return _CachedLast;
    }
    else {
        if (_First == _Last)
            return _CachedLast;
        
        return _FindLastScalar(_First, _Last, _Value);
    }
}

template <class _Type_>
simd_stl_declare_const_function simd_stl_always_inline _Type_* simd_stl_stdcall _FindLastVectorized(
    const void* _First,
    const void* _Last,
    _Type_      _Value) noexcept
{
    if constexpr (sizeof(_Type_) <= 2) {
        if (arch::ProcessorFeatures::AVX512BW())
            return const_cast<_Type_*>(static_cast<const _Type_*>(
                _FindLastVectorizedInternal<arch::CpuFeature::AVX512BW, _Type_>(_First, _Last, _Value)));
    }
    else {
        if (arch::ProcessorFeatures::AVX512F())
            return const_cast<_Type_*>(static_cast<const _Type_*>(
                _FindLastVectorizedInternal<arch::CpuFeature::AVX512F, _Type_>(_First, _Last, _Value)));
    }

    if (arch::ProcessorFeatures::AVX2())
        return const_cast<_Type_*>(static_cast<const _Type_*>(
            _FindLastVectorizedInternal<arch::CpuFeature::AVX2, _Type_>(_First, _Last, _Value)));
    else if (arch::ProcessorFeatures::SSE2())
        return const_cast<_Type_*>(static_cast<const _Type_*>(
            _FindLastVectorizedInternal<arch::CpuFeature::SSE2, _Type_>(_First, _Last, _Value)));

    return const_cast<_Type_*>(static_cast<const _Type_*>(_FindLastScalar(_First, _Last, _Value)));
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
