#pragma once

#include <simd_stl/SimdStlNamespace.h>

#include <simd_stl/compatibility/Inline.h>
#include <simd_stl/compatibility/FunctionAttributes.h>

#include <simd_stl/compatibility/SimdCompatibility.h>
#include <simd_stl/arch/ProcessorFeatures.h>

#include <src/simd_stl/algorithm/AdvanceBytes.h>

#include <simd_stl/math/BitMath.h>

#include <simd_stl/numeric/BasicSimd.h>

__SIMD_STL_ALGORITHM_NAMESPACE_BEGIN

template <class _Type_>
simd_stl_declare_const_function bool simd_stl_stdcall _ContainsScalar(
    const void* _First,
    const void* _Last,
    _Type_      _Value) noexcept
{
    auto _Current = static_cast<const _Type_*>(_First);

    while (_Current != _Last) {
        if (*_Current == _Value)
            return true;

        ++_Current;
    }

    return false;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_>
simd_stl_declare_const_function bool simd_stl_stdcall _ContainsVectorizedInternal(
    const void* _First,
    const void* _Last,
    _Type_      _Value) noexcept
{
    using _SimdType_ = numeric::basic_simd<_SimdGeneration_, _Type_>;
    numeric::zero_upper_at_exit_guard<_SimdGeneration_> _Guard;

    constexpr auto _Is_masked_memory_access_supported = _SimdType_::template is_native_mask_store_supported_v<> &&
        _SimdType_::template is_native_mask_load_supported_v<>;

    const auto _Size  = ByteLength(_First, _Last);
    auto _AlignedSize = _Size & (~(sizeof(_SimdType_) - 1));

    const auto _Comparand = _SimdType_(_Value);

    while (_AlignedSize != 0) {
        const auto _Mask = _Comparand.maskEqual<_Type_>(_SimdType_::loadUnaligned(_First));

        if (_Mask.anyOf())
            return true;

        AdvanceBytes(_First, sizeof(_SimdType_));
        _AlignedSize -= sizeof(_SimdType_);
    }

    if constexpr (_Is_masked_memory_access_supported) {
        const auto _TailSize = _Size & (sizeof(_SimdType_) - sizeof(_Type_));

        if (_TailSize != 0) {
            const auto _TailMask = _SimdType_::makeTailMask(_TailSize);
            const auto _Loaded = _SimdType_::maskLoadUnaligned(_First, _TailMask);

            const auto _Compared = _Comparand.nativeEqual(_Loaded) & _TailMask;
            const auto _Mask = numeric::basic_simd_mask<_SimdGeneration_,
                _Type_>(numeric::_SimdToNativeMask<_SimdGeneration_,
                    typename _SimdType_::policy_type, std::remove_cv_t<decltype(_Compared)>>(_Compared));

            if (_Mask.anyOf())
                return true;
        }
    }
    else {
        if (_First != _Last)
            return _ContainsScalar(_FirstPointer, _LastPointer, _Value);
    }
}

template <class _Type_>
simd_stl_declare_const_function bool simd_stl_stdcall _ContainsVectorized(
    const void* _First,
    const void* _Last,
    _Type_      _Value) noexcept
{
    if constexpr (sizeof(_Type_) <= 2) {
        if (arch::ProcessorFeatures::AVX512BW())
            return _ContainsVectorizedInternal<arch::CpuFeature::AVX512BW, _Type_>(_First, _Last, _Value);
    }
    else {
        if (arch::ProcessorFeatures::AVX512F())
            return _ContainsVectorizedInternal<arch::CpuFeature::AVX512F, _Type_>(_First, _Last, _Value);
    }

    if (arch::ProcessorFeatures::AVX2())
         return _ContainsVectorizedInternal<arch::CpuFeature::AVX2, _Type_>(_First, _Last, _Value);
    else if (arch::ProcessorFeatures::SSE2())
        return _ContainsVectorizedInternal<arch::CpuFeature::SSE2, _Type_>(_First, _Last, _Value);

    return _ContainsScalar(_First, _Last, _Value);
}


__SIMD_STL_ALGORITHM_NAMESPACE_END
