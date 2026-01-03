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

template <typename _Type_> 
simd_stl_declare_const_function simd_stl_always_inline bool simd_stl_stdcall _EqualScalar(
    const void* _First,
    const void* _Second,
    sizetype    _Size) noexcept
{
    const _Type_* _FirstPointer    = static_cast<const _Type_*>(_First);
    const _Type_* _SecondPointer   = static_cast<const _Type_*>(_Second);

    while (_Size--)
        if (*_FirstPointer++ != *_SecondPointer++)
            return false;

    return true;
}

template <
    arch::CpuFeature    _SimdGeneration_,
    typename            _Type_>
simd_stl_declare_const_function simd_stl_always_inline bool simd_stl_stdcall _EqualVectorizedInternal(
    const void*     _First,
    const void*     _Second,
    const sizetype  _Length) noexcept
{
    using _SimdType_ = numeric::simd<_SimdGeneration_, _Type_>;
    numeric::zero_upper_at_exit_guard<_SimdGeneration_> _Guard;

    constexpr auto _Is_masked_memory_access_supported = _SimdType_::template is_native_mask_store_supported_v<> &&
        _SimdType_::template is_native_mask_load_supported_v<>;

    const auto _Size            = _Length * sizeof(_Type_);
    const auto _AlignedSize     = (_Size & (~(sizeof(_SimdType_) - 1)));
 
    if (_AlignedSize != 0) {
        const void* _StopAt = _First;
        __advance_bytes(_StopAt, _AlignedSize);

        do {
            const auto _LoadedFirst = _SimdType_::loadUnaligned(_First);
            const auto _LoadedSecond = _SimdType_::loadUnaligned(_Second);

            const auto _Mask = _LoadedFirst.maskEqual(_LoadedSecond);

            if (_Mask.allOf() == false)
                return false;

            __advance_bytes(_First, sizeof(_SimdType_));
            __advance_bytes(_Second, sizeof(_SimdType_));
        } while (_First != _StopAt);
    }

    const auto _TailSize = _Size & (sizeof(_SimdType_) - sizeof(_Type_));

    if (_TailSize == 0)
        return true;

    if constexpr (_Is_masked_memory_access_supported) {
        const auto _TailMask = _SimdType_::makeTailMask(_TailSize);
            
        const auto _LoadedFirst = _SimdType_::maskLoadUnaligned(_First, _TailMask);
        const auto _LoadedSecond = _SimdType_::maskLoadUnaligned(_Second, _TailMask);
            
        const auto _Compared = _LoadedFirst.nativeEqual(_LoadedSecond) & _TailMask;
        const auto _Mask = numeric::simd_mask<_SimdGeneration_,
            _Type_>(numeric::_SimdToNativeMask<_SimdGeneration_,
                typename _SimdType_::policy_type, std::remove_cv_t<decltype(_Compared)>>(_Compared));

        const auto _AllEqualMask = (1u << (_TailSize / sizeof(_Type_))) - 1;
        return (_Mask == _AllEqualMask);
    }
    else {
        return _EqualScalar<_Type_>(_First, _Second, (_Size - _AlignedSize) / sizeof(_Type_));
    }
}


template <typename _Type_>
simd_stl_declare_const_function bool simd_stl_stdcall _EqualVectorized(
    const void*     _First,
    const void*     _Second,
    const sizetype  _Size) noexcept
{
    if constexpr (sizeof(_Type_) <= 2) {
        if (arch::ProcessorFeatures::AVX512BW())
            return _EqualVectorizedInternal<arch::CpuFeature::AVX512BW, _Type_>(_First, _Second, _Size);
    }
    else {
        if (arch::ProcessorFeatures::AVX512F())
            return _EqualVectorizedInternal<arch::CpuFeature::AVX512F, _Type_>(_First, _Second, _Size);
    }

    if (arch::ProcessorFeatures::AVX2())
        return _EqualVectorizedInternal<arch::CpuFeature::AVX2, _Type_>(_First, _Second, _Size);
    else if (arch::ProcessorFeatures::SSE2())
        return _EqualVectorizedInternal<arch::CpuFeature::SSE2, _Type_>(_First, _Second, _Size);

    return _EqualScalar<_Type_>(_First, _Second, _Size * sizeof(_Type_));
}

__SIMD_STL_ALGORITHM_NAMESPACE_END
